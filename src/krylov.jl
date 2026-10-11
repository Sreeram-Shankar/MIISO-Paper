function instability_interpolation(x, xp, yp)
    x <= xp[1] && return yp[1]
    x >= xp[end] && return yp[end]
    i = searchsortedlast(xp, x)
    t = (x - xp[i]) / (xp[i + 1] - xp[i])
    return (1 - t) * yp[i] + t * yp[i + 1]
end

#mag–arg blend of two stability functions (used to repair unstable Padé islands)
function blend_RD(r0, r1, th)
    th <= 0 && return r0
    th >= 1 && return r1
    mag = (abs(r1)^th) * (abs(r0)^(1 - th))
    arg = (1 - th) * angle(r0) + th * angle(r1)
    return mag * cis(arg)
end

#per-mode blend of conservative R_C and dissipative R_D across the taper band [omega0, omega_c]
function R_blend(z_vals, omega0, omega_c, R_C, R_D; p::Int=4)
    scalar = z_vals isa Number
    z = ComplexF64.(scalar ? [z_vals] : vec(z_vals))
    isempty(z) && return scalar ? 0.0im : ComplexF64[]
    ω0, ωc = float(omega0), float(omega_c)
    θ = map(z) do ζ
        w = abs(ζ)
        w <= ω0 ? 0.0 : w >= ωc ? 1.0 : ((w - ω0) / (ωc - ω0))^p
    end
    out = similar(z)
    mid = Int[]
    @inbounds for i in eachindex(z)
        if θ[i] <= 0
            out[i] = ComplexF64(R_C(z[i]))
        elseif θ[i] >= 1
            out[i] = ComplexF64(R_D(z[i]))
        else
            push!(mid, i)
        end
    end
    if !isempty(mid)
        zm, thm = z[mid], θ[mid]
        Rc, Rd = ComplexF64.(R_C.(zm)), ComplexF64.(R_D.(zm))
        order = sortperm(abs.(zm))
        arg_c, arg_d = angle.(Rc[order]), angle.(Rd[order])
        for i in 2:length(arg_c)
            d = arg_c[i] - arg_c[i - 1]; arg_c[i] = arg_c[i - 1] + (d - 2π * round(d / 2π))
            d = arg_d[i] - arg_d[i - 1]; arg_d[i] = arg_d[i - 1] + (d - 2π * round(d / 2π))
        end
        ϕc, ϕd = similar(thm), similar(thm)
        ϕc[order], ϕd[order] = arg_c, arg_d
        @. out[mid] = (abs(Rd)^thm) * (abs(Rc)^(1 - thm)) * cis((1 - thm) * ϕc + thm * ϕd)
    end
    return scalar ? out[1] : out
end


#Taylor coefficients r_0, r_1, ... of the conservative parent R_C about z = 0, computed once per
#R_C by a trapezoidal Cauchy integral on |z| = ρ (the Padé poles lie well outside this circle).
const _rc_series = IdDict{Any,Vector{ComplexF64}}()
const _rc_lock = ReentrantLock()
function rc_series(R_C; nterms::Int=30, ρ=1.0, N::Int=128)
    lock(_rc_lock) do
        get!(_rc_series, R_C) do
            θ = [2π * (q - 1) / N for q in 1:N]
            vals = [ComplexF64(R_C(ρ * cis(t))) for t in θ]
            [sum(vals[q] * cis(-j * θ[q]) for q in 1:N) / (N * ρ^j) for j in 0:nterms-1]
        end
    end
end

#phi_k of the blended map, any k >= 1.
#Away from 0: the (R-1)/z recurrence φ̃_{j+1} = (φ̃_j - 1/j!)/z.
#Near 0 (|z| < zsmall and below the taper, where R = R_C exactly): the Taylor series of R_C,
#φ̃_k(z) = Σ_{j≥k} r_j z^{j-k}, which avoids the cancellation of k divisions by a small z.
function phi_from_R!(out, z, R, r, k_phi, zs)
    @inbounds for i in eachindex(z)
        ζ = z[i]
        if abs(ζ) < zs
            s = zero(ComplexF64)
            for j in length(r)-1:-1:k_phi
                s = s * ζ + r[j + 1]
            end
            out[i] = s
        else
            φ = R[i]
            fact = 1.0
            for j in 0:k_phi-1
                j > 0 && (fact *= j)
                φ = (φ - 1 / fact) / ζ
            end
            out[i] = φ
        end
    end
    return out
end

function phi_blend(z_vals, omega0, omega_c, R_C, R_D, k_phi; p::Int=4, zsmall=0.25)
    z = ComplexF64.(vec(z_vals))
    R = R_blend(z, omega0, omega_c, R_C, R_D; p=p)
    r = rc_series(R_C)
    zs = min(float(zsmall), float(omega0))
    return phi_from_R!(similar(z), z, R, r, k_phi, zs)
end

#spec is (k, c) → φ_k(c z), or (terms, c) with terms = ((k, a), ...) → Σ a φ_k(c z).
function phi_from_spec(z, R, r, spec, zs)
    if spec[1] isa Integer
        return phi_from_R!(similar(z), z, R, r, Int(spec[1]), zs)
    end
    terms = spec[1]
    acc = zeros(ComplexF64, length(z))
    tmp = similar(acc)
    for (kp, a) in terms
        phi_from_R!(tmp, z, R, r, Int(kp), zs)
        @. acc += float(a) * tmp
    end
    return acc
end

#φ̃ action on e1 for every spec, from one eigendecomposition of H_j.
#Specs that share the same scale c reuse one R_blend.
function phi_coefs(Hj, specs, omega0, omega_c, R_C, R_D; p::Int=4, zsmall=0.25)
    j = size(Hj, 1)
    F = eigen(Matrix(Hj))
    vals, X = F.values, F.vectors
    e1 = zeros(ComplexF64, j)
    e1[1] = 1
    ξ = X \ e1
    zs = min(float(zsmall), float(omega0))
    r = rc_series(R_C)
    R_by_sc = Dict{Float64,Tuple{Vector{ComplexF64},Vector{ComplexF64}}}()
    C = Matrix{Float64}(undef, j, length(specs))
    for (i, spec) in pairs(specs)
        sc = float(spec[end])
        z, R = get!(R_by_sc, sc) do
            zz = ComplexF64.(sc .* vals)
            (zz, R_blend(zz, omega0, omega_c, R_C, R_D; p=p))
        end
        C[:, i] .= real.(X * (phi_from_spec(z, R, r, spec, zs) .* ξ))
    end
    return C
end

#Largest change, over the specs, between the dimension-j and dimension-jprev coefficients.
function coef_change(C, Cprev)
    jp = size(Cprev, 1)
    Δ = 0.0
    @inbounds for i in axes(C, 2)
        s = 0.0
        for r in 1:jp
            s += (C[r, i] - Cprev[r, i])^2
        end
        for r in jp+1:size(C, 1)
            s += C[r, i]^2
        end
        Δ = max(Δ, sqrt(s))
    end
    return Δ
end

#CGS2 Arnoldi on hJ. Stops when every requested φ̃-action has changed by less than
#safety*tol over the last d vectors. coefs(Hj) returns the matrix from phi_coefs.
function arnoldi(ws, J, v, mmax, h, coefs, tol; jmin::Int=2, d::Int=2, safety=0.1)
    n = length(v)
    mmax = clamp(Int(mmax), 1, n)
    β = norm(v)
    β < 1e-30 && return nothing, nothing, 0.0, 0
    V, H, w, hcol = ws.V, ws.H, ws.w, ws.hcol
    fill!(@view(H[1:mmax, 1:mmax]), 0)
    @views copyto!(V[:, 1], v)
    rmul!(@view(V[:, 1]), 1 / β)
    Cprev = nothing
    for j in 1:mmax
        xj = @view V[:, j]
        mul_hJ!(w, J, xj, h, ws)
        @views Vj, hj, Hcol = V[:, 1:j], hcol[1:j], H[1:j, j]
        for _ in 1:2
            mul!(hj, transpose(Vj), w)
            Hcol .+= hj
            mul!(w, Vj, hj, -1.0, 1.0)
        end
        hn = norm(w)
        breakdown = hn < 1e-14
        if breakdown || j == mmax || (j >= jmin && (j - jmin) % d == 0)
            C = coefs(@view(H[1:j, 1:j]))
            if breakdown || j == mmax || (Cprev !== nothing && β * coef_change(C, Cprev) < safety * tol)
                return @view(V[:, 1:j]), C, β, j
            end
            Cprev = C
        end
        H[j + 1, j] = hn
        @views copyto!(V[:, j + 1], w)
        rmul!(@view(V[:, j + 1]), 1 / hn)
    end
end

#φ_k(hJ) v. krylov_dim_max is the Arnoldi cap; adaptive tol picks j ≤ that.
#Outputs are bump-allocated from ws.phi_pool (call reset_phi_pool! once per step).
function apply_phi(ws, J, v, h, omega0, omega_c, krylov_dim_max, R_C, R_D, specs, tol; p::Int=4)
    v = asvec(v)
    n = length(v)
    nspecs = length(specs)
    coefs = Hj -> phi_coefs(Hj, specs, omega0, omega_c, R_C, R_D; p=p)
    V, C, β, j = arnoldi(ws, J, v, krylov_dim_max, float(h), coefs, tol)
    outs = Vector{Vector{Float64}}(undef, nspecs)
    if j == 0
        for i in 1:nspecs
            outs[i] = fill!(take_phi_out!(ws, n), 0.0)
        end
        return outs
    end
    for i in 1:nspecs
        out = take_phi_out!(ws, n)
        mul!(out, V, @view(C[:, i]), β, 0.0)
        outs[i] = out
    end
    return outs
end

#Shared preamble of one step: state, Jacobian, ft, cutoff, Krylov dim cap.
#cutoff_adaptive is unused here (single shot); ω_c from Omega_c or one Arnoldi+Rayleigh.
#fn/ft are copied so f/ft may return reused buffers.
function prepare(f, t, y, h; setup=nothing, k=nothing, Omega_c=nothing,
                 J=nothing, ft=nothing, krylov_dim_max::Int=40, cutoff_adaptive=true)
    y = asvec(y)
    fn = copy(asvec(f(t, y)))
    n = length(y)
    J === nothing && (J = JacVec(f, t, y))
    if ft === nothing
        y0 = y
        ft = vec(ForwardDiff.jacobian(tt -> collect(f(tt[1], y0)), [float(t)]))
    else
        ft = copy(asvec(ft))
    end
    k = setup !== nothing ? outlier_count(setup) : (k === nothing ? 0 : Int(k))
    km = clamp(Int(krylov_dim_max), 1, n)
    oc = if Omega_c !== nothing
        float(Omega_c)
    else
        k > 0 || error("need Omega_c, or setup/k with J")
        c = init_cutoff(J, k, n, h)
        step_cutoff!(c, J, Vector{JVPDual}(undef, n))
    end
    return y, fn, J, ft, oc, km
end

#Integrate with fixed step. ω_c from Omega_c or IRAM on J (k from setup/k).
#cutoff_adaptive=true  → Rayleigh-update ω_c every step (needs IRAM cache).
#cutoff_adaptive=false → freeze ω_c after the initial value (given or one-shot IRAM).
#krylov_dim_max: Arnoldi workspace / hard cap (default 40). Adaptive tol picks j ≤ that.
function miiso_solve(step, f, y0, T, h; t0=0.0, setup=nothing, k=nothing, Omega_c=nothing,
                     J=nothing, ft=nothing, krylov_dim_max::Int=40, cutoff_adaptive=true,
                     omega_frac=0.99, taper_order=4, tau=1e-10, n_ws=2, callback=nothing)
    y = vec(Float64.(y0))
    t = float(t0)
    nsteps = max(Int(round((float(T) - t) / float(h))), 0)
    nsteps == 0 && return y
    h = (float(T) - t) / nsteps
    n = length(y)
    km = clamp(Int(krylov_dim_max), 1, n)
    k = setup !== nothing ? outlier_count(setup) : (k === nothing ? 0 : Int(k))

    J_call = J isa Function
    ft_call = ft isa Function
    ws = ntuple(_ -> krylov_ws_new(n, km), Int(n_ws))
    cache = nothing
    oc = if Omega_c !== nothing
        float(Omega_c)
    else
        k > 0 || error("need Omega_c, or setup/k with J")
        J0 = J_call ? J(t, y) : (J === nothing ? JacVec(f, t, y) : J)
        cache = init_cutoff(J0, k, n, h)
        step_cutoff!(cache, J0, ws[1].yd)
    end
    callback !== nothing && callback(t, y)

    for _ in 1:nsteps
        Jj = J_call ? J(t, y) : (J === nothing ? JacVec(f, t, y) : J)
        ftj = if ft_call
            ft(t, y)
        elseif ft === nothing
            yj = y
            vec(ForwardDiff.jacobian(s -> collect(f(s[1], yj)), [float(t)]))
        else
            asvec(ft)
        end
        if cutoff_adaptive && cache !== nothing
            oc = step_cutoff!(cache, Jj, ws[1].yd)
        end
        for wsi in ws
            reset_phi_pool!(wsi)
        end
        y = step(f, t, y, h; Omega_c=oc, J=Jj, ft=ftj, krylov_dim_max=km,
                 omega_frac, taper_order, ws, tau)
        t += h
        callback !== nothing && callback(t, y)
    end
    return y
end
