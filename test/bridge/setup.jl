# Arch bridge geometry setup using NURBS.
#setup is done with Ferrite and FerriteIGA.
#The majority of this code is pulled from
#previous earthquake engineering codes

include(joinpath(@__DIR__, "..", "..", "src", "miiso4.jl"))

using Ferrite, FerriteIGA, LinearAlgebra, SparseArrays, Random

# Bridge parameters
Base.@kwdef struct BridgeParams
    L::Float64 = 120.0
    f::Float64 = 24.0
    d::Float64 = 2.5
    b::Float64 = 10.0
    p::Int = 4
    nel::NTuple{3,Int} = (32, 6, 6)
    E::Float64 = 34e9
    nu::Float64 = 0.2
    rho::Float64 = 2500.0
    deck_mass::Float64 = 20e3
    ncol::Int = 11
    col_width::Float64 = 1.2
    zeta::Float64 = 0.05
    f1::Float64 = 1.2
    f2::Float64 = 25.0
    g::Float64 = 9.81
    gravity::Bool = true
    t_ramp::Float64 = 1.0
    t_eq::Float64 = 1.5
    pga::Float64 = 0.4
    pga_v::Float64 = 2 / 3
    duration::Float64 = 10.0
    scale::Float64 = 1.0
    seed::Int = 2024
    record = nothing  
    jac_rtol::Float64 = 0.1
    jac_max_lag::Int = 50
end

# refinment of 1D NURBS curves with the Ferrite h and p refinement functions. I recently added these functions into FerriteIGA pls check them out
function _refine_1d(cps::Vector{<:Vec}, ws::Vector{Float64}, kv::Vector{Float64}, order::Int, p_target::Int, nel::Int)
    cps, ws, kv = copy(cps), copy(ws), copy(kv)
    kvt = (kv,)
    while order < p_target
        order = orderelevation!(kvt, (order,), cps, ws; dir=1)[1]
    end
    for xi in range(0.0, 1.0; length=nel + 1)[2:end-1]
        knotinsertion!(kvt, (order,), cps, ws, xi; dir=1)
    end
    return cps, ws, kvt[1], order
end

# arch bridge mesh setup, returns the mesh and the parameters.
function arch_rib_mesh(P::BridgeParams)
    (; L, f, d, b, p, nel) = P
    R = (L^2 / 4 + f^2) / (2f)
    theta = asin(L / (2R))
    yc = f - R
    rin, rout = R - d / 2, R + d / 2
    a_cps = Vec{2,Float64}[Vec(-sin(theta), cos(theta)), Vec(0.0, 1 / cos(theta)), Vec(sin(theta), cos(theta))]
    a_ws = Float64[1.0, cos(theta), 1.0]
    a_cps, a_ws, a_kv, oa = _refine_1d(a_cps, a_ws, Float64[0, 0, 0, 1, 1, 1], 2, p, nel[1])
    r_cps, _, r_kv, or_ = _refine_1d(Vec{1,Float64}[Vec(rin), Vec(rout)], ones(2), Float64[0, 0, 1, 1], 1, p, nel[2])
    z_cps, _, z_kv, oz = _refine_1d(Vec{1,Float64}[Vec(-b / 2), Vec(b / 2)], ones(2), Float64[0, 0, 1, 1], 1, p, nel[3])
    cps = Vec{3,Float64}[]; ws = Float64[]
    for iz in eachindex(z_cps), ir in eachindex(r_cps), ia in eachindex(a_cps)
        r = r_cps[ir][1]
        push!(cps, Vec(r * a_cps[ia][1], yc + r * a_cps[ia][2], z_cps[iz][1]))
        push!(ws, a_ws[ia])
    end
    return NURBSMesh((a_kv, r_kv, z_kv), (oa, or_, oz), cps, ws), (R=R, theta=theta, yc=yc, rin=rin, rout=rout)
end

# B-splines, this specific part and the derivative was written with some AI help for bookeeping
function bsplines(kv::Vector{Float64}, p::Int, x::Float64)
    m = length(kv) - 1
    N = zeros(m)
    for i in 1:m
        if kv[i] <= x < kv[i + 1] || (x == kv[end] && kv[i] < kv[i + 1] == kv[end])
            N[i] = 1.0
        end
    end
    dN = zeros(m)
    for q in 1:p
        Nn = zeros(m - q); dNn = zeros(m - q)
        for i in 1:m-q
            a = kv[i + q] - kv[i]; c = kv[i + q + 1] - kv[i + 1]
            t1 = a > 0 ? (x - kv[i]) / a * N[i] : 0.0
            t2 = c > 0 ? (kv[i + q + 1] - x) / c * N[i + 1] : 0.0
            Nn[i] = t1 + t2
            if q == p
                dNn[i] = (a > 0 ? q / a * N[i] : 0.0) - (c > 0 ? q / c * N[i + 1] : 0.0)
            end
        end
        N = Nn; dN = dNn
    end
    return N, dN
end

# Gaaaaaussian quadrautre
function gauss_legendre(n)
    J = SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:n-1])
    F = eigen(J)
    return F.values, 2 .* F.vectors[1, :] .^ 2
end

#Quadrature points on the NURBS cyurves
function span_quadrature(kv, nq; lo=kv[1], hi=kv[end])
    xg, wg = gauss_legendre(nq)
    X = Float64[]; W = Float64[]
    br = unique(vcat(filter(k -> lo < k < hi, kv), lo, hi)); sort!(br)
    for (a, c) in zip(br[1:end-1], br[2:end])
        c > a || continue
        append!(X, (c - a) / 2 .* xg .+ (a + c) / 2); append!(W, (c - a) / 2 .* wg)
    end
    return X, W
end

struct ArchDir
    kv::Vector{Float64}; p::Int; w::Vector{Float64}; P::Vector{NTuple{2,Float64}}
end

# Evaluation of the geometry
function arch_eval(A::ArchDir, xi)
    N, dN = bsplines(A.kv, A.p, xi)
    W = dot(A.w, N); dW = dot(A.w, dN)
    R = A.w .* N ./ W
    dR = A.w .* (dN .* W .- N .* dW) ./ W^2
    ax = sum(R[i] * A.P[i][1] for i in eachindex(R))
    ay = sum(R[i] * A.P[i][2] for i in eachindex(R))
    dax = sum(dR[i] * A.P[i][1] for i in eachindex(R))
    day = sum(dR[i] * A.P[i][2] for i in eachindex(R))
    return R, dR, hypot(dax, day), atan(ax, ay)
end

# Assembly of the mass, stiffness, and gravity operators
function arch_operators(A::ArchDir; nq=A.p + 3)
    n = length(A.w)
    M = zeros(n, n); K = zeros(n, n); g = zeros(n)
    X, Wq = span_quadrature(A.kv, nq)
    for (xi, wq) in zip(X, Wq)
        R, dR, s, _ = arch_eval(A, xi)
        M .+= (wq * s) .* (R * R'); K .+= (wq / s) .* (dR * dR'); g .+= (wq * s) .* R
    end
    return M, K, g
end

# Assembly of the mass, stiffness, and gravity operators for a line element
function line_operators(kv, p, x0, ell; weight=x -> 1.0, nq=p + 3)
    n = length(kv) - p - 1
    M = zeros(n, n); K = zeros(n, n); g = zeros(n)
    X, Wq = span_quadrature(kv, nq)
    for (eta, wq) in zip(X, Wq)
        N, dN = bsplines(kv, p, eta)
        x = x0 + ell * eta; omega = weight(x)
        M .+= (wq * ell * omega) .* (N * N'); K .+= (wq * omega / ell) .* (dN * dN'); g .+= (wq * ell * omega) .* N
    end
    return M, K, g
end

# 3D kronecker product
function kron3(A1, A2, A3, X::Array{Float64,4})
    c, n1, n2, n3 = size(X)
    Y = permutedims(X, (2, 1, 3, 4)); Y = reshape(A1 * reshape(Y, n1, :), n1, c, n2, n3)
    Y = permutedims(Y, (3, 2, 1, 4)); Y = reshape(A2 * reshape(Y, n2, :), n2, c, n1, n3)
    Y = permutedims(Y, (4, 2, 3, 1)); Y = reshape(A3 * reshape(Y, n3, :), n3, c, n1, n2)
    return permutedims(Y, (2, 3, 4, 1))
end

# 3D kronecker product in place
function kron3!(out::Array{Float64,4}, A1, A2, A3, X::Array{Float64,4},
                W1::Array{Float64,4}, W2::Array{Float64,4})
    c, n1, n2, n3 = size(X)
    @inbounds for i3 in 1:n3, i2 in 1:n2, ic in 1:c
        mul!(view(W1, ic, :, i2, i3), A1, view(X, ic, :, i2, i3))
    end
    @inbounds for i3 in 1:n3, i1 in 1:n1, ic in 1:c
        mul!(view(W2, ic, i1, :, i3), A2, view(W1, ic, i1, :, i3))
    end
    @inbounds for i2 in 1:n2, i1 in 1:n1, ic in 1:c
        mul!(view(out, ic, i1, i2, :), A3, view(W2, ic, i1, i2, :))
    end
    return out
end

# Inverse of the mass matrix
minv(S, r::AbstractVector) = vec(kron3(S.iMa, S.iMr, S.iMz, reshape(r, S.shape))) ./ S.rho_eff

function minv!(out::AbstractVector{Float64}, S, r::AbstractVector)
    tid = Threads.threadid()
    w = S.work[tid]
    copyto!(w.x, r)
    X = reshape(w.x, S.shape)
    Y = reshape(out, S.shape)
    kron3!(Y, S.iMa, S.iMr, S.iMz, X, w.w1, w.w2)
    @. out = out / S.rho_eff
    return out
end

# The synehetic earthquake motion comes from a random process, which is where the seed is used
function synthetic_record(; dt=0.005, duration=10.0, pga=0.4, g=9.81, omega_g=15.6, zeta_g=0.6,
                          t1=1.5, t2=6.0, c=0.35, rng=Random.default_rng())
    nt = round(Int, duration / dt) + 1
    x = 0.0; v = 0.0; a = zeros(nt)
    h = dt / 10
    for k in 1:nt
        for _ in 1:10
            w = randn(rng) / sqrt(h)
            v += h * (-2zeta_g * omega_g * v - omega_g^2 * x - w)
            x += h * v
        end
        t = (k - 1) * dt
        env = t < t1 ? (t / t1)^2 : t <= t2 ? 1.0 : exp(-c * (t - t2))
        a[k] = -(2zeta_g * omega_g * v + omega_g^2 * x) * env
    end
    a .-= range(a[1], a[end]; length=nt)
    return a .* (pga * g / maximum(abs, a))
end

#interpolation of the earthquake record
function interp_record(a::Vector{Float64}, dt, t)
    tau = float(t) / dt
    i = floor(Int, tau)
    (i < 0 || i >= length(a) - 1) && return 0.0, 0.0
    s = a[i + 2] - a[i + 1]
    return a[i + 1] + (tau - i) * s, s / dt
end

#bisection method to find the cutoff frequency
function bisect(g, a, b; iters=80)
    ga = g(a)
    for _ in 1:iters
        m = (a + b) / 2; gm = g(m)
        (sign(gm) == sign(ga)) ? (a = m; ga = gm) : (b = m)
    end
    return (a + b) / 2
end

kbar(p, bc::Symbol) = bc === :fixed ? (p - 1) ÷ 2 : p ÷ 2

#sparse Jacobian operator
mutable struct BridgeJOp
    Kt::SparseMatrixCSC{Float64,Int}
    K0::SparseMatrixCSC{Float64,Int}
    Sm
    a0::Float64
    a1::Float64
    nf::Int
    work::Vector{NamedTuple{(:r, :t, :acc, :out),
                            Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}}}
end
function BridgeJOp(Kt, K0, Sm, a0, a1, nf)
    nth = Threads.maxthreadid()
    work = [(; r=zeros(nf), t=zeros(nf), acc=zeros(nf), out=zeros(2nf)) for _ in 1:nth]
    BridgeJOp(Kt, K0, Sm, a0, a1, nf, work)
end
function (J::BridgeJOp)(x::AbstractVector)
    nf = J.nf
    b = J.work[Threads.threadid()]
    dq = view(x, 1:nf); dv = view(x, nf + 1:2nf)
    mul!(b.r, J.Kt, dq)
    mul!(b.t, J.K0, dv)
    @. b.r = b.r + J.a1 * b.t
    minv!(b.acc, J.Sm, b.r)
    @inbounds for i in 1:nf
        b.out[i] = dv[i]
        b.out[nf + i] = -b.acc[i] - J.a0 * dv[i]
    end
    return b.out
end

#free degree of freedom mapping and sparsity patterns
function free_sparsity(grid, free, nnodes)
    g2f = zeros(Int, 3 * nnodes)
    @inbounds for (i, g) in enumerate(free)
        g2f[g] = i
    end
    nf = length(free)
    cols = [Int[] for _ in 1:nf]
    for ic in 1:getncells(grid)
        nodes = grid.cells[ic].nodes
        ld = Int[]
        for a in eachindex(nodes), c in 1:3
            f = g2f[3 * (nodes[a] - 1) + c]
            f > 0 && push!(ld, f)
        end
        for v in ld
            append!(cols[v], ld)
        end
    end
    for v in 1:nf
        sort!(cols[v]); unique!(cols[v])
    end
    nnz = sum(length, cols)
    rowval = Vector{Int}(undef, nnz)
    colptr = Vector{Int}(undef, nf + 1)
    colptr[1] = 1
    p = 1
    for v in 1:nf
        r = cols[v]; n = length(r)
        copyto!(rowval, p, r, 1, n)
        p += n
        colptr[v + 1] = p
    end
    return SparseMatrixCSC(nf, nf, colptr, rowval, ones(nnz)), g2f
end

#stiffness matrix assembly
function assemble_K!(Kt::SparseMatrixCSC, K0, S, U::AbstractVector{<:Real}, g2f; do_K0::Bool=false)
    fill!(nonzeros(Kt), 0.0)
    aKt = start_assemble(Kt)
    aK0 = do_K0 ? (fill!(nonzeros(K0), 0.0); start_assemble(K0)) : nothing
    lambda, mu = S.lambda, S.mu
    cv = S.cvs[1]; bc = S.bcs[1]
    w = S.aw
    ue, locf, remap = w.ue, w.locf, w.remap
    Ke, Ke0 = w.Ke, w.Ke0
    fdofs, Kef, Ke0f = w.fdofs, w.Kef, w.Ke0f
    for ic in 1:getncells(S.grid)
        Ferrite.getcoordinates!(bc, S.grid, ic)
        Ferrite.reinit!(cv, bc)
        nodes = S.grid.cells[ic].nodes
        nb = getnbasefunctions(cv)
        fill!(Ke, 0)
        do_K0 && fill!(Ke0, 0)
        fill!(remap, 0)
        @inbounds for a in eachindex(nodes), c in 1:3
            li = 3(a - 1) + c
            gi = 3 * (nodes[a] - 1) + c
            ue[li] = U[gi]
            locf[li] = g2f[gi]
        end
        for q in 1:getnquadpoints(cv)
            dOmega = getdetJdV(cv, q)
            gradu = function_gradient(cv, q, ue)
            F = one(gradu) + gradu
            E = symmetric(0.5 * (transpose(F) ⋅ F - one(F)))
            Sv = lambda * tr(E) * one(E) + 2mu * E
            for j in 1:nb
                dF = shape_gradient(cv, q, j)
                dE = symmetric(transpose(F) ⋅ dF)
                dS = lambda * tr(dE) * one(E) + 2mu * dE
                dP = dF ⋅ Sv + F ⋅ dS
                for i in 1:nb
                    Ke[i, j] += (dP ⊡ shape_gradient(cv, q, i)) * dOmega
                end
                if do_K0
                    epsj = shape_symmetric_gradient(cv, q, j)
                    sigmaj = lambda * tr(epsj) * one(epsj) + 2mu * epsj
                    for i in 1:nb
                        Ke0[i, j] += (sigmaj ⊡ shape_gradient(cv, q, i)) * dOmega
                    end
                end
            end
        end
        nfe = 0
        @inbounds for i in 1:nb
            locf[i] == 0 && continue
            nfe += 1
            fdofs[nfe] = locf[i]
            remap[i] = nfe
        end
        nfe == 0 && continue
        @inbounds for j in 1:nfe, i in 1:nfe
            Kef[i, j] = 0.0
            do_K0 && (Ke0f[i, j] = 0.0)
        end
        @inbounds for j in 1:nb, i in 1:nb
            ri = remap[i]; rj = remap[j]
            (ri == 0 || rj == 0) && continue
            Kef[ri, rj] = Ke[i, j]
            do_K0 && (Ke0f[ri, rj] = Ke0[i, j])
        end
        assemble!(aKt, view(fdofs, 1:nfe), view(Kef, 1:nfe, 1:nfe))
        do_K0 && assemble!(aK0, view(fdofs, 1:nfe), view(Ke0f, 1:nfe, 1:nfe))
    end
    return Kt, K0
end

# Ferrite coloring is too coarse here, so we use thread-local buffers
function element_fint!(fint::Vector{Float64}, S, U::AbstractVector{Float64})
    lambda, mu = S.lambda, S.mu
    ncells = getncells(S.grid)
    fbufs = S.fbufs
    for b in fbufs
        fill!(b, 0.0)
    end
    Threads.@threads :static for ic in 1:ncells
        tid = Threads.threadid()
        cv = S.cvs[tid]; bc = S.bcs[tid]
        tls = S.tls[tid]
        fbuf = fbufs[tid]
        Ferrite.getcoordinates!(bc, S.grid, ic)
        Ferrite.reinit!(cv, bc)
        nodes = S.grid.cells[ic].nodes
        nb = getnbasefunctions(cv)
        ue, fe = tls.ue, tls.fe
        fill!(fe, 0)
        @inbounds for a in eachindex(nodes), c in 1:3
            gi = 3 * (nodes[a] - 1) + c
            ue[3(a - 1) + c] = U[gi]
        end
        for q in 1:getnquadpoints(cv)
            dOmega = getdetJdV(cv, q)
            gradu = function_gradient(cv, q, ue)
            F = one(gradu) + gradu
            E = symmetric(0.5 * (transpose(F) ⋅ F - one(F)))
            Sv = lambda * tr(E) * one(E) + 2mu * E
            Pk = F ⋅ Sv
            for i in 1:nb
                fe[i] += (Pk ⊡ shape_gradient(cv, q, i)) * dOmega
            end
        end
        @inbounds for a in eachindex(nodes), c in 1:3
            gi = 3 * (nodes[a] - 1) + c
            fbuf[gi] += fe[3(a - 1) + c]
        end
    end
    fill!(fint, 0.0)
    @inbounds for b in fbufs, i in eachindex(fint)
        fint[i] += b[i]
    end
    return fint
end

#arch bridge system setup that gives the whole operators
function arch_bridge_system(P::BridgeParams=BridgeParams())
    (; p, E, nu, rho, d, b, L, g) = P
    mesh, geo = arch_rib_mesh(P)
    (; R, yc, rin, rout) = geo
    n1, n2, n3 = length.(mesh.knot_vectors) .- mesh.orders .- 1

    w1 = mesh.weights[1:n1]
    Pu = [((mesh.control_points[i][1]) / rin, (mesh.control_points[i][2] - yc) / rin) for i in 1:n1]
    A = ArchDir(copy(mesh.knot_vectors[1]), p, w1, Pu)
    Ma, Ka, ga = arch_operators(A)
    Mr, Kr, gr = line_operators(mesh.knot_vectors[2], p, rin, d; weight=r -> r)
    Mz, Kz, gz = line_operators(mesh.knot_vectors[3], p, -b / 2, b)

    fa = 2:n1-1
    Maf = Ma[fa, fa]
    rho_eff = rho + P.deck_mass * L / (sum(ga) * sum(gr) * sum(gz))
    iMa, iMr, iMz = inv(Symmetric(Maf)), inv(Symmetric(Mr)), inv(Symmetric(Mz))

    shape_full = (3, n1, n2, n3)
    free = vec(LinearIndices(shape_full)[:, fa, :, :])
    nf = length(free)
    shape = (3, length(fa), n2, n3)
    nth = Threads.maxthreadid()
    Sm = (shape=shape, iMa=iMa, iMr=iMr, iMz=iMz, rho_eff=rho_eff,
          work=[(; x=zeros(nf), w1=zeros(shape), w2=zeros(shape)) for _ in 1:nth])

    Fg_full = zeros(shape_full)
    Fg_full[2, :, :, :] .= -rho * g .* reshape(kron(gz, kron(gr, ga)), n1, n2, n3)
    Pcol = P.deck_mass * L * g / P.ncol
    for m in 1:P.ncol
        xc = -L / 2 + L * m / (P.ncol + 1)
        phi_c = asin(xc / rout); dphi = P.col_width / (2rout)
        phi(xi) = arch_eval(A, xi)[4]
        xia = bisect(xi -> phi(xi) - (phi_c - dphi), 0.0, 1.0)
        xib = bisect(xi -> phi(xi) - (phi_c + dphi), 0.0, 1.0)
        X, Wq = span_quadrature(A.kv, p + 3; lo=xia, hi=xib)
        fa_col = zeros(n1); arc = 0.0
        for (xi, wq) in zip(X, Wq)
            Rb, _, s, _ = arch_eval(A, xi)
            fa_col .+= (wq * s * rout) .* Rb; arc += wq * s * rout
        end
        Fg_full[2, :, n2, :] .-= (Pcol / (arc * b)) .* (fa_col * gz')
    end
    Fg = vec(Fg_full)[free]

    Mones_a = (Ma * ones(n1))[fa]
    rig = reshape(kron(Mz * ones(n3), kron(Mr * ones(n2), Mones_a)), length(fa), n2, n3)
    Xr = zeros(shape); Xr[1, :, :, :] .= rig
    S_iota = (shape=shape, iMa=iMa, iMr=iMr, iMz=iMz, rho_eff=1.0)
    iota_x = minv(S_iota, vec(Xr))
    Xr .= 0; Xr[2, :, :, :] .= rig
    iota_y = minv(S_iota, vec(Xr))

    rng = Random.MersenneTwister(P.seed)
    if P.record === nothing
        dt_rec = 0.005
        ax = P.scale .* synthetic_record(; dt=dt_rec, duration=P.duration, pga=P.pga, g=g, rng=rng)
        ay = P.scale .* synthetic_record(; dt=dt_rec, duration=P.duration, pga=P.pga_v * P.pga, g=g, rng=rng)
    else
        dt_rec, ax, ay = P.record
        ax = P.scale .* ax; ay = P.scale .* ay
    end

    omega1, omega2 = 2pi * P.f1, 2pi * P.f2
    a0 = 2P.zeta * omega1 * omega2 / (omega1 + omega2); a1 = 2P.zeta / (omega1 + omega2)
    lambda = E * nu / ((1 + nu) * (1 - 2nu)); mu = E / (2(1 + nu))

    # Hiemstra omega_c from 1D directional spectra
    cs = sqrt(mu / rho_eff)
    Rm = (rin + rout) / 2
    la = sort(eigvals(Symmetric(Ka[fa, fa]), Symmetric(Maf))) ./ Rm^2
    lr = sort(eigvals(Symmetric(Kr), Symmetric(Mr)))
    lz = sort(eigvals(Symmetric(Kz), Symmetric(Mz)))
    kd = (kbar(p, :fixed) * 2, kbar(p, :free) * 2, kbar(p, :free) * 2)
    ls = (la, lr, lz)
    lambda_bdry = Float64[]
    for dd in 1:3
        k = min(kd[dd], length(ls[dd]))
        i = length(ls[dd]) - k + 1
        push!(lambda_bdry, ls[dd][i] + sum(ls[e][1] for e in 1:3 if e != dd))
    end
    omega_c = cs * sqrt(minimum(lambda_bdry))

    #here we use FerriteIGA's routines. More information on this can be found
    #in the FerriteIGA documentation and the talk I gave at the FerriteIGA conference 2026.
    grid = BezierGrid(mesh)
    ip = IGAInterpolation{RefHexahedron, p}()^3
    cv0 = BezierCellValues(QuadratureRule{RefHexahedron}(p + 1), ip)
    bc0 = getcoordinates(grid, 1)
    cvs = [deepcopy(cv0) for _ in 1:nth]; bcs = [deepcopy(bc0) for _ in 1:nth]
    colors = Ferrite.create_coloring(grid.grid)
    nnodes = length(mesh.control_points)
    nb0 = getnbasefunctions(cv0)
    tls = [(; ue=zeros(nb0), ve=zeros(nb0), fe=zeros(nb0), ke=zeros(nb0)) for _ in 1:nth]
    fbufs = [zeros(3nnodes) for _ in 1:nth]
    aw = (; ue=zeros(nb0), locf=zeros(Int, nb0), remap=zeros(Int, nb0),
          Ke=zeros(nb0, nb0), Ke0=zeros(nb0, nb0),
          fdofs=zeros(Int, nb0), Kef=zeros(nb0, nb0), Ke0f=zeros(nb0, nb0))
    asm = (; grid, cvs, bcs, colors, nnodes, free, nf, lambda, mu, tls, aw, fbufs)

    gfac(t) = !P.gravity ? 0.0 : t >= P.t_ramp ? 1.0 : (s = t / P.t_ramp; 3s^2 - 2s^3)
    dgfac(t) = !P.gravity ? 0.0 : t >= P.t_ramp ? 0.0 : (s = t / P.t_ramp; 6s * (1 - s) / P.t_ramp)

    Kpat, g2f = free_sparsity(grid, free, nnodes)
    K0 = copy(Kpat)
    assemble_K!(Kpat, K0, asm, zeros(3nnodes), g2f; do_K0=true)
    Kt = Kpat

    U = zeros(3nnodes)
    fint_buf = zeros(3nnodes)
    k0v_free = zeros(nf)
    rbuf = zeros(nf); accbuf = zeros(nf); yout = zeros(2nf); ftout = zeros(2nf)
    function f(t, y)
        fill!(U, 0)
        U[free] .= view(y, 1:nf)
        element_fint!(fint_buf, asm, U)
        mul!(k0v_free, K0, view(y, nf + 1:2nf))
        @inbounds for i in 1:nf
            rbuf[i] = gfac(t) * Fg[i] - fint_buf[free[i]] - a1 * k0v_free[i]
        end
        minv!(accbuf, Sm, rbuf)
        axg = interp_record(ax, dt_rec, t - P.t_eq)[1]
        ayg = interp_record(ay, dt_rec, t - P.t_eq)[1]
        @inbounds for i in 1:nf
            yout[i] = y[nf + i]
            yout[nf + i] = accbuf[i] - a0 * y[nf + i] - iota_x[i] * axg - iota_y[i] * ayg
        end
        return yout
    end

    MinvFg = minv(Sm, Fg)
    function ft(t, y)
        _, sx = interp_record(ax, dt_rec, t - P.t_eq)
        _, sy = interp_record(ay, dt_rec, t - P.t_eq)
        dg = dgfac(t)
        @inbounds for i in 1:nf
            ftout[i] = 0.0
            ftout[nf + i] = dg * MinvFg[i] - iota_x[i] * sx - iota_y[i] * sy
        end
        return ftout
    end

    Ubuf = zeros(3nnodes)
    q_J = zeros(nf)
    has_qJ = Ref(false)
    lag = Ref(0)
    nrebuild = Ref(0)
    Jop = BridgeJOp(Kt, K0, Sm, a0, a1, nf)
    function J(t, y)
        q = view(y, 1:nf)
        drift = if !has_qJ[]
            Inf
        else
            s = 0.0
            @inbounds for i in eachindex(q)
                delta = q[i] - q_J[i]; s += delta * delta
            end
            sqrt(s) / (1 + norm(q))
        end
        if lag[] >= P.jac_max_lag || drift > P.jac_rtol
            fill!(Ubuf, 0); Ubuf[free] .= q
            assemble_K!(Kt, nothing, asm, Ubuf, g2f; do_K0=false)
            copyto!(q_J, q); has_qJ[] = true; lag[] = 0; nrebuild[] += 1
        end
        lag[] += 1
        return Jop
    end

    return (f=f, ft=ft, J=J, y0=zeros(2nf), omega_c=omega_c, nfree=nf, nrebuild=nrebuild,
            mesh=mesh, free=free, n1=n1, n2=n2, n3=n3,
            Sm=Sm, asm=asm, Fg=Fg, a0=a0, a1=a1, g2f=g2f, Kt=Kt, K0=K0,
            iota_x=iota_x, iota_y=iota_y, ax=ax, ay=ay, dt_rec=dt_rec, P=P, gfac=gfac)
end

#evaluation of the NURBS point
function nurbs_point(kv, orders, cps::AbstractMatrix, ws, xi, eta, zeta, Ucp)
    n1, n2, n3 = length.(kv) .- orders .- 1
    Na, _ = bsplines(kv[1], orders[1], xi)
    Nr, _ = bsplines(kv[2], orders[2], eta)
    Nz, _ = bsplines(kv[3], orders[3], zeta)
    W = 0.0
    x = y = z = ux = uy = uz = 0.0
    @inbounds for iz in 1:n3, ir in 1:n2, ia in 1:n1
        idx = ia + (ir - 1) * n1 + (iz - 1) * n1 * n2
        R = ws[idx] * Na[ia] * Nr[ir] * Nz[iz]
        W += R
        x += R * cps[1, idx]; y += R * cps[2, idx]; z += R * cps[3, idx]
        ux += R * Ucp[1, ia, ir, iz]
        uy += R * Ucp[2, ia, ir, iz]
        uz += R * Ucp[3, ia, ir, iz]
    end
    invW = 1 / W
    return (x * invW, y * invW, z * invW), (ux * invW, uy * invW, uz * invW)
end

#conversion of the control points to a matrix
function cps_matrix(control_points)
    n = length(control_points)
    C = Matrix{Float64}(undef, 3, n)
    @inbounds for i in 1:n
        P = control_points[i]
        C[1, i] = P[1]; C[2, i] = P[2]; C[3, i] = P[3]
    end
    return C
end

#workspace for the U matrix
make_U_workspace(n1, n2, n3) = (Ucp = zeros(3, n1, n2, n3); (Ucp, reshape(Ucp, :)))
