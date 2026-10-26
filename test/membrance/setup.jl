# 2D IGA membrane physics + runners. Included by convergence / cost / scaling.

using LinearAlgebra
using SparseArrays
using JSON
using OrdinaryDiffEqFeagin

module M2
include(joinpath(@__DIR__, "..", "..", "src", "miiso2.jl"))
end
module M3
include(joinpath(@__DIR__, "..", "..", "src", "miiso3.jl"))
end
module M4
include(joinpath(@__DIR__, "..", "..", "src", "miiso4.jl"))
end
module M5
include(joinpath(@__DIR__, "..", "..", "src", "miiso5.jl"))
end
module M6
include(joinpath(@__DIR__, "..", "..", "src", "miiso6.jl"))
end
include(joinpath(@__DIR__, "..", "..", "src", "genalpha.jl"))

struct Side
    id::Vector{Int}
    B::Matrix{Float64}
    D::Matrix{Float64}
    w::Vector{Float64}
end

#at boundaries, multiplicity of p+1 gives C0 basis functions
function open_knots(nel, p, L=1.0)
    left_pad = fill(0.0, p + 1)
    interior = range(0.0, float(L); length=nel + 1)[2:end - 1]
    right_pad = fill(float(L), p + 1)
    return vcat(left_pad, interior, right_pad)
end

#we make the basis functions (splines) for the IGA membran
function basisfuns(span, x, p, knots)
    N = zeros(p + 1)
    left = zeros(p)
    right = zeros(p)
    N[1] = 1.0
    for j in 1:p
        left[j] = x - knots[span + 1 - j]
        right[j] = knots[span + j] - x
        saved = 0.0
        for r in 0:j - 1
            temp = N[r + 1] / (right[r + 1] + left[j - r])
            N[r + 1] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        end
        N[j + 1] = saved
    end
    return N
end

#we also need the derivatives of the basis functions
function dersbasisfuns(span, x, p, knots)
    ndu = zeros(p + 1, p + 1)
    left = zeros(p)
    right = zeros(p)
    ndu[1, 1] = 1.0
    for j in 1:p
        left[j] = x - knots[span + 1 - j]
        right[j] = knots[span + j] - x
        saved = 0.0
        for r in 0:j - 1
            ndu[j + 1, r + 1] = right[r + 1] + left[j - r]
            temp = ndu[r + 1, j] / ndu[j + 1, r + 1]
            ndu[r + 1, j + 1] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        end
        ndu[j + 1, j + 1] = saved
    end
    N = zeros(p + 1)
    for j in 0:p
        N[j + 1] = ndu[j + 1, p + 1]
    end
    Np = basisfuns(span, x, p - 1, knots)
    dN = zeros(p + 1)
    for j in 0:p
        gi = span - p + j
        denL = knots[gi + p] - knots[gi]
        denR = knots[gi + p + 1] - knots[gi + 1]
        if j == 0
            nL = 0.0
        else
            nL = Np[j]
        end
        if j == p
            nR = 0.0
        else
            nR = Np[j + 1]
        end
        if denL == 0
            termL = 0.0
        else
            termL = nL / denL
        end
        if denR == 0
            termR = 0.0
        else
            termR = nR / denR
        end
        dN[j + 1] = p * (termL - termR)
    end
    return N, dN
end

#we set up a Gausssssian quadruate rule
function leggauss(n)
    beta = zeros(n - 1)
    for i in 1:n - 1
        beta[i] = i / sqrt(4 * i^2 - 1)
    end
    F = eigen(SymTridiagonal(zeros(n), beta))
    return F.values, 2 .* F.vectors[1, :] .^ 2
end

#we now make the 2d membrane problem using the above ingredients
function membrane(nel, p, alpha; L=1.0)
    L = float(L)
    knots = open_knots(nel, p, L)
    n1 = length(knots) - p - 1
    free = 2:n1 - 1
    m1 = length(free)
    nq = p + 2
    xi, wg = leggauss(nq)
    xi = 0.5 .* (xi .+ 1)
    wg = 0.5 .* wg
    breaks = range(0.0, L; length=nel + 1)
    nqp = nel * nq
    Bq = zeros(nqp, m1)
    Dq = zeros(nqp, m1)
    wq = zeros(nqp)
    sides = Side[]
    colof = zeros(Int, n1)
    for (c, i) in enumerate(free)
        colof[i] = c
    end
    row = 0
    for e in 1:nel
        a = breaks[e]
        b = breaks[e + 1]
        jac = b - a
        span = p + e
        B = zeros(nq, p + 1)
        D = zeros(nq, p + 1)
        id = zeros(Int, p + 1)
        for g in 1:nq
            row += 1
            x = a + jac * xi[g]
            N, dN = dersbasisfuns(span, x, p, knots)
            i0 = span - p
            for ai in 1:p + 1
                glob = i0 + ai - 1
                col = colof[glob]
                id[ai] = col
                B[g, ai] = N[ai]
                D[g, ai] = dN[ai]
                if col == 0
                    continue
                end
                Bq[row, col] = N[ai]
                Dq[row, col] = dN[ai]
            end
            wq[row] = wg[g] * jac
        end
        keep = id .!= 0
        push!(sides, Side(id[keep], B[:, keep], D[:, keep], wq[row - nq + 1:row]))
    end
    M1 = zeros(m1, m1)
    K1 = zeros(m1, m1)
    for r in 1:nqp
        wt = wq[r]
        for a in 1:m1
            for b in 1:m1
                M1[a, b] += wt * Bq[r, a] * Bq[r, b]
                K1[a, b] += wt * Dq[r, a] * Dq[r, b]
            end
        end
    end
    return (
        nel=nel, p=p, alpha=alpha, L=L, m1=m1, N=m1 * m1,
        M1=M1, K1=K1, Bq=Bq, Dq=Dq, W2=wq * wq', sides=sides,
        M=sparse(kron(M1, M1)), MF=cholesky(Symmetric(M1)),
        setup=("bar", "fixed", p, 2, m1),
    )
end

#we set up the initial condition that is the first mode of the membrane
function mode_ic(mem, amp)
    v = eigen(Symmetric(mem.K1), Symmetric(mem.M1)).vectors[:, 1]
    Q = v * v'
    return (amp / maximum(abs, Q)) * vec(Q)
end

#we set up the initial condition that is a bump in the corner of the membrane
function corner_bump_ic(mem, amp; sigma=0.08, inset=0.08)
    m1 = mem.m1
    L = mem.L
    xs = range(0.0, L; length=m1)
    centers = ((inset, inset), (inset, L - inset), (L - inset, inset), (L - inset, L - inset))
    Q = zeros(m1, m1)
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(xs)
            s = 0.0
            for (x0, y0) in centers
                s += exp(-((x - x0)^2 + (y - y0)^2) / (2 * sigma^2))
            end
            Q[i, j] = s
        end
    end
    q = vec(Q)
    return (amp / maximum(abs, q)) * q
end

#function that solces the mass matric
function mass_solve(mem, r)
    R = reshape(r, mem.m1, mem.m1)
    X = mem.MF \ R
    return vec(transpose(mem.MF \ transpose(X)))
end

#computes the fields of equations used
function fields(mem, q)
    Q = reshape(q, mem.m1, mem.m1)
    Gx = mem.Dq * Q * transpose(mem.Bq)
    Gy = mem.Bq * Q * transpose(mem.Dq)
    c = 1.0 .+ mem.alpha .* (Gx .^ 2 .+ Gy .^ 2)
    return Gx, Gy, c
end

#internal forces
function f_int(mem, q)
    Gx, Gy, c = fields(mem, q)
    A = mem.W2 .* c
    F = transpose(mem.Dq) * (A .* Gx) * mem.Bq + transpose(mem.Bq) * (A .* Gy) * mem.Dq
    return vec(F)
end

#tangent action
function tangent_action(mem, v, Gx, Gy, c)
    V = reshape(v, mem.m1, mem.m1)
    Vx = mem.Dq * V * transpose(mem.Bq)
    Vy = mem.Bq * V * transpose(mem.Dq)
    d = Gx .* Vx .+ Gy .* Vy
    ax = mem.W2 .* (c .* Vx .+ (2 * mem.alpha) .* d .* Gx)
    ay = mem.W2 .* (c .* Vy .+ (2 * mem.alpha) .* d .* Gy)
    F = transpose(mem.Dq) * ax * mem.Bq + transpose(mem.Bq) * ay * mem.Dq
    return vec(F)
end

#the sparse matrix for the tangent action
function tangent_sparse(mem, q)
    Q = reshape(q, mem.m1, mem.m1)
    II = Int[]
    JJ = Int[]
    VV = Float64[]
    m1 = mem.m1
    for sy in mem.sides
        for sx in mem.sides
            nx, ny = length(sx.id), length(sy.id)
            if nx == 0 || ny == 0
                continue
            end
            ce = Q[sx.id, sy.id]
            Gx = sx.D * ce * transpose(sy.B)
            Gy = sx.B * ce * transpose(sy.D)
            Ex = kron(sy.B, sx.D)
            Ey = kron(sy.D, sx.B)
            w = vec(sx.w * transpose(sy.w))
            c = 1.0 .+ mem.alpha .* vec(Gx .^ 2 .+ Gy .^ 2)
            s = sqrt.(w .* c)
            Z = vec(Gx) .* Ex .+ vec(Gy) .* Ey
            zw = sqrt(2 * mem.alpha) .* sqrt.(w) .* Z
            Kloc = (s .* Ex)' * (s .* Ex) + (s .* Ey)' * (s .* Ey) + zw' * zw
            ids = Int[]
            for b in 1:ny
                for a in 1:nx
                    push!(ids, sx.id[a] + (sy.id[b] - 1) * m1)
                end
            end
            for jb in eachindex(ids)
                for ib in eachindex(ids)
                    push!(II, ids[ib])
                    push!(JJ, ids[jb])
                    push!(VV, Kloc[ib, jb])
                end
            end
        end
    end
    return sparse(II, JJ, VV, mem.N, mem.N)
end

#computes the relative L2 error
function l2_rel(mem, q, qref)
    e = q - qref
    E = reshape(e, mem.m1, mem.m1)
    R = reshape(qref, mem.m1, mem.m1)
    num = sqrt(sum(E .* (mem.M1 * E * mem.M1)))
    den = sqrt(sum(R .* (mem.M1 * R * mem.M1)))
    return num / den
end

#solces the ODE using the Feagin12 method
function feagin12_ref(f, y0, T, dt)
    nsteps = round(Int, T / dt)
    h = T / nsteps
    function rhs!(du, u, p, t)
        du .= f(t, u)
        return du
    end
    prob = ODEProblem(rhs!, copy(y0), (0.0, float(T)))
    sol = solve(prob, Feagin12(); dt=h, adaptive=false, save_everystep=false)
    return sol.u[end]
end

#the right hand side of the ODE
function membrane_rhs(mem)
    N = mem.N
    function f(t, y)
        pdot = @view(y[N + 1:end])
        q = @view(y[1:N])
        return vcat(pdot, -mass_solve(mem, f_int(mem, q)))
    end
    function Jac(t, y)
        Gx, Gy, c = fields(mem, @view(y[1:N]))
        function jacvec(z)
            vdot = @view(z[N + 1:end])
            v = @view(z[1:N])
            return vcat(vdot, -mass_solve(mem, tangent_action(mem, v, Gx, Gy, c)))
        end
        return jacvec
    end
    return f, Jac
end

#setupos for the convergence test
CONV = (
    nel = 6, p = 4, alpha = 1.0, amp = 0.2, T = 1.0,
    href = 5e-5,
    hs = (0.05, 0.025, 0.0125, 0.00625),
    km = 80,
    Omega_c = 1.0e12,
    out = joinpath(@__DIR__, "out", "convergence_nel6.json"),
)

#the cost vs error test
COST = (
    nel = 20, p = 4, alpha = 1.0, amp = 0.2, T = 1.0,
    href = 5e-5,
    hs = (0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625),
    km = 40, nrep = 2, rho = 0.5,
    out = joinpath(@__DIR__, "out", "cost_vs_error_nel20.json"),
)

#setupos for the scaling test
SCALE = (
    nels = (6, 12, 20, 28, 40, 56, 72, 96),
    p = 4, alpha = 1.0, amp = 0.2, T = 1.0,
    h = 0.00625, href = 6.25e-4,
    km = 40, nrep = 2, rho = 0.5,
    out = joinpath(@__DIR__, "out", "scaling_cost.json"),
)

#makes a case for the test
function make_case(nel, p, alpha, amp; ic=mode_ic, L=1.0)
    mem = membrane(nel, p, alpha; L=L)
    N = mem.N
    k = M2.outlier_count(mem.setup)
    q0 = ic(mem, amp)
    y0 = vcat(q0, zeros(N))
    f, Jac = membrane_rhs(mem)
    return (; mem, N, setup=mem.setup, k, q0, y0, f, Jac, ft=zeros(2N), nel, p, alpha, amp)
end
function make_case(prob::NamedTuple)
    return make_case(prob.nel, prob.p, prob.alpha, prob.amp)
end

#runs all of the methods for the tests
function runners(case, T; km=40, Omega_c=nothing, rho=nothing)
    (; mem, setup, q0, y0, f, Jac, ft, N) = case
    if Omega_c === nothing
        kw = (; setup=setup, J=Jac, ft=ft, krylov_dim_max=km, cutoff_adaptive=true)
    else
        kw = (; Omega_c=float(Omega_c), J=Jac, ft=ft, krylov_dim_max=km, cutoff_adaptive=false)
    end
    rs = [
        ("MIISO2", h -> M2.miiso2_solve(f, y0, T, h; kw...)),
        ("MIISO3", h -> M3.miiso3_solve(f, y0, T, h; kw...)),
        ("MIISO4", h -> M4.miiso4_solve(f, y0, T, h; kw...)),
        ("MIISO5", h -> M5.miiso5_solve(f, y0, T, h; kw...)),
        ("MIISO6", h -> M6.miiso6_solve(f, y0, T, h; kw...)),
    ]
    if rho !== nothing
        push!(rs, ("gen-a", h -> genalpha_solve(
            q -> f_int(mem, q), q -> tangent_sparse(mem, q), mem.M,
            r -> mass_solve(mem, r), q0, zeros(N), T, h; rho_inf=rho)[1]))
    end
    return rs
end

#dumps the results to a JSON file
function dump_json(path, data)
    open(path, "w") do io
        JSON.print(io, data)
    end
end
