# Nonlinear 2D IGA membrane — physics, solvers, problem defs, shared runners.
# Included by convergence.jl / cost_to_error.jl / scaling.jl.

using LinearAlgebra
using SparseArrays
using Printf
using OrdinaryDiffEqFeagin
using SciMLBase: DiscreteCallback

# ── MIISO2–6 + gen-α ──────────────────────────────────────────────────────────
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

# ── IGA membrane ──────────────────────────────────────────────────────────────
struct Side
    id::Vector{Int}
    B::Matrix{Float64}
    D::Matrix{Float64}
    w::Vector{Float64}
end

function open_knots(nel, p, L=1.0)
    left_pad = fill(0.0, p + 1)
    interior = range(0.0, float(L); length=nel + 1)[2:end - 1]
    right_pad = fill(float(L), p + 1)
    return vcat(left_pad, interior, right_pad)
end

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

function leggauss(n)
    β = zeros(n - 1)
    for i in 1:n - 1
        β[i] = i / sqrt(4 * i^2 - 1)
    end
    F = eigen(SymTridiagonal(zeros(n), β))
    return F.values, 2 .* F.vectors[1, :] .^ 2
end

function membrane(nel, p, alpha; L=1.0)
    L = float(L)
    knots = open_knots(nel, p, L)
    n1 = length(knots) - p - 1
    free = 2:n1 - 1
    m1 = length(free)
    nq = p + 2
    ξ, wg = leggauss(nq)
    ξ = 0.5 .* (ξ .+ 1)
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
            x = a + jac * ξ[g]
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

function mode_ic(mem, amp)
    v = eigen(Symmetric(mem.K1), Symmetric(mem.M1)).vectors[:, 1]
    Q = v * v'
    return (amp / maximum(abs, Q)) * vec(Q)
end

# Product tents at the four corners. Spatial IC only. No Φ_out seeding.
function corner_tent_ic(mem, amp; w=0.08)
    m1 = mem.m1
    L = mem.L
    xs = collect(range(0.0, L; length=m1))
    Q = zeros(m1, m1)
    corners = ((0.0, 0.0), (0.0, L), (L, 0.0), (L, L))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(xs)
            s = 0.0
            for (x0, y0) in corners
                dx = abs(x - x0)
                dy = abs(y - y0)
                if dx < w && dy < w
                    s += (1 - dx / w) * (1 - dy / w)
                end
            end
            Q[i, j] = s
        end
    end
    q = vec(Q)
    return (amp / maximum(abs, q)) * q
end

# Milder four-corner Gaussians. σ, inset are physical lengths (same size on any L).
function corner_bump_ic(mem, amp; σ=0.08, inset=0.08)
    m1 = mem.m1
    L = mem.L
    xs = range(0.0, L; length=m1)
    centers = ((inset, inset), (inset, L - inset), (L - inset, inset), (L - inset, L - inset))
    Q = zeros(m1, m1)
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(xs)
            s = 0.0
            for (x0, y0) in centers
                s += exp(-((x - x0)^2 + (y - y0)^2) / (2σ^2))
            end
            Q[i, j] = s
        end
    end
    q = vec(Q)
    return (amp / maximum(abs, q)) * q
end

# Single central Gaussian. σ is a physical length.
function center_bump_ic(mem, amp; σ=0.12)
    m1 = mem.m1
    L = mem.L
    xs = range(0.0, L; length=m1)
    cx = 0.5 * L
    cy = 0.5 * L
    Q = zeros(m1, m1)
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(xs)
            Q[i, j] = exp(-((x - cx)^2 + (y - cy)^2) / (2σ^2))
        end
    end
    q = vec(Q)
    return (amp / maximum(abs, q)) * q
end

function mass_solve(mem, r)
    R = reshape(r, mem.m1, mem.m1)
    X = mem.MF \ R
    return vec(transpose(mem.MF \ transpose(X)))
end

function fields(mem, q)
    Q = reshape(q, mem.m1, mem.m1)
    Gx = mem.Dq * Q * transpose(mem.Bq)
    Gy = mem.Bq * Q * transpose(mem.Dq)
    c = 1.0 .+ mem.alpha .* (Gx .^ 2 .+ Gy .^ 2)
    return Gx, Gy, c
end

function f_int(mem, q)
    Gx, Gy, c = fields(mem, q)
    A = mem.W2 .* c
    F = transpose(mem.Dq) * (A .* Gx) * mem.Bq + transpose(mem.Bq) * (A .* Gy) * mem.Dq
    return vec(F)
end

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

function l2_rel(mem, q, qref)
    e = q - qref
    E = reshape(e, mem.m1, mem.m1)
    R = reshape(qref, mem.m1, mem.m1)
    num = sqrt(sum(E .* (mem.M1 * E * mem.M1)))
    den = sqrt(sum(R .* (mem.M1 * R * mem.M1)))
    return num / den
end

function feagin12_ref(f, y0, T, dt; every=0)
    nsteps = max(round(Int, T / dt), 1)
    h = T / nsteps
    if every <= 0
        every = max(nsteps ÷ 20, 1)
    end
    k = Ref(0)
    function affect!(integrator)
        k[] += 1
        if k[] % every == 0 || k[] == nsteps
            @printf("  Feagin step %6d / %d  t=%.6g\n", k[], nsteps, integrator.t)
            flush(stdout)
        end
    end
    function always_true(u, t, integrator)
        return true
    end
    cb = DiscreteCallback(always_true, affect!; save_positions=(false, false))
    function rhs!(du, u, p, t)
        du .= f(t, u)
        return du
    end
    prob = ODEProblem(rhs!, copy(y0), (0.0, float(T)))
    sol = solve(prob, Feagin12(); dt=h, adaptive=false, save_everystep=false, callback=cb)
    return sol.u[end]
end

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

function spectrum(mem)
    λ1 = eigen(Symmetric(mem.K1), Symmetric(mem.M1)).values
    return sqrt.(max.(sort(vec(λ1 .+ λ1')), 0.0))
end

# ── problem configs ───────────────────────────────────────────────────────────
const KM = 40

const CONV = (
    label = "convergence",
    nel = 6, p = 4, alpha = 1.0, amp = 0.2, T = 1.0,
    href = 5e-5,
    hs = (0.05, 0.025, 0.0125, 0.00625),
    km = 80,
    Omega_c = 1.0e12,
    out = joinpath(@__DIR__, "out", "convergence_nel6.json"),
)

const COST = (
    label = "cost",
    nel = 20, p = 4, alpha = 1.0, amp = 0.2, T = 1.0,
    href = 5e-5,
    hs = (0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625),
    tols = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10),
    primary = 1e-6,
    km = KM, nrep = 2, rho = 0.5,
    out = joinpath(@__DIR__, "out", "cost_vs_error_nel20.json"),
)

const SCALE = (
    label = "scaling",
    nels = (6, 12, 20, 28, 40, 56, 72, 96),
    p = 4, alpha = 1.0, amp = 0.2, T = 1.0,
    h = 0.00625, href = 6.25e-4,
    km = KM, nrep = 2, rho = 0.5,
    out = joinpath(@__DIR__, "out", "scaling_cost.json"),
)

# ── shared case / runners / reporting ─────────────────────────────────────────
function make_case(nel, p, alpha, amp; ic=mode_ic, L=1.0)
    mem = membrane(nel, p, alpha; L=L)
    N = mem.N
    setup = mem.setup
    k = M2.outlier_count(setup)
    q0 = ic(mem, amp)
    y0 = vcat(q0, zeros(N))
    f, Jac = membrane_rhs(mem)
    ft = zeros(2N)
    ω = spectrum(mem)
    return (; mem, N, setup, k, q0, y0, f, Jac, ft, ω, nel, p, alpha, amp, L=mem.L)
end

function make_case(prob::NamedTuple)
    if hasproperty(prob, :L)
        L = prob.L
    else
        L = 1.0
    end
    return make_case(prob.nel, prob.p, prob.alpha, prob.amp; L=L)
end

function print_spectrum(case)
    N, k, ω = case.N, case.k, case.ω
    @printf("ω₁=%.4f  last phys=%.4f  first out=%.4f  ω_max=%.4f\n",
            ω[1], ω[N - k], ω[N - k + 1], ω[end])
end

function miiso_runner(solve_fun, f, y0, T, miiso_kw)
    function run(h)
        return solve_fun(f, y0, T, h; miiso_kw...)
    end
    return run
end

function genalpha_runner(mem, q0, N, T, rho)
    function force(q)
        return f_int(mem, q)
    end
    function stiff(q)
        return tangent_sparse(mem, q)
    end
    function massop(r)
        return mass_solve(mem, r)
    end
    function run(h)
        out = genalpha_solve(force, stiff, mem.M, massop, q0, zeros(N), T, h; rho_inf=rho)
        return out[1]
    end
    return run
end

function runners(case, T; km=40, Omega_c=nothing, rho=nothing)
    (; mem, setup, q0, y0, f, Jac, ft, N) = case
    if Omega_c === nothing
        miiso_kw = (; setup=setup, J=Jac, ft=ft, krylov_dim_max=km, cutoff_adaptive=true)
    else
        miiso_kw = (; Omega_c=float(Omega_c), J=Jac, ft=ft, krylov_dim_max=km, cutoff_adaptive=false)
    end
    rs = (
        ("MIISO2", miiso_runner(M2.miiso2_solve, f, y0, T, miiso_kw)),
        ("MIISO3", miiso_runner(M3.miiso3_solve, f, y0, T, miiso_kw)),
        ("MIISO4", miiso_runner(M4.miiso4_solve, f, y0, T, miiso_kw)),
        ("MIISO5", miiso_runner(M5.miiso5_solve, f, y0, T, miiso_kw)),
        ("MIISO6", miiso_runner(M6.miiso6_solve, f, y0, T, miiso_kw)),
    )
    if rho !== nothing
        rs = (rs...,
            ("gen-α", genalpha_runner(mem, q0, N, T, rho)),
        )
    end
    return rs
end

function cheapest(rows, tol)
    ok = NamedTuple[]
    for r in rows
        if r.err <= tol
            push!(ok, r)
        end
    end
    if isempty(ok)
        return nothing
    end
    best = ok[1]
    for r in ok
        if r.wall < best.wall
            best = r
        end
    end
    return best
end

function h_ladder(name, run, case, yref, T, hs; nrep=1)
    N = case.N
    rows = NamedTuple[]
    println("\n--- $name ---")
    @printf("%10s %7s %14s %10s\n", "h", "steps", "rel L2 err", "wall[s]")
    for h in hs
        nsteps = max(round(Int, T / h), 1)
        h_eff = T / nsteps
        walls = Float64[]
        local y
        for _ in 1:nrep
            push!(walls, @elapsed y = run(h_eff))
        end
        wall = minimum(walls)
        e = l2_rel(case.mem, @view(y[1:N]), @view(yref[1:N]))
        @printf("%10.6g %7d %14.4e %10.3f\n", h_eff, nsteps, e, wall)
        push!(rows, (h=h_eff, nsteps=nsteps, err=e, wall=wall))
    end
    return rows
end

function run_h_ladders(case, yref, T, hs; km=40, nrep=1, Omega_c=nothing, rho=nothing)
    rs = runners(case, T; km, Omega_c, rho)
    by = Dict{String, Vector{NamedTuple}}()
    for (name, run) in rs
        by[name] = h_ladder(name, run, case, yref, T, hs; nrep)
    end
    return names_from_rs(rs), by
end

function names_from_rs(rs)
    names = String[]
    for r in rs
        push!(names, r[1])
    end
    return names
end

function report_cost(by, names, tols; primary=nothing, nel=0, N=0, k=0)
    println("\n=== Cheapest wall to reach relative L2 tol  n_el=$nel ===")
    print(@sprintf("%8s", "tol"))
    for name in names
        print(@sprintf("  %10s %10s", name, "h"))
    end
    println()
    for tol in tols
        print(@sprintf("%8.0e", tol))
        for name in names
            r = cheapest(by[name], tol)
            if r === nothing
                print(@sprintf("  %10s %10s", "—", "—"))
            else
                print(@sprintf("  %10.3f %10.6g", r.wall, r.h))
            end
        end
        println()
    end
    if primary !== nothing
        ranked = Tuple[]
        for n in names
            push!(ranked, (n, cheapest(by[n], primary)))
        end
        kept = Tuple[]
        for (n, r) in ranked
            if r !== nothing
                push!(kept, (n, r))
            end
        end
        ranked = sort(kept; by=wall_of_pair)
        if !isempty(ranked)
            best = ranked[1][2].wall
            println("\n=== Ranking at $primary  (n_el=$nel, N=$N, k=$k) ===")
            for (i, (name, r)) in enumerate(ranked)
                @printf("  %d. %-10s wall=%.3fs  %5d steps  h=%-9.6g (%.2fx)\n",
                        i, name, r.wall, r.nsteps, r.h, r.wall / best)
            end
        end
    end
    println("\n=== Winner per tol ===")
    for tol in tols
        cands = Tuple[]
        for n in names
            push!(cands, (n, cheapest(by[n], tol)))
        end
        kept = Tuple[]
        for (n, r) in cands
            if r !== nothing
                push!(kept, (n, r))
            end
        end
        cands = kept
        if isempty(cands)
            @printf("  tol=%8.0e  (none)\n", tol)
            continue
        end
        best_c = cands[1]
        for c in cands
            if c[2].wall < best_c[2].wall
                best_c = c
            end
        end
        n, r = best_c
        @printf("  tol=%8.0e  %-10s  wall=%.3fs  h=%g  err=%.3e\n", tol, n, r.wall, r.h, r.err)
    end
end

function wall_of_pair(pair)
    return pair[2].wall
end

function write_cost_json(path, case, T, href, km, nrep, tols, names, by; Omega_c=nothing)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "{")
        @printf(io, "  \"nel\": %d, \"N\": %d, \"k\": %d, \"p\": %d, \"T\": %.1f,\n",
                case.nel, case.N, case.k, case.p, T)
        @printf(io, "  \"href\": %.2e, \"km\": %d, \"nrep\": %d,\n", href, km, nrep)
        if Omega_c === nothing
            println(io, "  \"Omega_c\": null,")
        else
            @printf(io, "  \"Omega_c\": %.6e,\n", Omega_c)
        end
        print(io, "  \"tols\": [")
        join(io, tols, ", ")
        println(io, "],")
        print(io, "  \"order\": [")
        name_first = true
        for n in names
            if !name_first
                print(io, ", ")
            end
            name_first = false
            print(io, "\"$n\"")
        end
        println(io, "],")
        println(io, "  \"methods\": {")
        for (mi, name) in enumerate(names)
            println(io, "    \"$name\": [")
            for (ri, r) in enumerate(by[name])
                @printf(io, "      {\"h\": %.8g, \"nsteps\": %d, \"err\": %.16e, \"wall\": %.8g}",
                        r.h, r.nsteps, r.err, r.wall)
                if ri < length(by[name])
                    println(io, ",")
                else
                    println(io, "")
                end
            end
            print(io, "    ]")
            if mi < length(names)
                println(io, ",")
            else
                println(io, "")
            end
        end
        println(io, "  },")
        println(io, "  \"optima\": {")
        for (mi, name) in enumerate(names)
            println(io, "    \"$name\": {")
            for (ti, tol) in enumerate(tols)
                r = cheapest(by[name], tol)
                key = @sprintf("%.0e", tol)
                if r === nothing
                    print(io, "      \"$key\": null")
                else
                    @printf(io, "      \"%s\": {\"wall\": %.8g, \"err\": %.16e, \"h\": %.8g, \"nsteps\": %d}",
                            key, r.wall, r.err, r.h, r.nsteps)
                end
                if ti < length(tols)
                    println(io, ",")
                else
                    println(io, "")
                end
            end
            print(io, "    }")
            if mi < length(names)
                println(io, ",")
            else
                println(io, "")
            end
        end
        println(io, "  }")
        println(io, "}")
    end
    println("wrote $path")
end

function write_scaling_json(path, prob, names, rows)
    mkpath(dirname(path))
    Oc = get(prob, :Omega_c, nothing)
    open(path, "w") do io
        println(io, "{")
        @printf(io, "  \"h\": %.8g, \"href\": %.8g, \"T\": %.1f, \"km\": %d, \"nrep\": %d,\n",
                prob.h, prob.href, prob.T, prob.km, prob.nrep)
        @printf(io, "  \"rho\": %.2f,\n", prob.rho)
        if Oc === nothing
            println(io, "  \"Omega_c\": null,")
        else
            @printf(io, "  \"Omega_c\": %.6e,\n", Oc)
        end
        print(io, "  \"nels\": [")
        join(io, prob.nels, ", ")
        println(io, "],")
        print(io, "  \"order\": [")
        name_first = true
        for n in names
            if !name_first
                print(io, ", ")
            end
            name_first = false
            print(io, "\"$n\"")
        end
        println(io, "],")
        println(io, "  \"methods\": {")
        for (mi, name) in enumerate(names)
            println(io, "    \"$name\": [")
            for (ri, r) in enumerate(rows[name])
                @printf(io,
                    "      {\"nel\": %d, \"N\": %d, \"k\": %d, \"h\": %.8g, \"nsteps\": %d, \"err\": %.16e, \"wall\": %.8g}",
                    r.nel, r.N, r.k, r.h, r.nsteps, r.err, r.wall)
                if ri < length(rows[name])
                    println(io, ",")
                else
                    println(io, "")
                end
            end
            print(io, "    ]")
            if mi < length(names)
                println(io, ",")
            else
                println(io, "")
            end
        end
        println(io, "  }")
        println(io, "}")
    end
    println("wrote $path")
end
