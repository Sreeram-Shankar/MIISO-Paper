# MIISO4 vs exprb43 (same stages; exprb = MIISO4 with cutoff above the spectrum).
# Setup that shows selective damping: nel=12, corner_bump, spectral h = 2π/ω_last_res.
# No gen-α (Jesse: compare MIISO to exprb).
#
#   julia --project=. -t 4 --startup-file=no test/membrance/outlier_visual/run.jl
#   python3 test/membrance/outlier_visual/plot.py
#   python3 test/membrance/outlier_visual/video_corner.py

include(joinpath(@__DIR__, "..", "setup.jl"))
using JSON

nel, p = 12, 4
alpha = 1.0
amp = 0.2
ngrid = 64

function findspan(n, p, u, U)
    if u >= U[n + 1]
        return n
    end
    low = p + 1
    high = n + 1
    mid = (low + high) ÷ 2
    while u < U[mid] || u >= U[mid + 1]
        if u < U[mid]
            high = mid
        else
            low = mid
        end
        mid = (low + high) ÷ 2
    end
    return mid
end

function full_controls(mem, q)
    n1 = mem.m1 + 2
    C = zeros(n1, n1)
    C[2:n1 - 1, 2:n1 - 1] .= reshape(q, mem.m1, mem.m1)
    return C
end

function sample_u(mem, q, xs, ys)
    knots = open_knots(mem.nel, mem.p)
    n1 = mem.m1 + 2
    pdeg = mem.p
    C = full_controls(mem, q)
    U = zeros(length(ys), length(xs))
    for (j, ξ) in enumerate(xs)
        for (i, η) in enumerate(ys)
            sx = findspan(n1, pdeg, ξ, knots)
            sy = findspan(n1, pdeg, η, knots)
            Nx, _ = dersbasisfuns(sx, ξ, pdeg, knots)
            Ny, _ = dersbasisfuns(sy, η, pdeg, knots)
            s = 0.0
            for a in 0:pdeg
                for b in 0:pdeg
                    @inbounds s += Nx[a + 1] * Ny[b + 1] * C[sx - pdeg + a, sy - pdeg + b]
                end
            end
            U[i, j] = s
        end
    end
    return U
end

function write_f64(path, A::Array{Float64})
    open(path, "w") do io
        write(io, Int64(ndims(A)))
        for d in size(A)
            write(io, Int64(d))
        end
        write(io, A)
    end
end

mem = membrane(nel, p, alpha)
N = mem.N
m1 = mem.m1
k = M2.outlier_count(mem.setup)
q0 = corner_bump_ic(mem, amp)
y0 = vcat(q0, zeros(N))
f, Jac = membrane_rhs(mem)
ft = zeros(2N)

M = Matrix(mem.M)
K = Matrix(tangent_sparse(mem, zeros(N)))
F0 = eigen(Symmetric(K), Symmetric(M))
ω = sqrt.(max.(F0.values, 0.0))
Φ = F0.vectors
nres = N - k
Φout = Φ[:, nres + 1:N]

h = 2π / ω[nres]
T = 16 * h
Ωc = h * ω[nres + 1]

function band(q)
    c = Φ' * (M * q)
    return (norm(c[1:nres]), norm(c[nres + 1:N]))
end
Eres0, Eout0 = band(q0)
Eq0 = sqrt(q0' * (M * q0))

@printf("corner-bump IC  nel=%d  p=%d  N=%d  k=%d  alpha=%.1f  amp=%.3g\n",
        nel, p, N, k, alpha, amp)
@printf("h=%.4g  T=%.4g  steps=%d  Ωc_ref=%.4g  (spectral h, MIISO4 adaptive cutoff)\n",
        h, T, round(Int, T / h), Ωc)
@printf("E_res(0)=%.4e  E_out(0)=%.4e  (%.2f%% of |q|_M)\n",
        Eres0, Eout0, 100 * Eout0 / Eq0)
flush(stdout)

function project_out(q)
    c = Φout' * (M * q)
    return (Φout * c, norm(c))
end

function integrate_miiso4(label, solve_kw)
    ts = Float64[]
    qs = Vector{Float64}[]
    crashed = Ref(false)
    crash_t = Ref(NaN)
    function cb(t, y)
        q = @view y[1:N]
        if !all(isfinite, q)
            crashed[] = true
            crash_t[] = float(t)
            return nothing
        end
        push!(ts, float(t))
        push!(qs, copy(q))
        return nothing
    end
    println(label, "...")
    flush(stdout)
    try
        M4.miiso4_solve(f, copy(y0), T, h; solve_kw..., callback=cb)
    catch e
        crashed[] = true
        if isempty(ts)
            crash_t[] = 0.0
        else
            crash_t[] = ts[end]
        end
        @printf("  %s CRASHED at t≈%.4g  (%s)\n", label, crash_t[], typeof(e))
        flush(stdout)
    end
    if crashed[] && isnan(crash_t[])
        if isempty(ts)
            crash_t[] = 0.0
        else
            crash_t[] = ts[end]
        end
        @printf("  %s CRASHED at t≈%.4g  (non-finite state)\n", label, crash_t[])
        flush(stdout)
    end
    return ts, qs, crashed[], crash_t[]
end

# MIISO4. adaptive cutoff (IRAM + Rayleigh). Same spectral h.
# exprb43. same stages, cutoff above the spectrum (no selective damping).
miiso_kw = (; setup=mem.setup, J=Jac, ft=ft, krylov_dim_max=40, cutoff_adaptive=true)
exprb_kw = (; Omega_c=10 * h * ω[end], J=Jac, ft=ft, krylov_dim_max=40, cutoff_adaptive=false)

ts_m, qs_m, crash_m, tcrash_m = integrate_miiso4("MIISO4", miiso_kw)
ts_e, qs_e, crash_e, tcrash_e = integrate_miiso4("exprb43", exprb_kw)

function pack(qs)
    Eres = Float64[]
    Eout = Float64[]
    u = Vector{Float64}[]
    uo = Vector{Float64}[]
    for q in qs
        Er, Eo = band(q)
        qo, _ = project_out(q)
        push!(Eres, Er)
        push!(Eout, Eo)
        push!(u, q)
        push!(uo, qo)
    end
    return Eres, Eout, u, uo
end

Er_m, Eo_m, u_m, uo_m = pack(qs_m)
Er_e, Eo_e, u_e, uo_e = pack(qs_e)

if crash_m
    status_m = "CRASH"
else
    status_m = "ok"
end
if crash_e
    status_e = "CRASH"
else
    status_e = "ok"
end
@printf("status   MIISO4=%s  exprb43=%s\n", status_m, status_e)
if isempty(Eo_m)
    eout_end_m = NaN
else
    eout_end_m = Eo_m[end]
end
if isempty(Eo_e)
    eout_end_e = NaN
else
    eout_end_e = Eo_e[end]
end
@printf("E_out(end) MIISO4=%.3e  exprb43=%.3e\n", eout_end_m, eout_end_e)
if isempty(Er_m)
    eres_end_m = NaN
else
    eres_end_m = Er_m[end]
end
if isempty(Er_e)
    eres_end_e = NaN
else
    eres_end_e = Er_e[end]
end
@printf("E_res(end) MIISO4=%.3e  exprb43=%.3e\n", eres_end_m, eres_end_e)

xs = collect(range(0.0, 1.0; length=ngrid))
ys = copy(xs)
function sample_stack(qs)
    nF = length(qs)
    U = zeros(nF, ngrid, ngrid)
    for (i, q) in enumerate(qs)
        U[i, :, :] .= sample_u(mem, q, xs, ys)
    end
    return U
end
println("sampling fine fields ngrid=$ngrid …")
flush(stdout)
Um = sample_stack(u_m)
Ue = sample_stack(u_e)
Uom = sample_stack(uo_m)
Uoe = sample_stack(uo_e)
X = repeat(reshape(xs, 1, :), ngrid, 1)
Y = repeat(reshape(ys, :, 1), 1, ngrid)

function finite_max(A)
    if isempty(A)
        return 0.0
    end
    return maximum(abs, A)
end
umax = max(finite_max(Um), finite_max(Ue))
uomax = max(finite_max(Uom), finite_max(Uoe))

function nan2null(x)
    if x isa Number && !isfinite(x)
        return nothing
    end
    return x
end
outdir = joinpath(@__DIR__, "out")
expdir = joinpath(outdir, "export")
mkpath(expdir)
for (name, A) in (
    ("X.bin", X), ("Y.bin", Y),
    ("U_miiso.bin", Um), ("U_exprb.bin", Ue),
    ("Uout_miiso.bin", Uom), ("Uout_exprb.bin", Uoe),
    ("times_miiso.bin", Float64.(ts_m)),
    ("times_exprb.bin", Float64.(ts_e)),
)
    write_f64(joinpath(expdir, name), A)
end

open(joinpath(outdir, "frames.json"), "w") do io
    JSON.print(io, Dict(
        "nel" => nel, "p" => p, "N" => N, "k" => k, "m1" => m1,
        "ngrid" => ngrid, "umax" => umax, "uomax" => uomax,
        "alpha" => alpha, "amp" => amp,
        "T" => T, "h" => h, "Omega_c" => Ωc,
        "ic" => "corner_bump",
        "methods" => ["MIISO4", "exprb43"],
        "E_res0" => Eres0, "E_out0" => Eout0,
        "crash_miiso" => crash_m, "crash_exprb" => crash_e,
        "tcrash_miiso" => nan2null(tcrash_m),
        "tcrash_exprb" => nan2null(tcrash_e),
        "times_miiso" => ts_m, "times_exprb" => ts_e,
        "E_res_miiso" => Er_m, "E_out_miiso" => Eo_m,
        "E_res_exprb" => Er_e, "E_out_exprb" => Eo_e,
        "u_miiso" => u_m, "u_exprb" => u_e,
        "uout_miiso" => uo_m, "uout_exprb" => uo_e,
    ))
end
println("wrote ", joinpath(outdir, "frames.json"))
println("wrote fine fields → ", expdir)
