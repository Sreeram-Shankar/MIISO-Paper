# Linear membrane free vibration: resolved / outlier band energies.
# Paper setup (p=4, 8×8, k=36) plus MIISO5/6.

include(joinpath(@__DIR__, "..", "membrance", "setup.jl"))
using JSON
using OrdinaryDiffEqFIRK
using SciMLBase: DiscreteCallback

nel, p = 8, 4
mem = membrane(nel, p, 0.0)
N = mem.N
k = M2.outlier_count(mem.setup)
M = Matrix(mem.M)
K = Matrix(tangent_sparse(mem, zeros(N)))

F = eigen(Symmetric(K), Symmetric(M))
ω = sqrt.(max.(F.values, 0.0))
Φ = F.vectors
Φres = Φ[:, 1:N - k]

q0 = Φ[:, 1] + (0.05 / sqrt(k)) * vec(sum(Φ[:, N - k + 1:N]; dims=2))
v0 = zeros(N)
y0 = vcat(q0, v0)

function band_energies(q)
    c = Φ' * (M * q)
    return (norm(c[1:N - k]), norm(c[N - k + 1:N]))
end
Eres0, Eout0 = band_energies(q0)

T = 16π / ω[1]
h = 2π / ω[N - k]
Ωc = h * ω[N - k + 1]
nsteps = round(Int, T / h)
f, Jac = membrane_rhs(mem)
ft = zeros(2N)
miiso_kw = (; Omega_c=Ωc, J=Jac, ft=ft, cutoff_adaptive=false, krylov_dim_max=40)
# exprb65 ≈ MIISO6 stages with cutoff above the spectrum (no selective damping)
exprb_kw = (; J=Jac, ft=ft, cutoff_adaptive=false, krylov_dim_max=40,
            Omega_c=10 * h * ω[end])

function project_resolved!(integrator)
    q = @view integrator.u[1:N]
    v = @view integrator.u[N + 1:2N]
    cq = Φres' * (M * q)
    cv = Φres' * (M * v)
    q .= Φres * cq
    v .= Φres * cv
    return nothing
end

function always_true(u, t, integrator)
    return true
end

function ode_rhs!(du, u, p, t)
    du .= f(t, u)
    return du
end

function spatial_gl3(y0, T, h)
    y = copy(y0)
    cq = Φres' * (M * @view(y[1:N]))
    cv = Φres' * (M * @view(y[N + 1:2N]))
    y[1:N] .= Φres * cq
    y[N + 1:2N] .= Φres * cv
    cb = DiscreteCallback(always_true, project_resolved!;
                          save_positions=(false, false))
    prob = ODEProblem(ode_rhs!, y, (0.0, float(T)))
    sol = solve(prob, GaussLegendre(num_stages=3); dt=h, adaptive=false,
                save_everystep=false, callback=cb)
    return sol.u[end]
end

@printf("N=%d  k=%d  ω1=%.6g  ω_last_res=%.6g  ω_first_out=%.6g\n",
        N, k, ω[1], ω[N - k], ω[N - k + 1])
@printf("T=%.6g  h=%.6g  steps=%d  Ωc=%.6g\n", T, h, nsteps, Ωc)
@printf("E_res(0)=%.6g  E_out(0)=%.6g\n\n", Eres0, Eout0)

function run_spatial_gl3()
    return spatial_gl3(y0, T, h)
end

function run_miiso2()
    return M2.miiso2_solve(f, y0, T, h; miiso_kw...)
end

function run_miiso3()
    return M3.miiso3_solve(f, y0, T, h; miiso_kw...)
end

function run_miiso4()
    return M4.miiso4_solve(f, y0, T, h; miiso_kw...)
end

function run_miiso5()
    return M5.miiso5_solve(f, y0, T, h; miiso_kw...)
end

function run_miiso6()
    return M6.miiso6_solve(f, y0, T, h; miiso_kw...)
end

function run_exprb65()
    return M6.miiso6_solve(f, y0, T, h; exprb_kw...)
end

function genalpha_fint(q)
    return f_int(mem, q)
end

function genalpha_tangent(q)
    return tangent_sparse(mem, q)
end

function genalpha_mass(r)
    return mass_solve(mem, r)
end

function run_genalpha_0()
    return genalpha_solve(genalpha_fint, genalpha_tangent, mem.M, genalpha_mass,
                          q0, v0, T, h; rho_inf=0.0)[1]
end

function run_genalpha_05()
    return genalpha_solve(genalpha_fint, genalpha_tangent, mem.M, genalpha_mass,
                          q0, v0, T, h; rho_inf=0.5)[1]
end

function run_genalpha_1()
    return genalpha_solve(genalpha_fint, genalpha_tangent, mem.M, genalpha_mass,
                          q0, v0, T, h; rho_inf=1.0)[1]
end

jobs = (
    ("spatial + GL3", run_spatial_gl3),
    ("MIISO2", run_miiso2),
    ("MIISO3", run_miiso3),
    ("MIISO4", run_miiso4),
    ("MIISO5", run_miiso5),
    ("MIISO6", run_miiso6),
    ("exprb65", run_exprb65),
    ("gen-α (ρ∞=0)", run_genalpha_0),
    ("gen-α (ρ∞=0.5)", run_genalpha_05),
    ("gen-α (ρ∞=1)", run_genalpha_1),
)

results = Dict{String, Any}[]
@printf("%-18s %14s %14s\n", "Method", "E_res(T)", "E_out(T)")
for (name, run) in jobs
    y = run()
    Eres, Eout = band_energies(@view(y[1:N]))
    @printf("%-18s %14.6e %14.6e\n", name, Eres, Eout)
    push!(results, Dict("method" => name, "E_res" => Eres, "E_out" => Eout))
end

out = joinpath(@__DIR__, "out", "energies.json")
mkpath(dirname(out))
open(out, "w") do io
    JSON.print(io, Dict(
        "nel" => nel, "p" => p, "N" => N, "k" => k,
        "omega1" => ω[1], "omega_last_res" => ω[N - k], "omega_first_out" => ω[N - k + 1],
        "T" => T, "h" => h, "nsteps" => nsteps, "Omega_c" => Ωc,
        "E_res0" => Eres0, "E_out0" => Eout0,
        "results" => results,
    ), 2)
end
println("\nwrote ", out)
