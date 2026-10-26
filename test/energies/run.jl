#this test measures the energies of the resolved and outlier modes
#for the linear membrane 

include(joinpath(@__DIR__, "..", "membrance", "setup.jl"))
using JSON
using OrdinaryDiffEqFIRK
using SciMLBase: DiscreteCallback

#parameters for the linear membrane
nel, p = 8, 4
mem = membrane(nel, p, 0.0)
N = mem.N
k = M2.outlier_count(mem.setup)
M = Matrix(mem.M)
K = Matrix(tangent_sparse(mem, zeros(N)))
F = eigen(Symmetric(K), Symmetric(M))
omega = sqrt.(F.values)
phi = F.vectors
phi_res = phi[:, 1:N - k]

q0 = phi[:, 1] + (0.05 / sqrt(k)) * vec(sum(phi[:, N - k + 1:N]; dims=2))
y0 = vcat(q0, zeros(N))

function band(q)
    c = phi' * (M * q)
    return norm(c[1:N - k]), norm(c[N - k + 1:N])
end

#time paraameters
T = 16 * pi / omega[1]
h = 2 * pi / omega[N - k]
omega_c = h * omega[N - k + 1]
f, Jac = membrane_rhs(mem)
ft = zeros(2N)
miiso_kw = (; Omega_c=omega_c, J=Jac, ft=ft, cutoff_adaptive=false, krylov_dim_max=40)
exprb_kw = (; J=Jac, ft=ft, cutoff_adaptive=false, krylov_dim_max=40,
            Omega_c=10 * h * omega[end])

#project the resolved modes onto the resolved subspace
function project_resolved!(integrator)
    q = @view integrator.u[1:N]
    v = @view integrator.u[N + 1:2N]
    q .= phi_res * (phi_res' * (M * q))
    v .= phi_res * (phi_res' * (M * v))
    return nothing
end

#this uses the outlier removal method of Hiemstra et al. and a 3rd order Gauss-Legendre FIRK, which has the 
#same stability function as MIISO5 and 6 on the resolved modes
function spatial_gl3()
    y = copy(y0)
    y[1:N] .= phi_res * (phi_res' * (M * @view(y[1:N])))
    y[N + 1:2N] .= phi_res * (phi_res' * (M * @view(y[N + 1:2N])))
    cb = DiscreteCallback((u, t, integrator) -> true, project_resolved!;
                          save_positions=(false, false))
    sol = solve(ODEProblem((du, u, p, t) -> (du .= f(t, u)), y, (0.0, float(T))),
                GaussLegendre(num_stages=3); dt=h, adaptive=false,
                save_everystep=false, callback=cb)
    return sol.u[end]
end

#every test case where we measure the energies of the resolved and outlier modes
jobs = (
    ("spatial+GL3", spatial_gl3),
    ("MIISO2", () -> M2.miiso2_solve(f, y0, T, h; miiso_kw...)),
    ("MIISO3", () -> M3.miiso3_solve(f, y0, T, h; miiso_kw...)),
    ("MIISO4", () -> M4.miiso4_solve(f, y0, T, h; miiso_kw...)),
    ("MIISO5", () -> M5.miiso5_solve(f, y0, T, h; miiso_kw...)),
    ("MIISO6", () -> M6.miiso6_solve(f, y0, T, h; miiso_kw...)),
    ("exprb65", () -> M6.miiso6_solve(f, y0, T, h; exprb_kw...)),
    ("gen-a_0", () -> genalpha_solve(q -> f_int(mem, q), q -> tangent_sparse(mem, q),
        mem.M, r -> mass_solve(mem, r), q0, zeros(N), T, h; rho_inf=0.0)[1]),
    ("gen-a_0.5", () -> genalpha_solve(q -> f_int(mem, q), q -> tangent_sparse(mem, q),
        mem.M, r -> mass_solve(mem, r), q0, zeros(N), T, h; rho_inf=0.5)[1]),
    ("gen-a_1", () -> genalpha_solve(q -> f_int(mem, q), q -> tangent_sparse(mem, q),
        mem.M, r -> mass_solve(mem, r), q0, zeros(N), T, h; rho_inf=1.0)[1]),
)

E_res0, E_out0 = band(q0)
rows = []
for (name, run) in jobs
    E_res, E_out = band(@view(run()[1:N]))
    push!(rows, Dict("method" => name, "E_res" => E_res, "E_out" => E_out))
end

open(joinpath(@__DIR__, "out", "energies.json"), "w") do io
    JSON.print(io, Dict("E_res0" => E_res0, "E_out0" => E_out0, "results" => rows))
end
