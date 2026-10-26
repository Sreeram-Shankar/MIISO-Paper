##this test measures the validity of the cutoff tracking method

include(joinpath(@__DIR__, "..", "membrance", "setup.jl"))
using JSON

#parameters for an iga setup, amps is the ampltiude of the 
#nonlinear membrane described in section 6 of the paper
p, alpha, h = 4, 1.0, 5e-4
amps = (0.20, 0.30, 0.35)
nel, T = 6, 2.5
n_step = round(Int, T / h)

#eigendecomposition of the current operators at a time
function live_spectrum(mem, q)
    F = eigen(Symmetric(Matrix(tangent_sparse(mem, q))), Symmetric(Matrix(mem.M)))
    omega = sqrt.(F.values)
    return omega, F.vectors, Symmetric(Matrix(mem.M))
end

case = make_case(nel, p, alpha, amps[1])
mem, N, k = case.mem, case.N, case.k

#for each ampltide, we run the simulation with MIISO2 and measure the gap and leakage as defined in the paper
by_A = Dict{String,Any}()
for A in amps
    q0 = mode_ic(mem, A)
    y = vcat(q0, zeros(N))
    f, Jac = membrane_rhs(mem)
    ft0 = zeros(2N)
    _, V0, Mass = live_spectrum(mem, q0)
    V_phys = V0[:, 1:N - k]
    t_hist = zeros(n_step + 1)
    gap = zeros(n_step + 1)
    leak = zeros(n_step + 1)
    t = 0.0
    for i in 1:n_step + 1
        q = @view y[1:N]
        omega, V, Mass = live_spectrum(mem, q)
        gap[i] = omega[N - k + 1] / omega[N - k] #ratio of the final resolved to the first outlier mode
        leak[i] = sum(abs2, V_phys' * Matrix(Mass) * V[:, N - k + 1:N]) #the principle angles between the resolved and outlier mode subspaces
        t_hist[i] = t
        if i == n_step + 1
            break
        end
        y = M2.miiso2_step(f, t, y, h; Omega_c=h * omega[N - k + 1], J=Jac(t, y), ft=ft0,
                           cutoff_adaptive=false, krylov_dim_max=40)
        t += h
    end
    by_A[string(A)] = Dict("t" => t_hist, "gap" => gap, "leak" => leak)
end

#save the results to a json file
open(joinpath(@__DIR__, "out", "outlier_tracking_validity.json"), "w") do io
    JSON.print(io, by_A)
end
