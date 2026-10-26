# MIISO4 vs exprb43 -> out/frames.json + out/export/*.bin

include(joinpath(@__DIR__, "..", "setup.jl"))
using JSON

nel, p, alpha, amp, ngrid = 12, 4, 1.0, 0.2, 64

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

function sample_u(mem, q, xs, ys)
    knots = open_knots(mem.nel, mem.p)
    n1 = mem.m1 + 2
    C = zeros(n1, n1)
    C[2:n1 - 1, 2:n1 - 1] .= reshape(q, mem.m1, mem.m1)
    U = zeros(length(ys), length(xs))
    for (j, x) in enumerate(xs), (i, y) in enumerate(ys)
        sx = findspan(n1, mem.p, x, knots)
        sy = findspan(n1, mem.p, y, knots)
        Nx, _ = dersbasisfuns(sx, x, mem.p, knots)
        Ny, _ = dersbasisfuns(sy, y, mem.p, knots)
        s = 0.0
        for a in 0:mem.p, b in 0:mem.p
            s += Nx[a + 1] * Ny[b + 1] * C[sx - mem.p + a, sy - mem.p + b]
        end
        U[i, j] = s
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
k = M2.outlier_count(mem.setup)
q0 = corner_bump_ic(mem, amp)
y0 = vcat(q0, zeros(N))
f, Jac = membrane_rhs(mem)
ft = zeros(2N)

M = Matrix(mem.M)
F0 = eigen(Symmetric(Matrix(tangent_sparse(mem, zeros(N)))), Symmetric(M))
omega = sqrt.(F0.values)
phi = F0.vectors
n_res = N - k
phi_out = phi[:, n_res + 1:N]
h = 2 * pi / omega[n_res]
T = 16 * h

function integrate(kw)
    ts = Float64[]
    qs = Vector{Float64}[]
    function cb(t, y)
        push!(ts, float(t))
        push!(qs, copy(@view y[1:N]))
        return nothing
    end
    M4.miiso4_solve(f, copy(y0), T, h; kw..., callback=cb)
    return ts, qs
end

ts_m, qs_m = integrate((; setup=mem.setup, J=Jac, ft=ft, krylov_dim_max=40, cutoff_adaptive=true))
ts_e, qs_e = integrate((; Omega_c=10 * h * omega[end], J=Jac, ft=ft, krylov_dim_max=40, cutoff_adaptive=false))

function pack(qs)
    E_res = Float64[]
    E_out = Float64[]
    u_out = Vector{Float64}[]
    for q in qs
        c = phi' * (M * q)
        push!(E_res, norm(c[1:n_res]))
        push!(E_out, norm(c[n_res + 1:N]))
        push!(u_out, phi_out * (phi_out' * (M * q)))
    end
    return E_res, E_out, u_out
end

E_res_m, E_out_m, u_out_m = pack(qs_m)
E_res_e, E_out_e, u_out_e = pack(qs_e)

xs = collect(range(0.0, 1.0; length=ngrid))
ys = copy(xs)
function sample_stack(qs)
    U = zeros(length(qs), ngrid, ngrid)
    for (i, q) in enumerate(qs)
        U[i, :, :] .= sample_u(mem, q, xs, ys)
    end
    return U
end
U_m = sample_stack(qs_m)
U_e = sample_stack(qs_e)
U_out_m = sample_stack(u_out_m)
U_out_e = sample_stack(u_out_e)
X = repeat(reshape(xs, 1, :), ngrid, 1)
Y = repeat(reshape(ys, :, 1), 1, ngrid)

expdir = joinpath(@__DIR__, "out", "export")
for (name, A) in (
    ("X.bin", X), ("Y.bin", Y),
    ("U_miiso.bin", U_m), ("U_exprb.bin", U_e),
    ("Uout_miiso.bin", U_out_m), ("Uout_exprb.bin", U_out_e),
    ("times_miiso.bin", Float64.(ts_m)),
    ("times_exprb.bin", Float64.(ts_e)),
)
    write_f64(joinpath(expdir, name), A)
end

open(joinpath(@__DIR__, "out", "frames.json"), "w") do io
    JSON.print(io, Dict(
        "times_miiso" => ts_m, "times_exprb" => ts_e,
        "E_res_miiso" => E_res_m, "E_out_miiso" => E_out_m,
        "E_res_exprb" => E_res_e, "E_out_exprb" => E_out_e,
    ))
end
