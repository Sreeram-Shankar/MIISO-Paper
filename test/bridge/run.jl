# calls the setup of the bridge and runs it with MIISO4
# the JSON handling of this file and the scripting to run it on TACC
#were AI-assisted

include(joinpath(@__DIR__, "setup.jl"))
using JSON

T, h, km, dt_save = 11.5, 0.001, 40, 0.05
sys = arch_bridge_system(BridgeParams())
nf = sys.nfree
omega_c = h * sys.omega_c

mesh = sys.mesh
free = sys.free
n1, n2, n3 = sys.n1, sys.n2, sys.n3
cps = cps_matrix(mesh.control_points)
ws = Vector{Float64}(mesh.weights)
kv = map(v -> Vector{Float64}(v), mesh.knot_vectors)
orders = Int.(mesh.orders)
eta_top = kv[2][end]
zeta_mid = 0.5 * (kv[3][1] + kv[3][end])
Ucp, Uflat = make_U_workspace(n1, n2, n3)

ts = Float64[]
u_crown = Float64[]
next_save = Ref(0.0)

function cb(t, y)
    fill!(Uflat, 0.0)
    Uflat[free] .= view(y, 1:nf)
    _, uc = nurbs_point(kv, orders, cps, ws, 0.5, eta_top, zeta_mid, Ucp)
    if t + 0.5 * h >= next_save[] || t >= T - 0.5 * h
        push!(ts, t)
        push!(u_crown, hypot(uc...))
        next_save[] += dt_save
    end
    return nothing
end

miiso4_solve(sys.f, sys.y0, T, h; ft=sys.ft, J=sys.J,
             Omega_c=omega_c, cutoff_adaptive=true,
             krylov_dim_max=km, callback=cb)

open(joinpath(@__DIR__, "out", "summary.json"), "w") do io
    JSON.print(io, Dict("t" => ts, "u_crown" => u_crown))
end
