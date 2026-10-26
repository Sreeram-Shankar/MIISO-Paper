# Mesh scaling at fixed h -> SCALE.out

include(joinpath(@__DIR__, "setup.jl"))

prob = SCALE
nsteps = round(Int, prob.T / prob.h)
h_eff = prob.T / nsteps

methods = Dict{String,Any}()
first = true
for nel in prob.nels
    case = make_case(nel, prob.p, prob.alpha, prob.amp)
    yref = feagin12_ref(case.f, case.y0, prob.T, float(prob.href))
    rs = runners(case, prob.T; km=prob.km, rho=prob.rho)
    if first
        for (name, _) in rs
            methods[name] = []
        end
        first = false
    end
    for (name, run) in rs
        walls = Float64[]
        local y
        for _ in 1:prob.nrep
            push!(walls, @elapsed y = run(h_eff))
        end
        e = l2_rel(case.mem, @view(y[1:case.N]), @view(yref[1:case.N]))
        push!(methods[name], Dict("nel" => nel, "N" => case.N, "err" => e, "wall" => minimum(walls)))
    end
end

dump_json(prob.out, Dict("h" => h_eff, "methods" => methods))
