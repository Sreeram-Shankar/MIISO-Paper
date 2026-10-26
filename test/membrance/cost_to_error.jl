# Cost vs accuracy -> COST.out

include(joinpath(@__DIR__, "setup.jl"))

prob = COST
case = make_case(prob)
yref = feagin12_ref(case.f, case.y0, prob.T, prob.href)
rs = runners(case, prob.T; km=prob.km, rho=prob.rho)

methods = Dict{String,Any}()
for (name, run) in rs
    rows = []
    for h in prob.hs
        nsteps = round(Int, prob.T / h)
        h_eff = prob.T / nsteps
        walls = Float64[]
        local y
        for _ in 1:prob.nrep
            push!(walls, @elapsed y = run(h_eff))
        end
        e = l2_rel(case.mem, @view(y[1:case.N]), @view(yref[1:case.N]))
        push!(rows, Dict("h" => h_eff, "err" => e, "wall" => minimum(walls)))
    end
    methods[name] = rows
end

dump_json(prob.out, Dict("methods" => methods))
