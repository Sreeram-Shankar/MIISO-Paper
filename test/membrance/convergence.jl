# Temporal convergence MIISO2-6 -> CONV.out

include(joinpath(@__DIR__, "setup.jl"))

prob = CONV
case = make_case(prob)
rs = runners(case, prob.T; km=prob.km, Omega_c=prob.Omega_c)
yref = feagin12_ref(case.f, case.y0, prob.T, prob.href)

methods = Dict{String,Any}()
for (name, run) in rs
    errs = Float64[]
    for h in prob.hs
        nsteps = round(Int, prob.T / h)
        y = run(prob.T / nsteps)
        push!(errs, l2_rel(case.mem, @view(y[1:case.N]), @view(yref[1:case.N])))
    end
    methods[name] = errs
end

dump_json(prob.out, Dict("hs" => collect(prob.hs), "err" => methods))
