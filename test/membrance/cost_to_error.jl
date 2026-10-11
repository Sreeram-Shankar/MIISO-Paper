# Cost vs accuracy — adaptive cutoff (IRAM), COST mesh (20×20).
#
#   julia -t 6 --startup-file=no --project=. test/membrance/cost_to_error.jl

include(joinpath(@__DIR__, "setup.jl"))

function main()
    prob = COST
    case = make_case(prob)
    (; N, k) = case
    Oc = get(prob, :Omega_c, nothing)

    println("=== Cost vs accuracy: nonlinear 2D IGA membrane ===")
    println("problem=$(prob.label)  p=$(prob.p), $(prob.nel)×$(prob.nel), N=$N, k=$k")
    println("alpha=$(prob.alpha), A=$(prob.amp), T=$(prob.T)")
    print_spectrum(case)
    if Oc === nothing
        println("MIISO: adaptive cutoff ON; Krylov m=$(prob.km)")
    else
        println("MIISO: filter OFF  Omega_c=$Oc; Krylov m=$(prob.km)")
    end
    println("gen-α ρ∞=$(prob.rho)")
    println("reference: Feagin12 Float64 dt=$(prob.href); wall = best of $(prob.nrep)")
    println("h ∈ $(prob.hs)")

    println("Feagin12 reference dt=$(prob.href) ...")
    yref = feagin12_ref(case.f, case.y0, prob.T, prob.href)
    println("done  ||q||=$(round(norm(@view(yref[1:N])), digits=6))")

    names, by = run_h_ladders(case, yref, prob.T, prob.hs;
                              km=prob.km, nrep=prob.nrep, Omega_c=Oc, rho=prob.rho)
    report_cost(by, names, prob.tols; primary=prob.primary, nel=prob.nel, N=N, k=k)
    write_cost_json(prob.out, case, prob.T, prob.href, prob.km, prob.nrep,
                    prob.tols, names, by; Omega_c=Oc)
end

main()
