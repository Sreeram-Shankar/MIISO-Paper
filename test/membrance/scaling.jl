# Mesh scaling — fixed h, adaptive cutoff (IRAM), gen-α ρ∞=0.5.
#
#   julia -t 6 --startup-file=no --project=. test/membrance/scaling.jl

include(joinpath(@__DIR__, "setup.jl"))

function main()
    prob = SCALE
    Oc = get(prob, :Omega_c, nothing)
    h = float(prob.h)
    href = float(prob.href)
    nsteps = max(round(Int, prob.T / h), 1)
    h_eff = prob.T / nsteps

    println("=== Mesh scaling: nonlinear 2D IGA membrane ===")
    println("nels=$(prob.nels)  p=$(prob.p)  alpha=$(prob.alpha)  A=$(prob.amp)  T=$(prob.T)")
    println("fixed h=$(prob.h)  href=$(prob.href) (=h/10)  wall = best of $(prob.nrep)")
    if Oc === nothing
        println("MIISO: adaptive cutoff ON (IRAM); Krylov m=$(prob.km)")
    else
        println("MIISO: filter OFF  Omega_c=$Oc; Krylov m=$(prob.km)")
    end
    println("gen-α ρ∞=$(prob.rho)")

    rows = Dict{String, Vector{NamedTuple}}()
    names = String[]
    first = true
    for nel in prob.nels
        case = make_case(nel, prob.p, prob.alpha, prob.amp)
        (; N, k) = case
        println("\n=== nel=$nel  N=$N  k=$k ===")
        print_spectrum(case)
        println("Feagin12 reference dt=$href ...")
        yref = feagin12_ref(case.f, case.y0, prob.T, href)
        println("done  ||q||=$(round(norm(@view(yref[1:N])), digits=6))")

        rs = runners(case, prob.T; km=prob.km, Omega_c=Oc, rho=prob.rho)
        if first
            names = String[]
            for r in rs
                push!(names, r[1])
            end
            for name in names
                rows[name] = NamedTuple[]
            end
            first = false
        end
        for (name, run) in rs
            walls = Float64[]
            local y
            for _ in 1:prob.nrep
                push!(walls, @elapsed y = run(h_eff))
            end
            wall = minimum(walls)
            e = l2_rel(case.mem, @view(y[1:N]), @view(yref[1:N]))
            @printf("  %-8s  err=%.4e  wall=%.3fs\n", name, e, wall)
            push!(rows[name], (nel=nel, N=N, k=k, h=h_eff, nsteps=nsteps, err=e, wall=wall))
        end
    end

    println("\n=== Wall vs N @ h=$(prob.h) ===")
    @printf("%6s %6s", "nel", "N")
    for name in names
        print(@sprintf("  %10s", name))
    end
    println()
    for i in 1:length(prob.nels)
        r0 = rows[names[1]][i]
        @printf("%6d %6d", r0.nel, r0.N)
        for name in names
            @printf("  %10.3f", rows[name][i].wall)
        end
        println()
    end
    write_scaling_json(prob.out, prob, names, rows)
end

main()
