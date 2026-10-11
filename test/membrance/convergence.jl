# Temporal convergence — MIISO2–6 only, filter off, asymptotic h-window.
#
#   julia -t 6 --startup-file=no --project=. test/membrance/convergence.jl

include(joinpath(@__DIR__, "setup.jl"))

const FORMAL = Dict("MIISO2"=>2, "MIISO3"=>3, "MIISO4"=>4, "MIISO5"=>5, "MIISO6"=>6)
const FLOOR = 1e-11

function write_conv_json(path, case, prob, names, errs, pobs; Omega_c=nothing)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "{")
        @printf(io, "  \"nel\": %d, \"N\": %d, \"k\": %d, \"p\": %d, \"T\": %.1f,\n",
                case.nel, case.N, case.k, case.p, prob.T)
        @printf(io, "  \"href\": %.2e, \"km\": %d, \"floor\": %.2e,\n", prob.href, prob.km, FLOOR)
        if Omega_c === nothing
            println(io, "  \"Omega_c\": null,")
        else
            @printf(io, "  \"Omega_c\": %.6e,\n", Omega_c)
        end
        print(io, "  \"hs\": [")
        join(io, prob.hs, ", ")
        println(io, "],")
        print(io, "  \"order\": [")
        name_first = true
        for n in names
            if !name_first
                print(io, ", ")
            end
            name_first = false
            print(io, "\"$n\"")
        end
        println(io, "],")
        println(io, "  \"methods\": {")
        for (mi, name) in enumerate(names)
            es = errs[name]
            println(io, "    \"$name\": {")
            @printf(io, "      \"formal\": %d,\n", FORMAL[name])
            print(io, "      \"err\": [")
            err_first = true
            for e in es
                if !err_first
                    print(io, ", ")
                end
                err_first = false
                print(io, @sprintf("%.16e", e))
            end
            println(io, "],")
            print(io, "      \"p_obs\": [")
            obs_first = true
            for o in pobs[name]
                if !obs_first
                    print(io, ", ")
                end
                obs_first = false
                print(io, @sprintf("%.8g", o))
            end
            println(io, "]")
            print(io, "    }")
            if mi < length(names)
                println(io, ",")
            else
                println(io, "")
            end
        end
        println(io, "  }")
        println(io, "}")
    end
    println("wrote $path")
end

function main()
    prob = CONV
    case = make_case(prob)
    (; N, setup, k) = case
    Oc = get(prob, :Omega_c, nothing)
    rs = runners(case, prob.T; km=prob.km, Omega_c=Oc)

    println("=== Temporal convergence: nonlinear 2D IGA membrane ===")
    println("problem=$(prob.label)  p=$(prob.p), $(prob.nel)×$(prob.nel), N=$N, setup=$setup")
    println("Hiemstra k=$k, alpha=$(prob.alpha), A=$(prob.amp), T=$(prob.T)")
    println("MIISO only; filter OFF  Omega_c=$Oc; km=$(prob.km)")
    print_spectrum(case)
    println("h ∈ $(prob.hs)  (asymptotic window; stop p_obs at err ≤ $FLOOR)")
    println()

    println("Feagin12 reference dt=$(prob.href) ...")
    yref = feagin12_ref(case.f, case.y0, prob.T, prob.href)
    println("done  ||q||=$(round(norm(@view(yref[1:N])), digits=6))")

    errs = Dict{String, Vector{Float64}}()
    for (name, run) in rs
        println("\n--- $name ---")
        @printf("%10s %7s %14s %8s\n", "h", "steps", "rel L2 err", "p_obs")
        es = Float64[]
        prev = NaN
        for h in prob.hs
            nsteps = max(round(Int, prob.T / h), 1)
            h_eff = prob.T / nsteps
            y = run(h_eff)
            e = l2_rel(case.mem, @view(y[1:N]), @view(yref[1:N]))
            if isnan(prev)
                pobs = ""
            elseif prev <= FLOOR || e <= FLOOR
                pobs = "   (floor)"
            else
                pobs = @sprintf("%8.2f", log(prev / e) / log(2))
            end
            @printf("%10.6g %7d %14.4e %8s\n", h_eff, nsteps, e, pobs)
            push!(es, e)
            prev = e
        end
        errs[name] = es
    end

    names = String[]
    for r in rs
        push!(names, r[1])
    end
    pobs = Dict{String, Vector{Float64}}()
    println("\n--- observed orders (err > $FLOOR) ---")
    println("We stop at $FLOOR and below since truncation error and the like have significant effects.")
    @printf("%-10s %8s  %s\n", "method", "formal", "p_obs")
    for name in names
        es = errs[name]
        ords = Float64[]
        for i in 1:length(es) - 1
            if es[i] <= FLOOR || es[i + 1] <= FLOOR
                break
            end
            push!(ords, log(es[i] / es[i + 1]) / log(2))
        end
        pobs[name] = ords
        if isempty(ords)
            ordstr = "—"
        else
            ord_parts = String[]
            for o in ords
                push!(ord_parts, @sprintf("%.2f", o))
            end
            ordstr = join(ord_parts, " ")
        end
        @printf("%-10s %8d  %s\n", name, FORMAL[name], ordstr)
    end

    write_conv_json(prob.out, case, prob, names, errs, pobs; Omega_c=Oc)
end

main()
