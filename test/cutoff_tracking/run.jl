# Two separate 2D membrane diagnostics.
#
# 1) Gap + subspace leakage on a small mesh (fixed horizon, not period-based)
#      6×6, A ∈ {0.20, 0.30, 0.35}  →  outlier_tracking_validity.json
#
# 2) Exact vs Rayleigh ω_c table on a finer mesh (two return periods)
#      12×12, A = 0.20 only         →  rayleigh_error_A02.json / .csv
#
#   julia --project=. --startup-file=no test/cutoff_tracking/run.jl
#   python3 test/cutoff_tracking/plot.py

include(joinpath(@__DIR__, "..", "membrance", "setup.jl"))

using LinearAlgebra
using SparseArrays
using JSON
using Printf

const P = 4
const ALPHA = 1.0
const H = 5e-4
const OUTDIR = joinpath(@__DIR__, "out")

const NEL_LEAK = 6
const AMPS_LEAK = (0.20, 0.30, 0.35)
const T_LEAK = 2.5

const NEL_RAY = 12
const A_RAY = 0.20

function live_spectrum(mem, q)
    Kt = Matrix(tangent_sparse(mem, q))
    M = Matrix(mem.M)
    F = eigen(Symmetric(Kt), Symmetric(M))
    ω = sqrt.(max.(F.values, 0.0))
    return ω, F.vectors, Symmetric(M)
end

omega_c_exact(ω, k) = ω[length(ω) - k + 1]
gap_ratio(ω, k) = ω[length(ω) - k + 1] / ω[length(ω) - k]
leakage(V_phys, M, V_live) = sum(abs2, V_phys' * Matrix(M) * V_live)

"""One oscillation: time from release peak to the next same-sense peak of
the mass-weighted projection onto q0 (after passing through a trough)."""
function estimate_return_period(mem, k, A, h; T_search=5.0)
    N = mem.N
    q0 = mode_ic(mem, A)
    Mq0 = mem.M * q0
    nq0 = sqrt(dot(q0, Mq0))
    y = vcat(q0, zeros(N))
    f, Jac = membrane_rhs(mem)
    ft0 = zeros(2N)

    passed_trough = false
    c_prev2 = NaN
    c_prev = NaN
    t = 0.0
    n_search = round(Int, T_search / h)
    for _ in 1:n_search
        q = @view y[1:N]
        c = dot(q, Mq0) / (nq0^2)

        if !passed_trough && c < 0.2
            passed_trough = true
        elseif passed_trough && !isnan(c_prev2)
            # local maximum of c after the trough = next same-sense peak
            if c_prev >= c_prev2 && c_prev >= c && c_prev > 0.5
                return t - h
            end
        end
        c_prev2 = c_prev
        c_prev = c

        spec = live_spectrum(mem, q)
        ωc = omega_c_exact(spec[1], k)
        Jj = Jac(t, y)
        y = M2.miiso2_step(f, t, y, h; Omega_c=h * ωc, J=Jj, ft=ft0,
                           cutoff_adaptive=false, krylov_dim_max=40)
        t += h
    end
    error("no second peak within T_search=$T_search (A=$A)")
end

function run_amplitude(A, mem, k, h, n_step; track_rayleigh::Bool=false)
    N = mem.N
    q0 = mode_ic(mem, A)
    y = vcat(q0, zeros(N))
    f, Jac = membrane_rhs(mem)
    ft0 = zeros(2N)

    _, V0, M0 = live_spectrum(mem, q0)
    V_phys = V0[:, 1:N - k]

    cache = nothing
    yd = Vector{M2.JVPDual}(undef, 2N)
    if track_rayleigh
        cache = M2.init_cutoff(Jac(0.0, y), k, 2N, h)
    end

    nt = n_step + 1
    t_hist = zeros(nt)
    gap = zeros(nt)
    leak = zeros(nt)
    ω_exact = zeros(nt)
    ω_rayleigh = zeros(nt)

    t = 0.0
    for i in 1:nt
        q = @view y[1:N]
        ω, V, M = live_spectrum(mem, q)
        V_live = V[:, N - k + 1:N]
        ω_exact[i] = omega_c_exact(ω, k)
        gap[i] = gap_ratio(ω, k)
        leak[i] = leakage(V_phys, M, V_live)

        Jj = Jac(t, y)
        if track_rayleigh
            ω_rayleigh[i] = M2.step_cutoff!(cache, Jj, yd) / h
        end

        t_hist[i] = t
        if i == nt
            break
        end

        y = M2.miiso2_step(f, t, y, h; Omega_c=h * ω_exact[i], J=Jj, ft=ft0,
                           cutoff_adaptive=false, krylov_dim_max=40)
        t += h
        if i % max(n_step ÷ 10, 1) == 0
            @printf("  A=%.2f  step %d/%d  t=%.3f  ω_c=%.4f\n", A, i, n_step, t, ω_exact[i])
            flush(stdout)
        end
    end
    return (; t=t_hist, gap, leak, ω_exact, ω_rayleigh)
end

function summarize_leak(A, hist, k)
    i_gap = argmin(hist.gap)
    i_leak = argmax(hist.leak)
    println("\n--- A = $(round(A; digits=2))  (T=$(round(hist.t[end]; digits=4))) ---")
    println("  min gap ratio = $(round(hist.gap[i_gap]; digits=4))  at t = $(round(hist.t[i_gap]; digits=4))")
    println("  max leakage   = $(round(hist.leak[i_leak]; digits=4)) / $k",
            "  at t = $(round(hist.t[i_leak]; digits=4))")
end

function run_leakage_test()
    h = H
    T = T_LEAK
    n_step = max(round(Int, T / h), 1)
    case = make_case(NEL_LEAK, P, ALPHA, AMPS_LEAK[1])
    (; mem, N, setup, k) = case

    println("=== 1) Gap + leakage (small mesh, fixed T=$T) ===")
    println("p=$P, $(NEL_LEAK)×$(NEL_LEAK), N=$N, k=$k, A ∈ $AMPS_LEAK, h=$h, steps=$n_step")
    print_spectrum(case)

    by_A = Dict{Float64,Any}()
    for A in AMPS_LEAK
        println("\nintegrating A=$A to T=$T …"); flush(stdout)
        by_A[A] = run_amplitude(A, mem, k, h, n_step; track_rayleigh=false)
        summarize_leak(A, by_A[A], k)
    end

    by_A_json = Dict{String, Any}()
    for A in AMPS_LEAK
        by_A_json[string(A)] = Dict(
            "t" => collect(by_A[A].t),
            "gap_ratio" => collect(by_A[A].gap),
            "leakage" => collect(by_A[A].leak),
            "T" => T,
            "min_gap_ratio" => minimum(by_A[A].gap),
            "max_leakage" => maximum(by_A[A].leak),
            "t_min_gap" => by_A[A].t[argmin(by_A[A].gap)],
            "t_max_leakage" => by_A[A].t[argmax(by_A[A].leak)],
        )
    end

    track = Dict(
        "meta" => Dict(
            "problem" => "2d_nonlinear_membrane",
            "test" => "gap_leakage",
            "nel" => NEL_LEAK,
            "p" => P,
            "N" => N,
            "k" => k,
            "alpha" => ALPHA,
            "h" => h,
            "amplitudes" => collect(AMPS_LEAK),
            "T" => T,
            "integrator" => "MIISO2 with live exact pencil ω_c",
        ),
        "quantity" => "gap ratio ω_{N-k+1}/ω_{N-k} and subspace leakage Σ_j sin²θ_j(q)",
        "by_A" => by_A_json,
    )
    path = joinpath(OUTDIR, "outlier_tracking_validity.json")
    open(path, "w") do io
        JSON.print(io, track)
    end
    println("\nwrote $path")
end

function write_rayleigh_A02_table!(hist; stride::Int=50, nel::Int=NEL_RAY, N::Int=0, k::Int=0,
                                  T_period::Float64=NaN)
    t, ωe, ωr = hist.t, hist.ω_exact, hist.ω_rayleigh
    abs_err = abs.(ωr .- ωe)
    rel_err = abs_err ./ max.(ωe, 1e-30)

    idx = collect(1:stride:length(t))
    if t[end] != t[idx[end]]
        push!(idx, length(t))
    end
    rows = []
    for i in idx
        push!(rows, Dict(
            "t" => t[i],
            "omega_c_exact" => ωe[i],
            "omega_c_rayleigh" => ωr[i],
            "abs_err" => abs_err[i],
            "rel_err" => rel_err[i],
        ))
    end

    i_T_period = argmin(abs.(t .- T_period))

    table = Dict(
        "A" => A_RAY,
        "problem" => "2d_nonlinear_membrane",
        "test" => "rayleigh_vs_exact",
        "nel" => nel,
        "N" => N,
        "k" => k,
        "T_period" => T_period,
        "T" => t[end],
        "n_periods" => 2,
        "note" => "IRAM + Rayleigh tracked ω_c against the exact (K(q),M) cutoff at A=0.2 for two return periods",
        "columns" => ["t", "omega_c_exact", "omega_c_rayleigh", "abs_err", "rel_err"],
        "summary" => Dict(
            "max_abs_err" => maximum(abs_err),
            "max_rel_err" => maximum(rel_err),
            "mean_abs_err" => sum(abs_err) / length(abs_err),
            "mean_rel_err" => sum(rel_err) / length(rel_err),
            "abs_err_t0" => abs_err[1],
            "rel_err_t0" => rel_err[1],
            "abs_err_tT" => abs_err[end],
            "rel_err_tT" => rel_err[end],
            "abs_err_at_T_period" => abs_err[i_T_period],
            "rel_err_at_T_period" => rel_err[i_T_period],
        ),
        "t" => collect(t),
        "omega_c_exact" => collect(ωe),
        "omega_c_rayleigh" => collect(ωr),
        "abs_err" => collect(abs_err),
        "rel_err" => collect(rel_err),
        "table_rows" => rows,
    )

    path_json = joinpath(OUTDIR, "rayleigh_error_A02.json")
    open(path_json, "w") do io
        JSON.print(io, table)
    end
    println("wrote $path_json")

    path_csv = joinpath(OUTDIR, "rayleigh_error_A02.csv")
    open(path_csv, "w") do io
        println(io, "t,omega_c_exact,omega_c_rayleigh,abs_err,rel_err")
        for row in rows
            println(io, row["t"], ",", row["omega_c_exact"], ",", row["omega_c_rayleigh"], ",",
                    row["abs_err"], ",", row["rel_err"])
        end
    end
    println("wrote $path_csv")

    println("\n=== A = $A_RAY  exact ω_c vs Rayleigh ω_c ($(nel)×$(nel), 2 periods) ===")
    println("  T_period=$(round(T_period; digits=5))  T=$(round(t[end]; digits=5))")
    println(rpad("t", 10), rpad("ω_exact", 14), rpad("ω_Rayleigh", 14),
            rpad("|Δω|", 12), "|Δω|/ω")
    for row in rows
        @printf("%-10.4f%-14.6f%-14.6f%-12.4e%.4e\n",
                row["t"], row["omega_c_exact"], row["omega_c_rayleigh"],
                row["abs_err"], row["rel_err"])
    end
    s = table["summary"]
    println("\n  max |Δω|    = $(round(s["max_abs_err"]; sigdigits=5))")
    println("  max |Δω|/ω  = $(round(s["max_rel_err"]; sigdigits=5))")
    println("  mean |Δω|/ω = $(round(s["mean_rel_err"]; sigdigits=5))")
    return table
end

function run_rayleigh_test()
    h = H
    case = make_case(NEL_RAY, P, ALPHA, A_RAY)
    (; mem, N, setup, k) = case

    println("\n=== 2) Rayleigh vs exact (finer mesh, 2 periods) ===")
    println("p=$P, $(NEL_RAY)×$(NEL_RAY), N=$N, k=$k, A=$A_RAY, h=$h")
    print_spectrum(case)

    println("\nestimating return period for A=$A_RAY …"); flush(stdout)
    T_period = estimate_return_period(mem, k, A_RAY, h)
    T = 2 * T_period
    n_step = max(round(Int, T / h), 1)
    println("  T_period=$(round(T_period; digits=5))  →  T=$(round(T; digits=5))  ($n_step steps)")

    println("integrating A=$A_RAY for 2 periods …"); flush(stdout)
    hist = run_amplitude(A_RAY, mem, k, h, n_step; track_rayleigh=true)
    reR = abs.(hist.ω_rayleigh .- hist.ω_exact) ./ max.(hist.ω_exact, 1e-30)
    println("  max |ω_Rayleigh - ω_exact|/ω = $(round(maximum(reR); sigdigits=4))")

    # table stride ~ every 0.05 time units
    stride = max(round(Int, 0.05 / h), 1)
    write_rayleigh_A02_table!(hist; stride=stride, nel=NEL_RAY, N=N, k=k, T_period=T_period)
end

function main()
    mkpath(OUTDIR)
    run_leakage_test()
    # Rayleigh 2-period run is kept as-is; do not overwrite rayleigh_error_A02.*
    println("\nplot with:  python3 test/cutoff_tracking/plot.py")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
