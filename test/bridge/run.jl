# Continuum arch library: NURBS face sampling, frame listing, and |u|/ψ export.
# Included by full_run.jl, which drives the MIISO4 integration on LS6.

include(joinpath(@__DIR__, "setup.jl"))
using Serialization
using LinearAlgebra

# ── face sampling ─────────────────────────────────────────────────────────────
# fix ∈ (:η,:ζ,:ξ); gridA × gridB span the free parametric directions.
# Pass λ,μ to sample SVK energy density instead of |u|.
function face_sample(kv, orders, cps, ws, Ucp, gridA, gridB; fix=:η, val=0.0, λ=nothing, μ=nothing)
    nB, nA = length(gridB), length(gridA)
    X = zeros(nB, nA); Y = similar(X); Z = similar(X); C = similar(X)
    for (j, a) in enumerate(gridA), (i, b) in enumerate(gridB)
        ξ, η, ζ = fix === :η ? (a, val, b) : fix === :ζ ? (a, b, val) : (val, a, b)
        if λ === nothing
            (x, y, z), u = nurbs_point(kv, orders, cps, ws, ξ, η, ζ, Ucp)
            C[i, j] = hypot(u...)
        else
            (x, y, z), ∇u = nurbs_grad(kv, orders, cps, ws, ξ, η, ζ, Ucp)
            C[i, j] = svk_energy(∇u, λ, μ)
        end
        X[i, j] = x; Y[i, j] = y; Z[i, j] = z
    end
    return X, Y, Z, C
end

function nurbs_grad(kv, orders, cps::AbstractMatrix, ws, ξ, η, ζ, Ucp)
    n1, n2, n3 = length.(kv) .- orders .- 1
    Na, dNa = bsplines(kv[1], orders[1], ξ)
    Nr, dNr = bsplines(kv[2], orders[2], η)
    Nz, dNz = bsplines(kv[3], orders[3], ζ)
    W = dWξ = dWη = dWζ = 0.0
    Px = zeros(3); Pu = zeros(3)
    dPx = zeros(3, 3); dPu = zeros(3, 3)
    @inbounds for iz in 1:n3, ir in 1:n2, ia in 1:n1
        idx = ia + (ir - 1) * n1 + (iz - 1) * n1 * n2
        w = ws[idx]
        N0 = Na[ia] * Nr[ir] * Nz[iz]
        Nξ = dNa[ia] * Nr[ir] * Nz[iz]
        Nη = Na[ia] * dNr[ir] * Nz[iz]
        Nζ = Na[ia] * Nr[ir] * dNz[iz]
        wN = w * N0; wξ = w * Nξ; wη = w * Nη; wζ = w * Nζ
        W += wN; dWξ += wξ; dWη += wη; dWζ += wζ
        for c in 1:3
            pc = cps[c, idx]; uc = Ucp[c, ia, ir, iz]
            Px[c] += wN * pc; Pu[c] += wN * uc
            dPx[c, 1] += wξ * pc; dPx[c, 2] += wη * pc; dPx[c, 3] += wζ * pc
            dPu[c, 1] += wξ * uc; dPu[c, 2] += wη * uc; dPu[c, 3] += wζ * uc
        end
    end
    invW = 1 / W
    X = (Px[1] * invW, Px[2] * invW, Px[3] * invW)
    dW = (dWξ, dWη, dWζ)
    J = zeros(3, 3); Guξ = zeros(3, 3)
    for j in 1:3, i in 1:3
        J[i, j] = (dPx[i, j] * W - Px[i] * dW[j]) * invW^2
        Guξ[i, j] = (dPu[i, j] * W - Pu[i] * dW[j]) * invW^2
    end
    return X, Guξ / J
end

svk_energy(∇u, λ, μ) = (F = I + ∇u; E = 0.5 .* (transpose(F) * F - I);
                         0.5 * λ * (E[1,1] + E[2,2] + E[3,3])^2 + μ * sum(abs2, E))

frame_list(framedir) = sort(filter(f -> startswith(f, "u_") && endswith(f, ".jls"), readdir(framedir)))

# ── export |u| surfaces + wire cage ───────────────────────────────────────────
function export_displacement!(outdir)
    framedir = joinpath(outdir, get(ENV, "FRAMEDIR", "frames"))
    expdir = joinpath(outdir, "export")
    mkpath(expdir)
    meta = deserialize(joinpath(framedir, "meta.jls"))
    kv, orders, cps, ws = meta.kv, meta.orders, meta.cps, meta.ws
    ηt, ηb = meta.η_top, meta.η_bot
    ξs = collect(range(kv[1][1], kv[1][end]; length=72))
    ηs = collect(range(ηb, ηt; length=16))
    ζs = collect(range(kv[3][1], kv[3][end]; length=28))
    η_layers = collect(range(ηb, ηt; length=5))
    ζ_layers = collect(range(ζs[1], ζs[end]; length=5))
    files = frame_list(framedir)
    nF = length(files)
    println("exporting |u|  $nF frames  η=$(length(η_layers))  ζ=$(length(ζ_layers))"); flush(stdout)

    faces = NamedTuple{(:name, :samp), Tuple{String,Function}}[]
    for (k, η) in enumerate(η_layers)
        push!(faces, (name="h$k", samp=U -> face_sample(kv, orders, cps, ws, U, ξs, ζs; fix=:η, val=η)))
    end
    for (k, ζ) in enumerate(ζ_layers)
        push!(faces, (name="w$k", samp=U -> face_sample(kv, orders, cps, ws, U, ξs, ηs; fix=:ζ, val=ζ)))
    end
    push!(faces, (name="e0", samp=U -> face_sample(kv, orders, cps, ws, U, ηs, ζs; fix=:ξ, val=ξs[1])))
    push!(faces, (name="e1", samp=U -> face_sample(kv, orders, cps, ws, U, ηs, ζs; fix=:ξ, val=ξs[end])))

    fr0 = deserialize(joinpath(framedir, files[1])).Ucp
    Cstack = Dict{String,Array{Float64,3}}()
    names = String[]
    for face in faces
        X, Y, Z, _ = face.samp(fr0)
        write_f64(joinpath(expdir, "X_$(face.name).bin"), X)
        write_f64(joinpath(expdir, "Y_$(face.name).bin"), Y)
        write_f64(joinpath(expdir, "Z_$(face.name).bin"), Z)
        Cstack[face.name] = zeros(nF, size(X, 1), size(X, 2))
        push!(names, face.name)
    end
    times = zeros(nF); cmax = 0.0
    for (k, f) in enumerate(files)
        fr = deserialize(joinpath(framedir, f))
        times[k] = fr.t
        for face in faces
            _, _, _, C = face.samp(fr.Ucp)
            Cstack[face.name][k, :, :] .= C
            cmax = max(cmax, maximum(C))
        end
        k % 40 == 0 && (println("  |u| $k/$nF"); flush(stdout))
    end
    write_f64(joinpath(expdir, "times.bin"), times)
    for name in names
        write_f64(joinpath(expdir, "C_$name.bin"), Cstack[name])
    end
    write_f64(joinpath(expdir, "xis.bin"), ξs)
    write_f64(joinpath(expdir, "etas.bin"), ηs)
    write_f64(joinpath(expdir, "zetas.bin"), ζs)

    ξu, ηu, ζu = unique(Float64.(kv[1])), unique(Float64.(kv[2])), unique(Float64.(kv[3]))
    npts_w = 32
    U0 = zeros(3, meta.n1, meta.n2, meta.n3)
    wlines = Matrix{Float64}[]
    function add_wire!(ξv, ηv, ζv)
        P = zeros(3, length(ξv))
        for i in eachindex(ξv)
            (x, y, z), _ = nurbs_point(kv, orders, cps, ws, ξv[i], ηv[i], ζv[i], U0)
            P[1, i] = x; P[2, i] = y; P[3, i] = z
        end
        push!(wlines, P)
    end
    for η in ηu, ζ in ζu
        v = collect(range(ξu[1], ξu[end]; length=npts_w))
        add_wire!(v, fill(η, npts_w), fill(ζ, npts_w))
    end
    for ξ in ξu, ζ in ζu
        v = collect(range(ηu[1], ηu[end]; length=npts_w))
        add_wire!(fill(ξ, npts_w), v, fill(ζ, npts_w))
    end
    for ξ in ξu, η in ηu
        v = collect(range(ζu[1], ζu[end]; length=npts_w))
        add_wire!(fill(ξ, npts_w), fill(η, npts_w), v)
    end
    nw = length(wlines)
    W = zeros(nw, 3, npts_w)
    for i in 1:nw; W[i, :, :] .= wlines[i]; end
    write_f64(joinpath(expdir, "wires.bin"), W)
    open(joinpath(expdir, "info.txt"), "w") do io
        println(io, "nF=$nF"); println(io, "cmax=$cmax"); println(io, "T=$(meta.T)")
        println(io, "dt_save=$(meta.dt_save)"); println(io, "nel=$(meta.nel)")
        println(io, "faces=" * join(names, ","))
        println(io, "nwires=$nw"); println(io, "wire_npts=$npts_w")
    end
    println("done |u|  cmax=$cmax  faces=$(join(names, ","))  wires=$nw")
end

# ── export SVK energy density ─────────────────────────────────────────────────
function export_energy!(outdir; Emat=34e9, ν=0.2)
    framedir = joinpath(outdir, get(ENV, "FRAMEDIR", "frames"))
    expdir = joinpath(outdir, "energy")
    mkpath(expdir)
    λ = Emat * ν / ((1 + ν) * (1 - 2ν))
    μ = Emat / (2(1 + ν))
    meta = deserialize(joinpath(framedir, "meta.jls"))
    kv, orders, cps, ws = meta.kv, meta.orders, meta.cps, meta.ws
    ηt, ηb = meta.η_top, meta.η_bot
    ξs = collect(range(kv[1][1], kv[1][end]; length=48))
    ηs = collect(range(ηb, ηt; length=10))
    ζs = collect(range(kv[3][1], kv[3][end]; length=16))
    files = frame_list(framedir)
    nF = length(files)
    println("exporting ψ  $nF frames → $expdir"); flush(stdout)
    facespec = (
        ("top", :η, ηt, ξs, ζs), ("bot", :η, ηb, ξs, ζs),
        ("s0", :ζ, ζs[1], ξs, ηs), ("s1", :ζ, ζs[end], ξs, ηs),
        ("e0", :ξ, ξs[1], ηs, ζs), ("e1", :ξ, ξs[end], ηs, ζs),
    )
    fr0 = deserialize(joinpath(framedir, files[1])).Ucp
    Cstack = Dict{String,Array{Float64,3}}()
    for (name, fix, val, gA, gB) in facespec
        X, Y, Z, _ = face_sample(kv, orders, cps, ws, fr0, gA, gB; fix, val, λ, μ)
        write_f64(joinpath(expdir, "X_$name.bin"), X)
        write_f64(joinpath(expdir, "Y_$name.bin"), Y)
        write_f64(joinpath(expdir, "Z_$name.bin"), Z)
        Cstack[name] = zeros(nF, size(X, 1), size(X, 2))
    end
    times = zeros(nF); ψmax = 0.0
    for (k, f) in enumerate(files)
        fr = deserialize(joinpath(framedir, f))
        times[k] = fr.t
        for (name, fix, val, gA, gB) in facespec
            _, _, _, Ψ = face_sample(kv, orders, cps, ws, fr.Ucp, gA, gB; fix, val, λ, μ)
            Cstack[name][k, :, :] .= Ψ
            ψmax = max(ψmax, maximum(Ψ))
        end
        k % 20 == 0 && (println("  ψ $k/$nF  max=$(round(ψmax; sigdigits=4))"); flush(stdout))
    end
    write_f64(joinpath(expdir, "times.bin"), times)
    for (name, _, _, _, _) in facespec
        write_f64(joinpath(expdir, "P_$name.bin"), Cstack[name])
    end
    wsrc = joinpath(outdir, "export", "wires.bin")
    isfile(wsrc) && cp(wsrc, joinpath(expdir, "wires.bin"); force=true)
    open(joinpath(expdir, "info.txt"), "w") do io
        println(io, "nF=$nF"); println(io, "psi_max=$ψmax"); println(io, "T=$(meta.T)")
        println(io, "E=$Emat"); println(io, "nu=$ν"); println(io, "nel=$(meta.nel)")
        println(io, "faces=top,bot,s0,s1,e0,e1")
    end
    println("done ψ  nF=$nF  ψ_max=$ψmax")
end
