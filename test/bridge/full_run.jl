# Continue MIISO4 arch from a saved state through T=11.5, with frames + probes.
# Appends into an existing OUTDIR when resuming. On errors / Slurm wall-time /
# SIGTERM, dumps the last good state.
#
#   julia --project=. -t 32 --startup-file=no test/bridge/full_run.jl
#
# ENV: STATE  OUTDIR  T  DT_SAVE  KM  WALL_MARGIN_SEC

include(joinpath(@__DIR__, "run.jl"))
using Serialization
using Printf

struct WallTimeUp <: Exception end

const STATE = get(ENV, "STATE", joinpath(@__DIR__, "out_bridge_nel3266", "state_timeout.jls"))
const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "out_bridge_nel3266"))
const T = parse(Float64, get(ENV, "T", "11.5"))
const DT_SAVE = parse(Float64, get(ENV, "DT_SAVE", "0.05"))
const KM_ENV = parse(Int, get(ENV, "KM", "40"))
const WALL_MARGIN_SEC = parse(Float64, get(ENV, "WALL_MARGIN_SEC", "300"))

const DEADLINE = if haskey(ENV, "SLURM_JOB_END_TIME")
    parse(Float64, ENV["SLURM_JOB_END_TIME"]) - WALL_MARGIN_SEC
else
    Inf
end

const WANT_STOP = Ref(false)
const STATE_SAVED = Ref(false)

function _sigterm(_signo::Cint)::Cvoid
    WANT_STOP[] = true
    return nothing
end
try
    ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), 15,
          @cfunction(_sigterm, Cvoid, (Cint,)))
catch
end

st = deserialize(STATE)
h = Float64(st.h)
t0 = Float64(st.t)
nel = Tuple(Int.(st.nel))
y0 = Vector{Float64}(st.y)
km = hasproperty(st, :km) ? Int(st.km) : KM_ENV
Omega_c0 = hasproperty(st, :Omega_c) ? Float64(st.Omega_c) : h * Float64(st.omega_c)

println("full_run continue  threads=$(Threads.nthreads())  nel=$nel")
println("  STATE=$STATE")
println("  t0=$t0  T=$T  h=$h  dt_save=$DT_SAVE  km=$km  Omega_c=$Omega_c0")
println("  OUTDIR=$OUTDIR")
if isfinite(DEADLINE)
    println("  Slurm soft deadline in $(round(DEADLINE - time(); digits=0))s ",
            "(margin $(WALL_MARGIN_SEC)s)")
end
flush(stdout)

framedir = joinpath(OUTDIR, "frames")
mkpath(framedir)

println("building …"); flush(stdout)
sys = arch_bridge_system(BridgeParams(nel=nel))
mesh, free, n1, n2, n3 = sys.mesh, sys.free, sys.n1, sys.n2, sys.n3
nf = sys.nfree
@assert nf == st.nf "nf mismatch: sys=$nf state=$(st.nf)"
nsteps = round(Int, (T - t0) / h)
println("  nf=$nf  ωc=$(round(sys.omega_c; sigdigits=5))  remaining steps≈$nsteps"); flush(stdout)

cps = cps_matrix(mesh.control_points)
ws = Vector{Float64}(mesh.weights)
kv = map(v -> Vector{Float64}(v), mesh.knot_vectors)
orders = Int.(mesh.orders)
η_top, η_bot = kv[2][end], kv[2][1]
ζ_mid = 0.5 * (kv[3][1] + kv[3][end])
ξs = collect(range(kv[1][1], kv[1][end]; length=64))
ζs = collect(range(kv[3][1], kv[3][end]; length=24))

serialize(joinpath(framedir, "meta.jls"),
          pack_meta(; T, h, dt_save=DT_SAVE, n1, n2, n3, nf, free, omega_c=sys.omega_c,
                    ξs, ζs, η_top, η_bot, kv, orders, cps_mat=cps, ws,
                    nel, t_ramp=sys.P.t_ramp, t_eq=sys.P.t_eq,
                    jac_rtol=sys.P.jac_rtol, jac_max_lag=sys.P.jac_max_lag, krylov_dim_max=km))

# Resume: append probes / frames. Frame IDs come from existing u_*.jls files
# (not the index last line) so a corrupt index cannot overwrite early frames.
probe_path = joinpath(OUTDIR, "probes.csv")
index_path = joinpath(framedir, "index.csv")
existing = Int[]
for f in readdir(framedir)
    m = match(r"^u_(\d+)\.jls$", f)
    m === nothing && continue
    push!(existing, parse(Int, m.captures[1]))
end
resume = !isempty(existing)
iframe0 = resume ? maximum(existing) + 1 : 0
next_t = Float64(t0)
if resume
    # next save just after the latest frame time on disk
    tmax = -Inf
    for i in existing
        fr = deserialize(joinpath(framedir, "u_" * lpad(string(i), 4, '0') * ".jls"))
        tmax = max(tmax, Float64(fr.t))
    end
    next_t = tmax + DT_SAVE
    println("  resume append  next_frame=$iframe0  next_save_t=$(round(next_t; digits=4))  ",
            "existing_frames=$(length(existing))")
    flush(stdout)
end

new_probes = !isfile(probe_path) || filesize(probe_path) == 0
new_index = !isfile(index_path) || filesize(index_path) == 0
probe_io = open(probe_path, new_probes ? "w" : "a")
index_io = open(index_path, new_index ? "w" : "a")
new_probes && println(probe_io, "t,u_crown,u_end")
new_index && println(index_io, "iframe,t,file")

Ucp, Uflat = make_U_workspace(n1, n2, n3)
next_save = Ref(next_t)
iframe = Ref(iframe0)
tprog = Ref(time())
tw0 = time()
last_t = Ref(t0)
last_y = Ref(copy(y0))

function save_state!(path, t, y; extra=NamedTuple())
    fill!(Uflat, 0.0)
    @inbounds Uflat[free] .= view(y, 1:nf)
    serialize(path, (; t=Float64(t), h, y=copy(y), Ucp=copy(Ucp), free=copy(free),
                     n1, n2, n3, nf, omega_c=sys.omega_c, Omega_c=Omega_c0,
                     nrebuild=sys.nrebuild[], nel, km, extra...))
    STATE_SAVED[] = true
    return path
end

function cb(t, y)
    fill!(Uflat, 0.0)
    @inbounds Uflat[free] .= view(y, 1:nf)
    _, uc = nurbs_point(kv, orders, cps, ws, 0.5, η_top, ζ_mid, Ucp)
    _, ue0 = nurbs_point(kv, orders, cps, ws, 0.0, η_top, ζ_mid, Ucp)
    _, ue1 = nurbs_point(kv, orders, cps, ws, 1.0, η_top, ζ_mid, Ucp)
    ucr = hypot(uc...)
    uen = max(hypot(ue0...), hypot(ue1...))
    println(probe_io, t, ',', ucr, ',', uen)

    last_t[] = t
    last_y[] = copy(y)

    if t + 0.5h >= next_save[] || t >= T - 0.5h
        fname = "u_$(lpad(iframe[], 4, '0')).jls"
        serialize(joinpath(framedir, fname), (; t=Float64(t), Ucp=copy(Ucp)))
        println(index_io, iframe[], ',', t, ',', fname)
        flush(probe_io); flush(index_io)
        iframe[] += 1
        next_save[] += DT_SAVE
    end

    now = time()
    if now - tprog[] >= 30 || t >= T - h
        tprog[] = now
        done = max(t - t0, 0.0)
        frac = done / max(T - t0, eps())
        eta = frac > 1e-9 ? (now - tw0) * (1 - frac) / frac : NaN
        istep = round(Int, done / h)
        println("  step $istep/$nsteps  t=$(round(t; digits=3))/$(T)  ",
                "$(round(100frac; digits=1))%  |u|_crown=$(round(ucr; sigdigits=3))  ",
                "frames=$(iframe[])  ETA=$(round(eta / 60; digits=1)) min")
        flush(stdout)
    end

    if WANT_STOP[] || now >= DEADLINE
        throw(WallTimeUp())
    end
    return nothing
end

atexit() do
    STATE_SAVED[] && return
    try
        path = save_state!(joinpath(OUTDIR, "state_atexit.jls"), last_t[], last_y[];
                           extra=(; err="atexit"))
        println("atexit — saved state → $path"); flush(stdout)
    catch
    end
end

println("MIISO4 integrating from t=$t0 …"); flush(stdout)
status = "ok"
try
    yT = miiso4_solve(sys.f, y0, T, h; t0=t0, ft=sys.ft, J=sys.J,
                      Omega_c=Omega_c0, cutoff_adaptive=true,
                      krylov_dim_max=km, callback=cb)
    save_state!(joinpath(OUTDIR, "state_final.jls"), T, yT)
    serialize(joinpath(OUTDIR, "yT.jls"), Vector{Float64}(yT))
catch e
    close(probe_io); close(index_io)
    if e isa WallTimeUp
        path = save_state!(joinpath(OUTDIR, "state_timeout.jls"), last_t[], last_y[];
                           extra=(; err="walltime_or_sigterm"))
        println("WALL TIME / SIGTERM — saved state → $path  t=$(last_t[])")
        flush(stdout)
        exit(0)
    end
    err_msg = sprint(showerror, e)
    path = save_state!(joinpath(OUTDIR, "state_error.jls"), last_t[], last_y[];
                       extra=(; err=err_msg))
    println("ERROR — saved state → $path")
    println(err_msg)
    flush(stdout)
    rethrow()
end
close(probe_io); close(index_io)

wall = time() - tw0
@printf("done status=%s  wall=%.1fs  nrebuild=%d  nframes=%d\n",
        status, wall, sys.nrebuild[], iframe[])
println("  state_final → $(joinpath(OUTDIR, "state_final.jls"))")

export_displacement!(OUTDIR)
export_energy!(OUTDIR)
println("all exports ready → $OUTDIR")
