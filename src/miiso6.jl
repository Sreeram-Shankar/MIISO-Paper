#MIISO6, Modal Integrator for Isogeometric Stabilization of Outliers (order 6).
#the construction and methodi is described in the paper. The pade maps
#are the [3,3] and [0,6] Pades. This method is repaired with the [2,4] Pade
#on the 0/6 instability islands (around the LHP poles of R06, z ~ -0.8036 +/- 3.6977i).
#the islands do not touch the imaginary axis so it is only in the left half

include(joinpath(@__DIR__, "utilities.jl"))

using LinearAlgebra

#Gauss/conservative 6th order 3/3 diagonal Pade (3-stage Gauss–Legendre)
function R_C(z)
    z = ComplexF64(z)
    return (1 + z/2 + z^2/10 + z^3/120) / (1 - z/2 + z^2/10 - z^3/120)
end

Q06(z) = 1 - z + z^2/2 - z^3/6 + z^4/24 - z^5/120 + z^6/720
Q05(z) = 1 - z + z^2/2 - z^3/6 + z^4/24 - z^5/120        # Q06'(z) = -Q05(z)

#highly dissipative 6th order 0/6 Pade
function R06(z)
    z = ComplexF64(z)
    return 1 / Q06(z)
end

#6th order 2/4 Pade (4-stage Lobatto IIIC, L-stable) used to repair the 0/6 instability
function R24(z)
    z = ComplexF64(z)
    return (1 + z/3 + z^2/30) / (1 - 2z/3 + z^2/5 - z^3/30 + z^4/360)
end

#this indicator is negative inside the 0/6 instability islands
F06(z) = abs(Q06(z))^2 - 1

#upper-half-plane LHP pole of R06, by Newton on Q06, then there is an island around it
function pole06(z0 = ComplexF64(-0.8, 3.7))
    z = z0
    for _ = 1:50
        dz = Q06(z) / (-Q05(z))
        z -= dz
        abs(dz) < 1e-15 && break
    end
    return z
end
const p06 = pole06()     # ≈ -0.8036 + 3.6977i

#distance from p06 to the 0/6 stability boundary along direction alpha in [0, 2π] via bisection
function ray_boundary(α; rmax = 1.0, n = 4001)
    d = cis(α)
    rs = range(0.0, rmax; length = n)
    Fs = [F06(p06 + r * d) for r in rs]
    i = findfirst(j -> Fs[j] < 0 && Fs[j+1] >= 0, 1:(n-1))
    i === nothing && return 0.0
    a, b = float(rs[i]), float(rs[i+1])
    for _ = 1:50
        m = 0.5 * (a + b)
        F06(p06 + m * d) < 0 ? (a = m) : (b = m)
    end
    return 0.5 * (a + b)
end

#save the boundary closed geometry once before running the code (full circle, endpoints coincide)
const angles = range(0.0, 2π; length = 2001)
const raymap = [ray_boundary(α) for α in angles]

#blend weight that switches from the 0/6 to the 2/4 Pade on the instability islands
function theta_wedge(z; pad = 0.10)
    z = ComplexF64(z)
    d = ComplexF64(real(z), abs(imag(z))) - p06
    r = abs(d)
    r < 1e-14 && return 1.0
    α = mod(angle(d), 2π)
    s = r / instability_interpolation(α, angles, raymap)
    s <= 1 && return 1.0
    s >= 1 + pad && return 0.0
    return (1 + pad - s) / pad
end

#dissipative branch is 0/6 Pade repaired by the 2/4 Pade where it is unstable
function R_D(z)
    z = ComplexF64(z)
    th = theta_wedge(z)
    th <= 0 && return R06(z)
    th >= 1 && return R24(z)
    return blend_RD(R06(z), R24(z), th)
end

#warm the small-|z| Taylor series of R_C used by phi_blend
rc_series(R_C)

#node values used by the stages
const C6 = (0.5, 0.25, 1/3, 1.0)


#arguments match miiso4_step (cutoff / operators / krylov / taper) for a miiso6 step, needs length(ws) >= 4 for parallel
function miiso6_step(
    f,
    t,
    y,
    h;
    setup = nothing,
    k = nothing,
    Omega_c = nothing,
    J = nothing,
    ft = nothing,
    krylov_dim_max::Int = 40,
    cutoff_adaptive = true,
    omega_frac = 0.99,
    taper_order = 4,
    ws = nothing,
    tau = 1e-10,
)
    y, fn, J, ft, oc, km =
        prepare(f, t, y, h; setup, k, Omega_c, J, ft, krylov_dim_max, cutoff_adaptive)
    o0 = float(omega_frac) * oc
    n = length(y)
    m = clamp(km, 1, n)
    (ws === nothing || length(ws) < 4) && (ws = ntuple(_ -> krylov_ws_new(n, m), 4))
    foreach(reset_phi_pool!, ws)
    base = tau * (1 + norm(y))
    thr = 1e-14 * (1 + norm(fn))
    ap(wsi, v, specs, tol) =
        apply_phi(wsi, J, v, h, o0, oc, km, R_C, R_D, specs, tol; p = taper_order)
    zero_n(wsi) = fill!(ensure!(wsi.z0, n), 0.0)

    #principal actions phi1 f, phi2 f_t at c = 1/2, 1/4, 1/3, 1 (one Arnoldi each)
    if norm(ft) > 1e-30
        p1, p2 = parallel_pair(
            () -> ap(ws[1], fn, [(1, c) for c in C6], base / h),
            () -> ap(ws[2], ft, [(2, c) for c in C6], base / h^2),
        )
    else
        p1 = ap(ws[1], fn, [(1, c) for c in C6], base / h)
        z = zero_n(ws[1])
        p2 = [z, z, z, z]
    end
    #P(c) for c = 1/2, 1/4, 1/3, 1, into the stage buffers
    P = (
        ensure!(ws[1].y2, n),
        ensure!(ws[2].y2, n),
        ensure!(ws[3].y2, n),
        ensure!(ws[4].y2, n),
    )
    for (i, c) in pairs(C6)
        @. P[i] = y + c * h * p1[i] + c^2 * h^2 * p2[i]
    end

    #stages 2 and 3 are the principal parts at c = 1/2 and 1/4; their defects in parallel
    D2 = ensure!(ws[1].d2, n)
    D3 = ensure!(ws[2].d2, n)
    parallel_pair(
        () -> defect!(D2, ws[1], f, t, 0.5, h, P[1], y, fn, ft, J),
        () -> defect!(D3, ws[2], f, t, 0.25, h, P[2], y, fn, ft, J),
    )

    #stage coefficients (a3, a4) on D2 and (b3, b4) on D3 for U4..U7
    A2 = ((-1.0, 6.0), (-1/8, 3/8), (-8/27, 32/27), (-8.0, 96.0))
    A3 = ((8.0, -24.0), (1.0, -3/2), (64/27, -128/27), (64.0, -384.0))
    #one Arnoldi per defect: combo (a3 phi3 + a4 phi4)(c hJ) D at each node 
    specs2 = [(((3, a3), (4, a4)), c) for ((a3, a4), c) in zip(A2, C6)]
    specs3 = [(((3, b3), (4, b4)), c) for ((b3, b4), c) in zip(A3, C6)]
    zeros4() = (z = zero_n(ws[3]); [z, z, z, z])
    q2, q3 = parallel_pair(
        () -> norm(D2) > thr ? ap(ws[1], D2, specs2, base / (384h)) : zeros4(),
        () -> norm(D3) > thr ? ap(ws[2], D3, specs3, base / (384h)) : zeros4(),
    )

    U = (
        ensure!(ws[1].y3, n),
        ensure!(ws[2].y3, n),
        ensure!(ws[3].y3, n),
        ensure!(ws[4].y3, n),
    )
    for i = 1:4
        @. U[i] = P[i] + h * (q2[i] + q3[i])
    end

    #stages 4..7 in parallel (nodes 1/2, 1/4, 1/3, 1)
    D = (
        ensure!(ws[1].d3, n),
        ensure!(ws[2].d3, n),
        ensure!(ws[3].d3, n),
        ensure!(ws[4].d3, n),
    )
    parallel_n(
        ntuple(i -> () -> defect!(D[i], ws[i], f, t, C6[i], h, U[i], y, fn, ft, J), 4)...,
    )
    D4, D5, D6, D7 = D

    #linear combos of D4..D7 become w3..w6 for phi3..phi6
    W = (
        ensure!(ws[1].u3, n),
        ensure!(ws[2].u3, n),
        ensure!(ws[3].u3, n),
        ensure!(ws[4].u3, n),
    )
    @. W[1] = 32 * D4 + (1024/3) * D5 - 243 * D6 - (1/3) * D7
    @. W[2] = -768 * D4 - 6144 * D5 + 5103 * D6 + 9 * D7
    @. W[3] = 7296 * D4 + 45056 * D5 - 40824 * D6 - 104 * D7
    @. W[4] = -23040 * D4 - 122880 * D5 + 116640 * D6 + 480 * D7
    phis = parallel_n(
        ntuple(
            i ->
                () -> begin
                    norm(W[i]) > thr ?
                    ap(ws[i], W[i], [(i + 2, 1.0)], base / h)[1] : zero_n(ws[i])
                end,
            4,
        )...,
    )

    @. y += h * p1[4] + h^2 * p2[4] + h * (phis[1] + phis[2] + phis[3] + phis[4])
    return y
end


#integrate y'=f from t0 to T. arguments match miiso4_solve.
function miiso6_solve(
    f,
    y0,
    T,
    h;
    t0 = 0.0,
    setup = nothing,
    k = nothing,
    Omega_c = nothing,
    J = nothing,
    ft = nothing,
    krylov_dim_max::Int = 40,
    cutoff_adaptive = true,
    omega_frac = 0.99,
    taper_order = 4,
    tau = 1e-10,
    callback = nothing,
)
    miiso_solve(
        miiso6_step,
        f,
        y0,
        T,
        h;
        t0,
        setup,
        k,
        Omega_c,
        J,
        ft,
        krylov_dim_max,
        cutoff_adaptive,
        omega_frac,
        taper_order,
        tau,
        n_ws = 4,
        callback,
    )
end
