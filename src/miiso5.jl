#MIISO5, Modal Integrator for Isogeometric Stabilization of Outliers (order 5).
#the construction and methodi is described in the paper. The pade maps
#are the [3,3] and [0,5] Pades. This method is rparied with the [2,3] Pade
#on the 0/5 instability region (a band on the imaginary axis, 1.862 < |w| < 3.396 approximately).

include(joinpath(@__DIR__, "utilities.jl"))

using LinearAlgebra

#Gauss/conservative 6th order 3/3 diagonal Pade (3-stage Gauss–Legendre)
function R_C(z)
    z = ComplexF64(z)
    return (1 + z/2 + z^2/10 + z^3/120) / (1 - z/2 + z^2/10 - z^3/120)
end

#highly dissipative 5th order 0/5 Pade (not A-stable)
function R05(z)
    z = ComplexF64(z)
    return 1 / (1 - z + z^2/2 - z^3/6 + z^4/24 - z^5/120)
end

#5th order 2/3 Pade (3-stage Radau IIA, L-stable) used to repair the 0/5 instability
function R23(z)
    z = ComplexF64(z)
    return (1 + 2z/5 + z^2/20) / (1 - 3z/5 + 3z^2/20 - z^3/60)
end

#negative inside the 0/5 instability region
F05(z) = abs(1 - z + z^2/2 - z^3/6 + z^4/24 - z^5/120)^2 - 1

#imaginary-axis crossings of the 0/5 instability band, roots of w^4 - 15w^2 + 40
const y_lo = sqrt((15 - sqrt(65)) / 2)   # ≈ 1.8625
const y_hi = sqrt((15 + sqrt(65)) / 2)   # ≈ 3.3958
const y_c = (y_lo + y_hi) / 2           # the region is star-shaped about i*y_c
const c05 = ComplexF64(0, y_c)

#distance from c05 to the 0/5 stability boundary along direction alpha in [π/2, 3π/2] via bisection
function ray_boundary(α; rmax = 1.5, n = 3001)
    d = cis(α)
    rs = range(0.0, rmax; length = n)
    Fs = [F05(c05 + r * d) for r in rs]
    i = findfirst(j -> Fs[j] < 0 && Fs[j+1] >= 0, 1:(n-1))
    i === nothing && return 0.0
    a, b = float(rs[i]), float(rs[i+1])
    for _ = 1:50
        m = 0.5 * (a + b)
        F05(c05 + m * d) < 0 ? (a = m) : (b = m)
    end
    return 0.5 * (a + b)
end

#save the boundary closed geometry once before running the code
const angles = range(π/2, 3π/2; length = 2001)
const raymap = [ray_boundary(α) for α in angles]

#the blend weight goes from0 to 1 to go from 0/5 to 2/3 on the instability region
#since it does not touch the imaginary axis, we mirror in the y-direction
#and add a 10% pad about i*y_c. it is also a radial blend
function theta_wedge(z; pad = 0.10)
    z = ComplexF64(z)
    x, ay = min(real(z), 0.0), abs(imag(z))
    d = ComplexF64(x, ay) - c05
    r = abs(d)
    r < 1e-14 && return 1.0
    α = angle(d)
    α < 0 && (α += 2π)
    α = clamp(α, π/2, 3π/2)
    s = r / instability_interpolation(α, angles, raymap)
    s <= 1 && return 1.0
    s >= 1 + pad && return 0.0
    return (1 + pad - s) / pad
end

function R_D(z)
    z = ComplexF64(z)
    th = theta_wedge(z)
    th <= 0 && return R05(z)
    th >= 1 && return R23(z)
    return blend_RD(R05(z), R23(z), th)
end

#warm the small-|z| Taylor series of R_C used by phi_blend
rc_series(R_C)


#one pexprb54s5 + nonautonomous MIISO5 step. needs length(ws) >= 3
function miiso5_step(
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
    (ws === nothing || length(ws) < 3) && (ws = ntuple(_ -> krylov_ws_new(n, m), 3))
    foreach(reset_phi_pool!, ws)
    base = tau * (1 + norm(y))
    thr = 1e-14 * (1 + norm(fn))
    ap(wsi, v, specs, tol) =
        apply_phi(wsi, J, v, h, o0, oc, km, R_C, R_D, specs, tol; p = taper_order)
    zero_n(wsi) = fill!(ensure!(wsi.z0, n), 0.0)

    #phi1, phi2 at c in {1, 1/2, 1/3}
    if norm(ft) > 1e-30
        (p1, p1h, p1t), (p2, p2h, p2t) = parallel_pair(
            () -> ap(ws[1], fn, [(1, 1.0), (1, 0.5), (1, 1/3)], base / h),
            () -> ap(ws[2], ft, [(2, 1.0), (2, 0.5), (2, 1/3)], base / h^2),
        )
    else
        p1, p1h, p1t = ap(ws[1], fn, [(1, 1.0), (1, 0.5), (1, 1/3)], base / h)
        p2 = p2h = p2t = zero_n(ws[1])
    end

    U2 = ensure!(ws[1].y2, n)
    @. U2 = y + 0.5 * h * p1h + 0.25 * h^2 * p2h
    D2 = ensure!(ws[1].d2, n)
    defect!(D2, ws[1], f, t, 0.5, h, U2, y, fn, ft, J)

    #phi3(c hJ) D2 at c in {1/2, 1/3, 1}
    if norm(D2) > thr
        p3h, p3t, p3 = ap(ws[1], D2, [(3, 0.5), (3, 1/3), (3, 1.0)], base / (8h))
    else
        p3h = p3t = p3 = zero_n(ws[2])
    end

    U3, U4, U5 = ensure!(ws[1].y3, n), ensure!(ws[2].y3, n), ensure!(ws[3].y3, n)
    @. U3 = U2 + 0.5 * h * p3h
    @. U4 = y + (h / 3) * p1t + (h^2 / 9) * p2t + (8h / 27) * p3t
    @. U5 = y + h * p1 + h^2 * p2 + 8h * p3
    D3, D4, D5 = ensure!(ws[1].d3, n), ensure!(ws[2].d3, n), ensure!(ws[3].d3, n)
    parallel_n(
        () -> defect!(D3, ws[1], f, t, 0.5, h, U3, y, fn, ft, J),
        () -> defect!(D4, ws[2], f, t, 1/3, h, U4, y, fn, ft, J),
        () -> defect!(D5, ws[3], f, t, 1.0, h, U5, y, fn, ft, J),
    )

    w3, w4, w5 = ensure!(ws[1].u3, n), ensure!(ws[2].u3, n), ensure!(ws[3].u3, n)
    @. w3 = -32 * D3 + 81 * D4 + D5
    @. w4 = 384 * D3 - 729 * D4 - 15 * D5
    @. w5 = -1152 * D3 + 1944 * D4 + 72 * D5
    φ3, φ4, φ5 = parallel_n(
        () -> norm(w3) > thr ? ap(ws[1], w3, [(3, 1.0)], base / h)[1] : zero_n(ws[1]),
        () -> norm(w4) > thr ? ap(ws[2], w4, [(4, 1.0)], base / h)[1] : zero_n(ws[2]),
        () -> norm(w5) > thr ? ap(ws[3], w5, [(5, 1.0)], base / h)[1] : zero_n(ws[3]),
    )
    @. y += h * p1 + h^2 * p2 + h * (φ3 + φ4 + φ5)
    return y
end


#integrate y'=f from t0 to T.
function miiso5_solve(
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
        miiso5_step,
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
        n_ws = 3,
        callback,
    )
end
