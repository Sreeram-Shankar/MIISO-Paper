#MIISO4, Modal Integrator for Isogeometric Stabilization of Outliers (order 4).
#the construction and methodi is described in the paper. The pade maps
#are the [2,2] and [0,4] Pades. This method is rparied with the [1,3] Pade
#inside its instability wedge (|R04(iw)| > 1 on 0 < |w| < sqrt(8)).

include(joinpath(@__DIR__, "utilities.jl"))

using LinearAlgebra

#Gauss/conservative 4th order 2/2 diagonal Pade
function R_C(z)
    z = ComplexF64(z)
    return (1 + z/2 + z^2/12) / (1 - z/2 + z^2/12)
end

#highly dissipative 4th order 0/4 diagonal Pade (not A-stable)
function R04(z)
    z = ComplexF64(z)
    return 1 / (1 - z + z^2/2 - z^3/6 + z^4/24)
end

#fairly dissipative 4th order 1/3 diagonal Pade (A-stable) used to stitch the instability of the 0/4 Pade
function R13(z)
    z = ComplexF64(z)
    return (1 + 0.25 * z) / (1 - 0.75 * z + 0.25 * z^2 - z^3/24)
end

#stability boundary for the highly dissipative 4th order 0/4 diagonal Pade
function F04(x, y)
    z = float(x) + im * float(y)
    return abs(1 - z + z^2/2 - z^3/6 + z^4/24)^2 - 1
end

#finds where the 0/4 Pade crosses back into stability (|R04|=1) at height y, via bisection
function instability_boundary(y; xlo = -0.5, n = 2001)
    y, ay = float(y), abs(float(y))
    (ay <= 0 || ay >= sqrt(8)) && return 0.0
    xs = range(xlo, 0.0; length = n)
    Fs = F04.(xs, y)
    hits = findall(i -> signbit(Fs[i]) != signbit(Fs[i+1]), 1:(n-1))
    isempty(hits) && return 0.0
    i = hits[end]
    a, b = float(xs[i]), float(xs[i+1])
    for _ = 1:50
        m = 0.5 * (a + b)
        a, b = float(F04(a, y)) * float(F04(m, y)) <= 0 ? (a, m) : (m, b)
    end
    return 0.5 * (a + b)
end

#save the boundary for the instability repair once before running the code
const spaces = range(0.0, sqrt(8); length = 1001)
const boundarymap = [instability_boundary(y) for y in spaces]

#blend weight that switches from the 0/4 to the 1/3 Pade inside the 0/4 instability wedge
function theta_wedge(z; pad = 0.10)
    z = ComplexF64(z)
    x, y, ay = real(z), imag(z), abs(imag(z))
    y_max = (1 + pad) * sqrt(8)
    ay >= y_max && return 0.0
    if ay < sqrt(8)
        xb = instability_interpolation(ay, spaces, boundarymap)
        xb >= -1e-14 && return (x >= -1e-14 && ay > 0) ? 1.0 : 0.0
        x_outer = (1 + pad) * xb
        x <= x_outer && return 0.0
        x >= xb && return 1.0
        return (x - x_outer) / (xb - x_outer)
    end
    s = (y_max - ay) / (pad * sqrt(8))
    x_outer = -pad * 0.25 * s
    x <= x_outer && return 0.0
    x >= 0 && return s
    return s * (x - x_outer) / (0 - x_outer)
end

#dissipative branch is 0/4 Pade repaired by the 1/3 Pade where it is unstable
function R_D(z)
    z = ComplexF64(z)
    th = theta_wedge(z)
    th <= 0 && return R04(z)
    th >= 1 && return R13(z)
    return blend_RD(R04(z), R13(z), th)
end

#warm the small-|z| Taylor series of R_C used by phi_blend
rc_series(R_C)


#arguments match miiso2_step (cutoff / operators / krylov / taper) for a miiso4 step
function miiso4_step(
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
    ws === nothing &&
        (ws = (krylov_ws_new(n, clamp(km, 1, n)), krylov_ws_new(n, clamp(km, 1, n))))
    foreach(reset_phi_pool!, ws)
    base = tau * (1 + norm(y))
    ap(wsi, v, specs, tol) =
        apply_phi(wsi, J, v, h, o0, oc, km, R_C, R_D, specs, tol; p = taper_order)
    if norm(ft) > 1e-30
        (phi1_f, phi1_f_half), (phi2_ft, phi2_ft_half) = parallel_pair(
            () -> ap(ws[1], fn, [(1, 1.0), (1, 0.5)], base / h),
            () -> ap(ws[2], ft, [(2, 1.0), (2, 0.5)], base / h^2),
        )
    else
        phi1_f, phi1_f_half = ap(ws[1], fn, [(1, 1.0), (1, 0.5)], base / h)
        phi2_ft = phi2_ft_half = fill!(ensure!(ws[1].z0, n), 0.0)
    end

    Y2 = ensure!(ws[1].y2, n)
    Y3 = ensure!(ws[1].y3, n)
    @. Y2 = y + 0.5 * h * phi1_f_half + 0.25 * h^2 * phi2_ft_half
    @. Y3 = y + h * phi1_f + h^2 * phi2_ft
    D2 = ensure!(ws[1].d2, n)
    D3 = ensure!(ws[1].d3, n)
    #sequential defects: f may use @threads :static (cannot nest under parallel_pair)
    defect!(D2, ws[1], f, t, 0.5, h, Y2, y, fn, ft, J)
    defect!(D3, ws[2], f, t, 1.0, h, Y3, y, fn, ft, J)
    #(16 phi3 - 48 phi4) D2 + (-2 phi3 + 12 phi4) D3 = phi3(w3) + phi4(w4), and w4 = O(h^3).
    w3 = ensure!(ws[1].u3, n)
    w4 = ensure!(ws[1].u4, n)
    @. w3 = 16 * D2 - 2 * D3
    @. w4 = 12 * (D3 - 4 * D2)
    thr = 1e-14 * (1 + norm(fn))
    big3, big4 = norm(w3) > thr, norm(w4) > thr
    if big3 && big4
        phi3_w3, phi4_w4 = parallel_pair(
            () -> ap(ws[1], w3, [(3, 1.0)], base / h)[1],
            () -> ap(ws[2], w4, [(4, 1.0)], base / h)[1],
        )
    elseif big3
        phi3_w3 = ap(ws[1], w3, [(3, 1.0)], base / h)[1]
        phi4_w4 = fill!(ensure!(ws[1].z0, n), 0.0)
    elseif big4
        phi3_w3 = fill!(ensure!(ws[2].z0, n), 0.0)
        phi4_w4 = ap(ws[1], w4, [(4, 1.0)], base / h)[1]
    else
        phi3_w3 = phi4_w4 = fill!(ensure!(ws[1].z0, n), 0.0)
    end
    @. y += h * phi1_f + h^2 * phi2_ft + h * (phi3_w3 + phi4_w4)
    return y
end


#integrate y'=f from t0 to T. arguments match miiso2_solve.
function miiso4_solve(
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
        miiso4_step,
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
        callback,
    )
end
