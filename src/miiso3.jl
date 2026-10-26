#MIISO3, Modal Integrator for Isogeometric Stabilization of Outliers (order 3).
#the construction and methodi is described in the paper. The pade maps
#are the [2,2] and [0,3] Pades. This method is rparied with the [1,2] Pade
#inside its instability wedge (|R03(iw)| > 1 on 0 < |w| < sqrt(3)).

include(joinpath(@__DIR__, "utilities.jl"))

using LinearAlgebra

#Gauss/conservative 4th order 2/2 diagonal Pade for the physical modes
function R_C(z)
    z = ComplexF64(z)
    return (1 + z/2 + z^2/12) / (1 - z/2 + z^2/12)
end

#highly dissipative 3rd order 0/3 Pade (not A-stable)
function R03(z)
    z = ComplexF64(z)
    return 1 / (1 - z + z^2/2 - z^3/6)
end

#fairly dissipative 3rd order 1/2 Pade (A-stable) used to stitch the instability of the 0/3 Pade
function R12(z)
    z = ComplexF64(z)
    return (1 + z/3) / (1 - 2z/3 + z^2/6)
end

#stability boundary for the 0/3 Pade (|R03|^2 - 1 at z = x + iy)
function F03(x, y)
    re = 1 - x + 0.5 * (x * x - y * y) - (x^3 - 3 * x * y * y) / 6
    im = -y + x * y - (3 * x * x * y - y^3) / 6
    return re * re + im * im - 1
end

#finds where the 0/3 Pade crosses back into stability via bisection
function instability_boundary(y; xlo = -0.2, n = 2001)
    y, ay = float(y), abs(float(y))
    (ay <= 0 || ay >= sqrt(3)) && return 0.0
    xs = range(xlo, 0.0; length = n)
    Fs = F03.(xs, y)
    hits = findall(i -> signbit(Fs[i]) != signbit(Fs[i+1]), 1:(n-1))
    isempty(hits) && return 0.0
    i = hits[end]
    a, b = float(xs[i]), float(xs[i+1])
    for _ = 1:50
        m = 0.5 * (a + b)
        a, b = F03(a, y) * F03(m, y) <= 0 ? (a, m) : (m, b)
    end
    return 0.5 * (a + b)
end

#tabulate the boundary once before running the code
const spaces = range(0.0, sqrt(3); length = 1001)
const boundarymap = [instability_boundary(y) for y in spaces]

#blend weight 0->1 that switches from the 0/3 to the 1/2 Pade inside the 0/3 instability wedge
function theta_wedge(z; pad = 0.10)
    z = ComplexF64(z)
    x, y, ay = real(z), imag(z), abs(imag(z))
    y_max = (1 + pad) * sqrt(3)
    ay >= y_max && return 0.0
    if ay < sqrt(3)
        xb = instability_interpolation(ay, spaces, boundarymap)
        xb >= -1e-14 && return (x >= -1e-14 && ay > 0) ? 1.0 : 0.0
        x_outer = (1 + pad) * xb
        x <= x_outer && return 0.0
        x >= xb && return 1.0
        return (x - x_outer) / (xb - x_outer)
    end
    s = (y_max - ay) / (pad * sqrt(3))
    x_outer = -pad * 0.04 * s
    x <= x_outer && return 0.0
    x >= 0 && return s
    return s * (x - x_outer) / (0 - x_outer)
end

#the 0/3 Pade is repaired by the 1/2 Pade where it is unstable
function R_D(z)
    z = ComplexF64(z)
    th = theta_wedge(z)
    th <= 0 && return R03(z)
    th >= 1 && return R12(z)
    return blend_RD(R03(z), R12(z), th)
end

#warm the small-|z| Taylor series of R_C used by phi_blend
rc_series(R_C)



#arguments match miiso2_step (cutoff / operators / krylov / taper) for a miiso3 step
function miiso3_step(
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
        v1, v2 = parallel_pair(
            () -> ap(ws[1], fn, [(1, 1.0)], base / h)[1],
            () -> ap(ws[2], ft, [(2, 1.0)], base / h^2)[1],
        )
    else
        v1 = ap(ws[1], fn, [(1, 1.0)], base / h)[1]
        v2 = fill!(ensure!(ws[2].z0, n), 0.0)
    end
    Y2 = ensure!(ws[1].y2, n)
    @. Y2 = y + h * v1 + h^2 * v2
    D2 = ensure!(ws[1].d2, n)
    defect!(D2, ws[1], f, t, 1.0, h, Y2, y, fn, ft, J)
    v3 = if norm(D2) > 1e-14 * (1 + norm(fn))
        ap(ws[1], D2, [(3, 1.0)], base / (2 * h))[1]
    else
        fill!(ensure!(ws[1].z0, n), 0.0)
    end
    @. y += h * v1 + h^2 * v2 + 2 * h * v3
    return y
end


#integrate y'=f from t0 to T. arguments match miiso2_solve.
function miiso3_solve(
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
)
    miiso_solve(
        miiso3_step,
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
    )
end
