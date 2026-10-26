#MIISO2, Modal Integrator for Isogeometric Stabilization of Outliers (order 2).
#the construction and methd is described in the paper. The pade maps
#are the [1,1] and [0,2] Pades. This method has no repair

include(joinpath(@__DIR__, "utilities.jl"))

using LinearAlgebra

#conservative 2nd order 1/1 Pade (A-stable, non-dissipative) for the physical modes
function R_C(z)
    z = ComplexF64(z)
    return (1 + z/2) / (1 - z/2)
end

#dissipative 2nd order 0/2 Pade (L-stable) for the outlier modes
function R_D(z)
    z = ComplexF64(z)
    return 1 / (1 - z + z^2/2)
end

#warm the small-|z| Taylor series of R_C used by phi_blend
rc_series(R_C)


#one step gives y = y + h phi1(hJ) f + (h^2/2) f_t
function miiso2_step(
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
    ws === nothing && (
        ws = (
            krylov_ws_new(length(y), clamp(km, 1, length(y))),
            krylov_ws_new(length(y), clamp(km, 1, length(y))),
        )
    )
    foreach(reset_phi_pool!, ws)
    tol = tau * (1 + norm(y)) / h
    φ1 = apply_phi(
        ws[1],
        J,
        fn,
        h,
        float(omega_frac) * oc,
        oc,
        km,
        R_C,
        R_D,
        [(1, 1.0)],
        tol;
        p = taper_order,
    )[1]
    @. y += h * φ1 + 0.5 * h^2 * ft
    return y
end


#integrate y'=f from t0 to T with fixed step h
function miiso2_solve(
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
        miiso2_step,
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
