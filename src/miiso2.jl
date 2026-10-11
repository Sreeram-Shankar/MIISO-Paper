"""
MIISO2 — Modal Integrator for Isogeometric Stabilization of Outliers (order 2).

Spectral map (z = h λ, ω = |z|):
  ω₀ = r·ω_c,  θ(ω) = 0 / s^p / 1 on [0,ω₀] / (ω₀,ω_c) / [ω_c,∞)
  R(z) = |R_D|^θ |R_C|^(1−θ) · exp(i[(1−θ) arg R_C + θ arg R_D])

R_C = Padé (1,1), R_D = Padé (0,2)
ω_c from Omega_c, else IRAM on J with k from setup/k.
cutoff_adaptive: Rayleigh-update ω_c each step (true) or freeze after init (false).

Update:  y⁺ = y + h φ₁(hJ) f + (h²/2) f_t ,  φ₁ = (R−1)/z
φ₁ via Arnoldi + Ritz map (non-holomorphic). Cost O(N m² + m³).

Hiemstra outlier count: CMAME 387 (2021) 114115.
"""

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

rc_series(R_C)


"""One step: y ← y + h φ₁(hJ) f + (h²/2) f_t."""
function miiso2_step(f, t, y, h; setup=nothing, k=nothing, Omega_c=nothing,
                     J=nothing, ft=nothing, krylov_dim_max::Int=40, cutoff_adaptive=true,
                     omega_frac=0.99, taper_order=4, ws=nothing, tau=1e-10)
    y, fn, J, ft, oc, km = prepare(f, t, y, h; setup, k, Omega_c, J, ft, krylov_dim_max, cutoff_adaptive)
    ws === nothing && (ws = (krylov_ws_new(length(y), clamp(km, 1, length(y))), krylov_ws_new(length(y), clamp(km, 1, length(y)))))
    foreach(reset_phi_pool!, ws)
    tol = tau * (1 + norm(y)) / h
    φ1 = apply_phi(ws[1], J, fn, h, float(omega_frac) * oc, oc, km, R_C, R_D, [(1, 1.0)], tol; p=taper_order)[1]
    @. y += h * φ1 + 0.5 * h^2 * ft
    return y
end


"""Integrate y'=f from t0 to T with fixed step h."""
function miiso2_solve(f, y0, T, h; t0=0.0, setup=nothing, k=nothing, Omega_c=nothing,
                      J=nothing, ft=nothing, krylov_dim_max::Int=40, cutoff_adaptive=true,
                      omega_frac=0.99, taper_order=4, tau=1e-10, callback=nothing)
    miiso_solve(miiso2_step, f, y0, T, h; t0, setup, k, Omega_c, J, ft,
                krylov_dim_max, cutoff_adaptive, omega_frac, taper_order, tau, callback)
end
