# Chung–Hulbert generalized-α with classical ρ∞ parameters.
# Full Newton: Kt(q_af) + K_eff factor every Newton iteration; residual every iter.
#
#   y, nfac = genalpha_solve(fint, Kt, M, mass_solve, u0, v0, T, h; rho_inf=1.0)
#
# Optional Rayleigh C and force(t) for M ü + C ú + fint(u) = force(t):
#   genalpha_solve(..., C=C, force=force)
#
# fint(u) → internal force,  Kt(u) → sparse tangent,
# M → sparse/dense mass,  mass_solve(r) → M⁻¹ r.
# Residual: M a_am + C v_af + fint(q_af) = force(t),  K_eff = (1−α_m)M + c_C C + c_K Kt.
# ρ∞ = 1 is spectrally undamped. Returned state is [u; ú].

using SparseArrays
using LinearAlgebra

function classical_gen_alpha(rho_inf)
    ρ = float(rho_inf)
    αm = (2ρ - 1) / (1 + ρ)
    αf = ρ / (1 + ρ)
    β = 1 / (1 + ρ)^2
    γ = (3 - ρ) / (2 * (1 + ρ))
    return αm, αf, β, γ
end

# HHT-α ⊂ generalized-α with α_m = 0. Hilber α ∈ [-1/3, 0]; α = -0.05 is the
# usual structural / bridge default (mild high-frequency damping).
function hht_alpha_params(α=-0.05)
    αm = 0.0
    αf = -float(α)
    β = ((1 - αm + αf) / 2)^2
    γ = 0.5 - αm + αf
    return αm, αf, β, γ
end

function genalpha_solve(fint, Kt, M, mass_solve, u0, v0, T, h; rho_inf=1.0, t0=0.0,
                        callback=nothing, progress=nothing, C=nothing, force=nothing,
                        params=nothing)
    αm, αf, β, γ = params === nothing ? classical_gen_alpha(rho_inf) : params
    u, v = copy(u0), copy(v0)
    F0 = force === nothing ? zero(u) : force(float(t0))
    a = mass_solve(F0 .- fint(u))
    an, r, da = similar(u), similar(u), similar(u)
    q_pred, v_pred, q_af, v_af = similar(u), similar(u), similar(u), similar(u)
    nfac = 0
    t = float(t0)
    nsteps = max(Int(round((float(T) - t) / float(h))), 0)
    nsteps == 0 && return vcat(u, v), nfac
    h = (float(T) - t) / nsteps
    Mam = (1 - αm) * M
    cK = (1 - αf) * β * h^2
    cC = C === nothing ? 0.0 : (1 - αf) * γ * h
    callback !== nothing && callback(t, vcat(u, v))
    progress !== nothing && progress(0, nsteps, 0, t)
    for istep in 1:nsteps
        @. q_pred = u + h * v + h^2 * (0.5 - β) * a
        @. v_pred = v + h * (1 - γ) * a
        copyto!(an, a)
        tn1 = t + h
        t_af = (1 - αf) * tn1 + αf * t
        Faf = force === nothing ? zero(u) : force(t_af)
        Rscale = max(norm(M * a), norm(Faf), 1.0)
        nnewt = 0
        for _ in 1:30
            nnewt += 1
            @. q_af = (1 - αf) * (q_pred + β * h^2 * a) + αf * u
            @. v_af = (1 - αf) * (v_pred + γ * h * a) + αf * v
            mul!(r, M, (1 - αm) * a + αm * an)
            r .+= fint(q_af) .- Faf
            if C !== nothing
                mul!(da, C, v_af)
                r .+= da
            end
            Keff = Mam + cK * Kt(q_af)
            C !== nothing && (Keff = Keff + cC * C)
            nfac += 1
            da .= cholesky(Symmetric(Keff)) \ (-r)
            a .+= da
            progress !== nothing && progress(istep, nsteps, nnewt, t)
            (norm(r) <= 1e-8 * Rscale && norm(da) <= 1e-8 * (1 + norm(a))) && break
        end
        @. u = q_pred + β * h^2 * a
        @. v = v_pred + γ * h * a
        t = tn1
        callback !== nothing && callback(t, vcat(u, v))
        progress !== nothing && progress(istep, nsteps, nnewt, t)
    end
    return vcat(u, v), nfac
end
