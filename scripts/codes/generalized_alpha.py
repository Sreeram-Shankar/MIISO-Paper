# gen-alpha amplification helpers used by fig_modal_gallery
import numpy as np


def amplification_matrix(Omega, alpha_m, alpha_f, beta):
    Om2 = Omega ** 2
    D = Om2 * alpha_f * beta - Om2 * beta + alpha_m - 1.0
    return np.array([
        [
            (Om2 * alpha_f * beta + alpha_m - 1.0) / D,
            (alpha_m - 1.0) / D,
            (alpha_m / 2.0 + beta - 0.5) / D,
        ],
        [
            Om2 * (alpha_f - alpha_m + 0.5) / D,
            (-Om2 * alpha_f ** 2 + Om2 * alpha_f * alpha_m + Om2 * alpha_f * beta
             + 0.5 * Om2 * alpha_f - Om2 * alpha_m - Om2 * beta + 0.5 * Om2
             + alpha_m - 1.0) / D,
            (-0.5 * Om2 * alpha_f ** 2 + 0.5 * Om2 * alpha_f * alpha_m
             + Om2 * alpha_f * beta + 0.25 * Om2 * alpha_f - 0.5 * Om2 * alpha_m
             - Om2 * beta + 0.25 * Om2 + alpha_f - 0.5) / D,
        ],
        [
            Om2 / D,
            Om2 * (1.0 - alpha_f) / D,
            (Om2 * alpha_f * beta - 0.5 * Om2 * alpha_f - Om2 * beta
             + 0.5 * Om2 + alpha_m) / D,
        ],
    ])


def classical_gen_alpha(rho_inf):
    alpha_m = (2.0 * rho_inf - 1.0) / (1.0 + rho_inf)
    alpha_f = rho_inf / (1.0 + rho_inf)
    beta = 1.0 / (1.0 + rho_inf) ** 2
    gamma = (3.0 - rho_inf) / (2.0 * (1.0 + rho_inf))
    return alpha_m, alpha_f, beta, gamma
