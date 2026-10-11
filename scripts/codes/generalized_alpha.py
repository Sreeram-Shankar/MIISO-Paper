"""
Adaptive generalized-alpha with prescribed cutoff spectral radius rho_cutoff.

  K_tan(q_n) -> omega_c -> Omega_c -> (alpha_m, alpha_f, beta, gamma) -> step

At the tracked cutoff Omega_c the amplification satisfies
  det(lambda I - G(Omega_c)) = (lambda + rho_cutoff)^3,
so rho(G(Omega_c)) = rho_cutoff.  The case rho_cutoff = 0 is exact finite-step annihilation.

Outlier count k from Hiemstra et al., CMAME 387 (2021) 114115, Tables 3-4 / §4.3.
"""

import numpy as np
from scipy.linalg import eigh, solve
from scipy.optimize import brentq


# ---- Hiemstra outlier lookup ------------------------------------------------

def _k_end(p, bc, problem):
    p = int(p)
    if problem == "bar":
        if bc == "fixed":
            return (p - 1) // 2
        if bc == "free":
            return p // 2
    elif problem == "beam":
        if bc == "fixed":
            return max((p - 2) // 2, 0)
        if bc == "pinned":
            return (p - 1) // 2
        if bc == "free":
            return p // 2
    raise ValueError(f"unknown problem/bc: {problem!r}, {bc!r}")


def _k_1d(p, bc, problem):
    if isinstance(bc, (tuple, list)):
        left, right = bc
    else:
        left = bc
        right = bc
    return _k_end(p, left, problem) + _k_end(p, right, problem)


def outlier_count(setup):
    """
    setup = (problem, bc, order)
          = (problem, bc, order, dim, n)
    """
    problem = setup[0]
    bc = setup[1]
    p = int(setup[2])
    if len(setup) == 3:
        return _k_1d(p, bc, problem)
    dim = int(setup[3])
    n = setup[4]
    if isinstance(bc, list):
        bcs = bc
    else:
        bcs = [bc] * dim
    if isinstance(n, (tuple, list)):
        ns = list(n)
    else:
        ns = [int(n)] * dim
    N = 1
    Nhat = 1
    for ni, bci in zip(ns, bcs):
        ki = _k_1d(p, bci, problem)
        N *= int(ni)
        Nhat *= int(ni) - ki
    return N - Nhat


# ---- amplification matrix ---------------------------------------------------

def amplification_matrix(Omega, alpha_m, alpha_f, beta):
    """Frozen modal G(Omega); gamma = 1/2 + alpha_f - alpha_m assumed."""
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


# ---- cutoff parameter map (Omega_c, rho_cutoff) -> (alpha_m, alpha_f, beta, gamma)

def omega_c_min(rho_cutoff):
    """Feasibility threshold: Omega_c >= sqrt(12) * (1+rho_cutoff)/(1-rho_cutoff)."""
    return np.sqrt(12.0) * (1.0 + rho_cutoff) / (1.0 - rho_cutoff)


def classical_gen_alpha(rho_inf):
    """Classical generalized-alpha with asymptotic spectral radius rho_inf in [0,1]."""
    alpha_m = (2.0 * rho_inf - 1.0) / (1.0 + rho_inf)
    alpha_f = rho_inf / (1.0 + rho_inf)
    beta = 1.0 / (1.0 + rho_inf) ** 2
    gamma = (3.0 - rho_inf) / (2.0 * (1.0 + rho_inf))
    return alpha_m, alpha_f, beta, gamma


def _cutoff_residual(yy, Omega_c, rho_cutoff):
    return (
        Omega_c ** 2 * yy ** 3
        - (1.0 + rho_cutoff) ** 2 * (yy + 1.0 - rho_cutoff)
    )


def select_parameters(Omega_c, rho_cutoff=0.0):
    """
    Adaptive map (Omega_c, rho_cutoff) -> (alpha_m, alpha_f, beta, gamma).

    Impose det(lambda I - G(Omega_c)) = (lambda + rho_cutoff)^3, so the three
    amplification eigenvalues at the cutoff are -rho_cutoff (repeated).

    Auxiliary y in [0, (1-rho_cutoff)/2] solves
        Omega_c^2 y^3 = (1+rho_cutoff)^2 (y + 1 - rho_cutoff).

    Then
        alpha_m = (2 rho_cutoff - 1)/(1 + rho_cutoff)
        alpha_f = (y + rho_cutoff)/(1 + rho_cutoff)
        beta    = [(2-rho_cutoff)(y^2+y) + (1-rho_cutoff)] / [(1+rho_cutoff)^2 (y+1-rho_cutoff)]
        gamma   = (2 y + 3 - rho_cutoff) / [2 (1+rho_cutoff)]

    rho_cutoff = 0 recovers exact annihilation.
    If Omega_c is below the feasibility threshold, fall back to classical
    generalized-alpha with the same asymptotic radius rho_cutoff.
    """
    if not (0.0 <= rho_cutoff < 1.0):
        raise ValueError("rho_cutoff must satisfy 0 <= rho_cutoff < 1")

    thr = omega_c_min(rho_cutoff)
    if Omega_c < thr:
        return classical_gen_alpha(rho_cutoff)

    upper = 0.5 * (1.0 - rho_cutoff)
    y = brentq(_cutoff_residual, 0.0, upper, args=(Omega_c, rho_cutoff), xtol=1e-14)
    alpha_m = (2.0 * rho_cutoff - 1.0) / (1.0 + rho_cutoff)
    alpha_f = (y + rho_cutoff) / (1.0 + rho_cutoff)
    beta = (
        (2.0 - rho_cutoff) * (y ** 2 + y) + (1.0 - rho_cutoff)
    ) / (
        (1.0 + rho_cutoff) ** 2 * (y + 1.0 - rho_cutoff)
    )
    gamma = (2.0 * y + 3.0 - rho_cutoff) / (2.0 * (1.0 + rho_cutoff))
    return alpha_m, alpha_f, beta, gamma


# keep old name for scripts that imported it
OMEGA_C_MIN = np.sqrt(12.0)


# ---- spectral tracking ------------------------------------------------------

def track_outlier_subspace(K, M, k):
    K = np.asarray(K, float)
    M = np.asarray(M, float)
    n = K.shape[0]
    n_track = min(k + 1, n)
    lam, Phi = eigh(K, M, subset_by_index=[n - n_track, n - 1])
    if k >= 1 and n_track >= 2:
        omega_phys = np.sqrt(max(lam[0], 0.0))
        omega_c = np.sqrt(max(lam[1], 0.0))
    else:
        omega_phys = np.sqrt(max(lam[0], 0.0))
        omega_c = omega_phys
    return lam, Phi, float(omega_c), float(omega_phys)


# ---- one timestep -----------------------------------------------------------

def step(q_n, v_n, a_n, t_n, h, M, f_int, K_tan, f_ext, k, C=None,
         rho_cutoff=0.0, tol=1e-8, max_newton=25):
    """
    One adaptive gen-alpha step. Acceleration is the Newton unknown.

    a_{n+1-alpha_m} = (1-alpha_m) a + alpha_m a_n
    q = q_pred + beta h^2 a,  v = v_pred + gamma h a
    R = M a_am + C v_af + f_int(q_af) - f_ext(t_af)
    K_eff = (1-alpha_m) M + (1-alpha_f) gamma h C + (1-alpha_f) beta h^2 K_tan
    """
    q_n = np.asarray(q_n, float)
    v_n = np.asarray(v_n, float)
    a_n = np.asarray(a_n, float)
    M = np.asarray(M, float)

    lam, Phi, omega_c, omega_phys = track_outlier_subspace(K_tan(q_n), M, k)
    if omega_c <= 0.0:
        Omega_c = 0.0
    else:
        Omega_c = h * omega_c
    alpha_m, alpha_f, beta, gamma = select_parameters(Omega_c, rho_cutoff)

    q_pred = q_n + h * v_n + h ** 2 * (0.5 - beta) * a_n
    v_pred = v_n + h * (1.0 - gamma) * a_n

    a = a_n.copy()
    R_scale = 1.0
    for it in range(max_newton):
        q = q_pred + beta * h ** 2 * a
        v = v_pred + gamma * h * a
        q_af = (1.0 - alpha_f) * q + alpha_f * q_n
        v_af = (1.0 - alpha_f) * v + alpha_f * v_n
        t_af = t_n + (1.0 - alpha_f) * h
        a_am = (1.0 - alpha_m) * a + alpha_m * a_n

        R = M @ a_am + f_int(q_af) - f_ext(t_af)
        Keff = (
            (1.0 - alpha_m) * M
            + (1.0 - alpha_f) * beta * h ** 2 * np.asarray(K_tan(q_af), float)
        )
        if C is not None:
            C = np.asarray(C, float)
            R = R + C @ v_af
            Keff = Keff + (1.0 - alpha_f) * gamma * h * C

        da = solve(Keff, -R, assume_a="sym")
        a = a + da
        Rn = np.linalg.norm(R)
        if it == 0:
            R_scale = max(Rn, 1.0)
        if Rn <= tol * R_scale and np.linalg.norm(da) <= tol * (1.0 + np.linalg.norm(a)):
            break
    else:
        raise RuntimeError("Newton did not converge")

    q = q_pred + beta * h ** 2 * a
    v = v_pred + gamma * h * a
    return {
        "q": q, "v": v, "a": a, "t": t_n + h,
        "alpha_m": alpha_m, "alpha_f": alpha_f, "beta": beta, "gamma": gamma,
        "rho_cutoff": rho_cutoff, "Omega_c": Omega_c,
        "omega_c": omega_c, "omega_phys": omega_phys,
        "lam": lam, "Phi": Phi,
    }
