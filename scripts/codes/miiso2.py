# MIISO2 maps used by the outlier mapping and amplification plots.
#it is the same as the julia proper src files but is an older, python version.
import numpy as np


def _outlier_end_k(p, problem, bc):
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


def _outlier_axis_k(p, problem, bci):
    if isinstance(bci, (tuple, list)):
        left, right = bci
    else:
        left = bci
        right = bci
    return _outlier_end_k(p, problem, left) + _outlier_end_k(p, problem, right)


def outlier_count(setup):
    problem = setup[0]
    bc = setup[1]
    p = int(setup[2])
    if len(setup) == 3:
        return _outlier_axis_k(p, problem, bc)
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
        ki = _outlier_axis_k(p, problem, bci)
        N *= int(ni)
        Nhat *= int(ni) - ki
    return N - Nhat


def R_C(z):
    z = complex(z)
    return (1.0 + 0.5 * z) / (1.0 - 0.5 * z)


def R_D(z):
    z = complex(z)
    return 1.0 / (1.0 - z + 0.5 * z ** 2)


def R_blend(z_vals, omega0, omega_c, p=4):
    scalar = np.ndim(z_vals) == 0
    z_vals = np.asarray(z_vals, complex).reshape(-1)
    n = z_vals.size
    omega0 = float(omega0)
    omega_c = float(omega_c)
    th = np.empty(n, float)
    for i, z in enumerate(z_vals):
        w = abs(z)
        if w <= omega0:
            th[i] = 0.0
        elif w >= omega_c:
            th[i] = 1.0
        else:
            th[i] = ((w - omega0) / (omega_c - omega0)) ** p
    out = np.empty(n, complex)
    only_c = th <= 0.0
    only_d = th >= 1.0
    mid = (th > 0.0) & (th < 1.0)
    if np.any(only_c):
        idx = np.where(only_c)[0]
        for j in range(idx.size):
            out[idx[j]] = R_C(z_vals[idx[j]])
    if np.any(only_d):
        idx = np.where(only_d)[0]
        for j in range(idx.size):
            out[idx[j]] = R_D(z_vals[idx[j]])
    if np.any(mid):
        idx = np.where(mid)[0]
        zm = z_vals[idx]
        thm = th[idx]
        Rc = np.empty(zm.size, complex)
        Rd = np.empty(zm.size, complex)
        for j in range(zm.size):
            Rc[j] = R_C(zm[j])
            Rd[j] = R_D(zm[j])
        order = np.argsort(np.abs(zm))
        arg_c = np.empty(idx.size, float)
        arg_d = np.empty(idx.size, float)
        arg_c[order] = np.unwrap(np.angle(Rc[order]))
        arg_d[order] = np.unwrap(np.angle(Rd[order]))
        mag = (np.abs(Rd) ** thm) * (np.abs(Rc) ** (1.0 - thm))
        out[idx] = mag * np.exp(1j * ((1.0 - thm) * arg_c + thm * arg_d))
    if scalar:
        return complex(out[0])
    return out
