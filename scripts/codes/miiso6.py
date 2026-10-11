"""Rudimentary MIISO6 amplification maps (R_C / R_D / R_blend only).

R_C = Gauss (3,3). R_D = Padé (0,6) repaired by (2,4) on the 0/6
instability islands about the LHP poles of R06 (≈ −0.80 ± 3.70i).
"""

import numpy as np

# Gauss / conservative 6th-order 3/3 diagonal Padé
def R_C(z):
    z = complex(z)
    return (1 + z / 2 + z ** 2 / 10 + z ** 3 / 120) / (
        1 - z / 2 + z ** 2 / 10 - z ** 3 / 120
    )


def Q06(z):
    z = complex(z)
    return 1 - z + z ** 2 / 2 - z ** 3 / 6 + z ** 4 / 24 - z ** 5 / 120 + z ** 6 / 720


def Q05(z):
    z = complex(z)
    return 1 - z + z ** 2 / 2 - z ** 3 / 6 + z ** 4 / 24 - z ** 5 / 120


def R06(z):
    return 1 / Q06(z)


def R24(z):
    z = complex(z)
    return (1 + z / 3 + z ** 2 / 30) / (
        1 - 2 * z / 3 + z ** 2 / 5 - z ** 3 / 30 + z ** 4 / 360
    )


def F06(z):
    return abs(Q06(z)) ** 2 - 1.0


def _pole06(z0=-0.8 + 3.7j):
    z = complex(z0)
    for _ in range(50):
        dz = Q06(z) / (-Q05(z))
        z -= dz
        if abs(dz) < 1e-15:
            break
    return z


P06 = _pole06()


def _ray_boundary(alpha, rmax=1.0, n=1001):
    d = np.exp(1j * alpha)
    rs = np.linspace(0.0, rmax, n)
    Fs = np.empty(n, float)
    for j in range(n):
        Fs[j] = F06(P06 + rs[j] * d)
    for j in range(n - 1):
        if Fs[j] < 0 and Fs[j + 1] >= 0:
            a, b = float(rs[j]), float(rs[j + 1])
            for _ in range(50):
                m = 0.5 * (a + b)
                if F06(P06 + m * d) < 0:
                    a = m
                else:
                    b = m
            return 0.5 * (a + b)
    return 0.0


_ANGLES = np.linspace(0.0, 2 * np.pi, 501)
_RAYMAP = np.empty(_ANGLES.size, float)
for _i in range(_ANGLES.size):
    _RAYMAP[_i] = _ray_boundary(_ANGLES[_i])


def _interp_ray(alpha):
    if alpha <= _ANGLES[0]:
        return float(_RAYMAP[0])
    if alpha >= _ANGLES[-1]:
        return float(_RAYMAP[-1])
    return float(np.interp(alpha, _ANGLES, _RAYMAP))


def theta_wedge(z, pad=0.10):
    z = complex(z)
    d = complex(z.real, abs(z.imag)) - P06
    r = abs(d)
    if r < 1e-14:
        return 1.0
    alpha = float(np.mod(np.angle(d), 2 * np.pi))
    rb = _interp_ray(alpha)
    if rb <= 0:
        return 0.0
    s = r / rb
    if s <= 1.0:
        return 1.0
    if s >= 1.0 + pad:
        return 0.0
    return (1.0 + pad - s) / pad


def R_D(z):
    z = complex(z)
    th = theta_wedge(z)
    if th <= 0.0:
        return R06(z)
    if th >= 1.0:
        return R24(z)
    r0, r1 = R06(z), R24(z)
    mag = (abs(r1) ** th) * (abs(r0) ** (1.0 - th))
    arg = (1.0 - th) * np.angle(r0) + th * np.angle(r1)
    return mag * np.exp(1j * arg)


def R_blend(z_vals, omega0, omega_c, p=4):
    scalar = np.ndim(z_vals) == 0
    z_vals = np.asarray(z_vals, complex).reshape(-1)
    n = z_vals.size
    if n == 0:
        return z_vals.copy()
    omega0, omega_c = float(omega0), float(omega_c)
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
        c_vals = np.empty(idx.size, complex)
        for j in range(idx.size):
            c_vals[j] = R_C(z_vals[idx[j]])
        out[idx] = c_vals
    if np.any(only_d):
        idx = np.where(only_d)[0]
        d_vals = np.empty(idx.size, complex)
        for j in range(idx.size):
            d_vals[j] = R_D(z_vals[idx[j]])
        out[idx] = d_vals
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
