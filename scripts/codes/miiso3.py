# MIISO3 maps used by fig scripts. It is the s ame as the julia proper src files but is an older, python version.
import numpy as np


def R_C(z):
    z = complex(z)
    return (1.0 + 0.5 * z + z ** 2 / 12.0) / (1.0 - 0.5 * z + z ** 2 / 12.0)


def R03(z):
    z = complex(z)
    return 1.0 / (1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0)


def R12(z):
    z = complex(z)
    return (1.0 + z / 3.0) / (1.0 - 2.0 * z / 3.0 + z ** 2 / 6.0)


def F03(x, y):
    re = 1.0 - x + 0.5 * (x * x - y * y) - (x ** 3 - 3.0 * x * y * y) / 6.0
    im = -y + x * y - (3.0 * x * x * y - y ** 3) / 6.0
    return re * re + im * im - 1.0


def instability_boundary(y, xlo=-0.2, n=2001):
    y = float(y)
    ay = abs(float(y))
    if ay <= 0.0 or ay >= np.sqrt(3.0):
        return 0.0
    xs = np.linspace(xlo, 0.0, n)
    hits = np.where(np.diff(np.signbit(F03(xs, y))))[0]
    if hits.size == 0:
        return 0.0
    i = int(hits[-1])
    a = float(xs[i])
    b = float(xs[i + 1])
    for _ in range(50):
        m = 0.5 * (a + b)
        if F03(a, y) * F03(m, y) <= 0.0:
            a, b = a, m
        else:
            a, b = m, b
    return 0.5 * (a + b)


spaces = np.linspace(0.0, np.sqrt(3.0), 1001)
boundary = np.empty(spaces.size, float)
for _i in range(spaces.size):
    boundary[_i] = instability_boundary(spaces[_i])


def theta_wedge(z, pad=0.10):
    z = complex(z)
    x = z.real
    ay = abs(z.imag)
    y_max = (1.0 + pad) * np.sqrt(3.0)
    if ay >= y_max:
        return 0.0
    if ay < np.sqrt(3.0):
        xb = float(np.interp(ay, spaces, boundary))
        if xb >= -1e-14:
            if x >= -1e-14 and ay > 0.0:
                return 1.0
            return 0.0
        x_outer = (1.0 + pad) * xb
        if x <= x_outer:
            return 0.0
        if x >= xb:
            return 1.0
        return (x - x_outer) / (xb - x_outer)
    s = (y_max - ay) / (pad * np.sqrt(3.0))
    x_outer = -pad * 0.04 * s
    if x <= x_outer:
        return 0.0
    if x >= 0.0:
        return s
    return s * (x - x_outer) / (0.0 - x_outer)


def R_D(z):
    z = complex(z)
    th = theta_wedge(z)
    if th <= 0.0:
        return R03(z)
    if th >= 1.0:
        return R12(z)
    r0 = R03(z)
    r1 = R12(z)
    mag = (abs(r1) ** th) * (abs(r0) ** (1.0 - th))
    arg = (1.0 - th) * np.angle(r0) + th * np.angle(r1)
    return mag * np.exp(1j * arg)


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
