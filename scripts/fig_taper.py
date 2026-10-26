#2d plot of the transiton between 
#the conservative and dissipative Pades for all 5 miiso methods. DISCLAIMER:
# this file is mostly AI generated, because it 
# is almost all mechanical plotting, and whatever
# could be messed up was verified.
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

import fig_style as sty
sys.path.insert(0, sty.CODES)
import miiso2, miiso3, miiso4, miiso5, miiso6

sty.apply_style()
os.makedirs(sty.OUTDIR, exist_ok=True)

def find_span(n, p, u, U):
    if u >= U[n + 1]:
        return n
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def ders_basis_funs(span, u, p, n_deriv, U):
    ndu = np.zeros((p + 1, p + 1))
    left, right = np.zeros(p + 1), np.zeros(p + 1)
    ndu[0, 0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    ders = np.zeros((n_deriv + 1, p + 1))
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]
    a = np.zeros((2, p + 1))
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, n_deriv + 1):
            d, rk, pk = 0.0, r - k, p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else pk - r + k
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
    r = p
    for k in range(1, n_deriv + 1):
        for j in range(p + 1):
            ders[k, j] *= r
        r *= (p - k)
    return ders

def assemble_1d(n_el, p, order):
    interior = np.linspace(0.0, 1.0, n_el + 1)[1:-1]
    U = np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])
    n1 = len(U) - p - 1
    M = np.zeros((n1, n1))
    K = np.zeros((n1, n1))
    nq = p + 2
    xi, w = np.polynomial.legendre.leggauss(nq)
    xi = 0.5 * (xi + 1.0)
    w = 0.5 * w
    breaks = np.linspace(0.0, 1.0, n_el + 1)
    for e in range(n_el):
        a, b = breaks[e], breaks[e + 1]
        jac = b - a
        for g in range(nq):
            x = a + jac * xi[g]
            span = find_span(n1 - 1, p, x, U)
            ders = ders_basis_funs(span, x, p, order, U)
            N0, D = ders[0], ders[order]
            wt = w[g] * jac
            i0 = span - p
            for ai in range(p + 1):
                I = i0 + ai
                for aj in range(p + 1):
                    J = i0 + aj
                    M[I, J] += wt * N0[ai] * N0[aj]
                    K[I, J] += wt * D[ai] * D[aj]
    return M, K, n1

p, n_el, steps = 4, 20, 2.5
M, K, n1 = assemble_1d(n_el, p, 1)
free = np.arange(1, n1 - 1)
lambda_ = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True)
omega = np.sqrt(np.maximum(lambda_, 0.0))
N = len(omega)
k = int(min(max(miiso2.outlier_count(("bar", "fixed", 4)), 1), N - 1))
i_out = N - k
h = (2.0 * np.pi / steps) / omega[i_out - 1]
omega_c = float(h * omega[i_out])

r, s = 0.99, 4
x_lo, x_hi = 0.985, 1.005
ratio = np.linspace(x_lo, x_hi, 2001)
omega0_ratio, omega_c_ratio = r, 1.0
omega_abs = ratio * omega_c
omega0 = r * omega_c

theta = np.zeros_like(ratio)
mid = (ratio > omega0_ratio) & (ratio < 1.0)
theta[mid] = ((ratio[mid] - omega0_ratio) / (1.0 - omega0_ratio)) ** s
theta[ratio >= 1.0] = 1.0

blends = [
    ("MIISO2", miiso2.R_blend, "-"),
    ("MIISO3", miiso3.R_blend, "-"),
    ("MIISO4", miiso4.R_blend, "--"),
    ("MIISO5", miiso5.R_blend, "-."),
    ("MIISO6", miiso6.R_blend, ":"),
]

fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.55))

ax = axes[0]
ax.axvspan(omega0_ratio, omega_c_ratio, color="0.88", alpha=0.9, lw=0, zorder=0)
ax.plot(ratio, theta, color="0.12", lw=2.15, zorder=2)
ax.axvline(omega0_ratio, color="0.40", ls="--", lw=1.05)
ax.axvline(omega_c_ratio, color="0.40", ls="--", lw=1.05)
ax.text(omega0_ratio - 0.0006, 1.04, r"$\omega_0$", ha="right", va="top", fontsize=10, color="0.30")
ax.text(omega_c_ratio + 0.0006, 1.04, r"$\omega_c$", ha="left", va="top", fontsize=10, color="0.30")
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(-0.02, 1.06)
ax.set_xticks([0.985, 0.99, 1.00, 1.005])
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_xlabel(r"$\omega/\omega_c$")
ax.set_ylabel(r"$\theta(\omega)$")
ax.set_title(r"Taper ($r=0.99$, $s=4$)")
sty.style_axes(ax)

ax = axes[1]
ax.axvspan(omega0_ratio, omega_c_ratio, color="0.88", alpha=0.9, lw=0, zorder=0)
z = 1j * omega_abs
for name, blend, ls in blends:
    ax.plot(ratio, np.abs(blend(z, omega0, omega_c, p=s)),
            color=sty.COLORS[name], ls=ls, lw=2.0, label=name, zorder=2)
ax.axvline(omega0_ratio, color="0.40", ls="--", lw=1.05)
ax.axvline(omega_c_ratio, color="0.40", ls="--", lw=1.05)
ax.text(omega0_ratio - 0.0006, 1.015, r"$\omega_0$", ha="right", va="top", fontsize=10, color="0.30")
ax.text(omega_c_ratio + 0.0006, 1.015, r"$\omega_c$", ha="left", va="top", fontsize=10, color="0.30")
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(0.0, 1.02)
ax.set_xticks([0.985, 0.99, 1.00, 1.005])
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_xlabel(r"$\omega/\omega_c$")
ax.set_ylabel(r"$|R(i\omega)|$")
ax.set_title(r"Imaginary-axis amplification")
ax.legend(loc="lower left", handlelength=2.4)
sty.style_axes(ax)

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.86, wspace=0.28)
path = os.path.join(sty.OUTDIR, "Fig5.png")
fig.savefig(path)
plt.close(fig)
paper = os.path.join(os.path.dirname(sty.HERE), "paper", "figures", "Fig5.png")
open(paper, "wb").write(open(path, "rb").read())
