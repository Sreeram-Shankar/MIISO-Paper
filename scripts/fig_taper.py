"""Paper Fig.5: taper θ(ω) and imaginary-axis |R(iω)| for MIISO2–6.

Uses ω_c from the fixed–fixed p=4 bar at a coarse step (few steps per
period) so absolute |z| is large enough that |R_D| damps visibly past ω_c.

Writes figures/Fig5.png.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from fig_style import CODES, COLORS, OUTDIR, apply_style, style_axes

sys.path.insert(0, str(CODES))
import miiso2
import miiso3
import miiso4
import miiso5
import miiso6

R, S = 0.99, 4
SETUP = ("bar", "fixed", 4)
N_EL_1D = 20
# Moderate coarse step: large enough absolute ω_c that |R_D| damps,
# small enough that MIISO3--6 stay distinguishable past ω_c.
STEPS_PER_PERIOD = 2.5
# Original tight window around the taper band.
X_LO, X_HI = 0.985, 1.005
X_TICKS = [0.985, 0.99, 1.00, 1.005]
METHOD_LS = {
    "MIISO2": "-", "MIISO3": "-", "MIISO4": "--",
    "MIISO5": "-.", "MIISO6": ":",
}
BLEND = {
    "MIISO2": miiso2.R_blend,
    "MIISO3": miiso3.R_blend,
    "MIISO4": miiso4.R_blend,
    "MIISO5": miiso5.R_blend,
    "MIISO6": miiso6.R_blend,
}


def open_uniform_knots(n_el, p):
    interior = np.linspace(0.0, 1.0, n_el + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])


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
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
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
    U = open_uniform_knots(n_el, p)
    n1 = len(U) - p - 1
    M = np.zeros((n1, n1))
    K = np.zeros((n1, n1))
    nq = p + 2
    xi, w = np.polynomial.legendre.leggauss(nq)
    xi = 0.5 * (xi + 1.0)
    w = 0.5 * w
    breaks = np.linspace(0.0, 1.0, n_el + 1)
    for e in range(n_el):
        a = breaks[e]
        b = breaks[e + 1]
        jac = b - a
        for g in range(nq):
            x = a + jac * xi[g]
            span = find_span(n1 - 1, p, x, U)
            ders = ders_basis_funs(span, x, p, order, U)
            N0 = ders[0]
            D = ders[order]
            wt = w[g] * jac
            i0 = span - p
            for ai in range(p + 1):
                I = i0 + ai
                for aj in range(p + 1):
                    J = i0 + aj
                    M[I, J] += wt * N0[ai] * N0[aj]
                    K[I, J] += wt * D[ai] * D[aj]
    return M, K, n1


def omega_c_bar_fixed(p, steps_per_period):
    M, K, n1 = assemble_1d(N_EL_1D, p, 1)
    free = np.arange(1, n1 - 1)
    Mf = M[np.ix_(free, free)]
    Kf = K[np.ix_(free, free)]
    lam = eigh(Kf, Mf, eigvals_only=True)
    omega = np.sqrt(np.maximum(lam, 0.0))
    N = len(omega)
    k = int(min(max(miiso2.outlier_count(SETUP), 1), N - 1))
    i_out = N - k
    h = (2.0 * np.pi / steps_per_period) / omega[i_out - 1]
    return float(h * omega[i_out]), h, k


def theta_taper(w, r=R, s=S):
    w = np.asarray(w, float)
    w0 = r
    th = np.zeros_like(w)
    mid = (w > w0) & (w < 1.0)
    th[mid] = ((w[mid] - w0) / (1.0 - w0)) ** s
    th[w >= 1.0] = 1.0
    return th


def mark_onset_cutoff(ax, w0, oc, y):
    ax.axvline(w0, color="0.40", ls="--", lw=1.05, zorder=1)
    ax.axvline(oc, color="0.40", ls="--", lw=1.05, zorder=1)
    ax.text(w0 - 0.0006, y, r"$\omega_0$", ha="right", va="top",
            fontsize=10, color="0.30")
    ax.text(oc + 0.0006, y, r"$\omega_c$", ha="left", va="top",
            fontsize=10, color="0.30")


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    paper_fig = OUTDIR.parents[1] / "paper" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "Fig5.png"

    omega_c, h, k = omega_c_bar_fixed(int(SETUP[2]), STEPS_PER_PERIOD)
    print("omega_c=%.4f  h=%.4g  k=%d  steps/period=%.2g" % (omega_c, h, k, STEPS_PER_PERIOD))

    # Left panel: dimensionless taper in ω/ω_c.
    # Right panel: |R(iω)| at absolute ω = (ω/ω_c)·ω_c (needs large ω_c to damp).
    ratio = np.linspace(X_LO, X_HI, 2001)
    w0_ratio = R
    oc_ratio = 1.0
    omega_abs = ratio * omega_c
    w0_abs = R * omega_c

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.55))

    ax = axes[0]
    ax.axvspan(w0_ratio, oc_ratio, color="0.88", alpha=0.9, lw=0, zorder=0)
    ax.plot(ratio, theta_taper(ratio), color="0.12", lw=2.15, zorder=2)
    mark_onset_cutoff(ax, w0_ratio, oc_ratio, y=1.04)
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xticks(X_TICKS)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(r"$\omega/\omega_c$")
    ax.set_ylabel(r"$\theta(\omega)$")
    ax.set_title(r"Taper ($r=0.99$, $s=4$)")
    style_axes(ax)

    ax = axes[1]
    ax.axvspan(w0_ratio, oc_ratio, color="0.88", alpha=0.9, lw=0, zorder=0)
    z = 1j * omega_abs
    for name, blend in BLEND.items():
        mag = np.abs(blend(z, w0_abs, omega_c, p=S))
        ax.plot(ratio, mag, color=COLORS[name], ls=METHOD_LS[name],
                lw=2.0, label=name, zorder=2)
    mark_onset_cutoff(ax, w0_ratio, oc_ratio, y=1.015)
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(X_TICKS)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(r"$\omega/\omega_c$")
    ax.set_ylabel(r"$|R(i\omega)|$")
    ax.set_title(r"Imaginary-axis amplification")
    ax.legend(loc="lower left", handlelength=2.4)
    style_axes(ax)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.86, wspace=0.28)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
    (paper_fig / "Fig5.png").write_bytes(path.read_bytes())
    print("copied", paper_fig / "Fig5.png")


if __name__ == "__main__":
    main()
