"""Paper Fig.7–12: fully discrete modal amplification gallery.

All bar / beam / 2D / cube setups used in the paper. Writes
figures/Fig7.png … figures/Fig12.png.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import eigh

from fig_style import CODES, COLORS, OUTDIR, apply_style, style_axes

sys.path.insert(0, str(CODES))
import miiso2
import miiso3
import miiso4
import miiso5
import miiso6
from generalized_alpha import amplification_matrix, classical_gen_alpha

N_EL_1D = 20
STEPS_PER_PERIOD = 0.5

MIISO = [
    ("MIISO2", miiso2.R_blend, "D", COLORS["MIISO2"]),
    ("MIISO3", miiso3.R_blend, "s", COLORS["MIISO3"]),
    ("MIISO4", miiso4.R_blend, "o", COLORS["MIISO4"]),
    ("MIISO5", miiso5.R_blend, "^", COLORS["MIISO5"]),
    ("MIISO6", miiso6.R_blend, "v", COLORS["MIISO6"]),
]
GA_STYLE = [
    (0.0, "-", "o", "#56B4E9", r"gen-$\alpha$, $\rho_\infty=0$"),
    (0.5, "--", "v", "#999999", r"gen-$\alpha$, $\rho_\infty=0.5$"),
    (1.0, ":", "^", "#000000", r"gen-$\alpha$, $\rho_\infty=1$"),
]

# (output Fig name, list of setups)
GROUPS = [
    ("Fig7.png", [
        ("bar", "fixed", 3),
        ("bar", "fixed", 4),
        ("bar", "fixed", 5),
    ]),
    ("Fig8.png", [
        ("bar", "free", 3),
        ("bar", "free", 4),
        ("bar", "free", 5),
    ]),
    ("Fig9.png", [
        ("bar", "fixed", 4, 2, 12),
        ("bar", "fixed", 4, 2, 24),
    ]),
    ("Fig10.png", [
        ("bar", "free", 4, 2, 12),
    ]),
    ("Fig11.png", [
        ("beam", "fixed", 4),
        ("beam", "pinned", 4),
        ("beam", "free", 4),
    ]),
    ("Fig12.png", [
        ("bar", "fixed", 3, 3, 12),
        ("bar", "fixed", 4, 3, 12),
    ]),
]


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


def dirichlet_end(problem, bc_end):
    if problem == "bar":
        if bc_end == "fixed":
            return 1
        return 0
    if problem == "beam":
        if bc_end == "fixed":
            return 2
        if bc_end == "pinned":
            return 1
        if bc_end == "free":
            return 0
        raise KeyError(bc_end)
    raise ValueError(problem)


def free_slice_1d(problem, bc, n1):
    left = dirichlet_end(problem, bc)
    right = dirichlet_end(problem, bc)
    return np.arange(left, n1 - right)


def spectrum(setup):
    problem = setup[0]
    bc = setup[1]
    p = int(setup[2])
    if len(setup) == 3:
        if problem == "bar":
            order = 1
        else:
            order = 2
        M, K, n1 = assemble_1d(N_EL_1D, p, order)
        free = free_slice_1d(problem, bc, n1)
        Mf = M[np.ix_(free, free)]
        Kf = K[np.ix_(free, free)]
    else:
        dim = int(setup[3])
        n = int(setup[4])
        n_el = n - p
        M1, K1, n1 = assemble_1d(n_el, p, 1)
        f1 = free_slice_1d(problem, bc, n1)
        if dim == 2:
            M = np.kron(M1, M1)
            K = np.kron(K1, M1) + np.kron(M1, K1)
            free_list = []
            for i in f1:
                for j in f1:
                    free_list.append(i * n1 + j)
            free = np.array(free_list, dtype=int)
        elif dim == 3:
            M = np.kron(np.kron(M1, M1), M1)
            K = (np.kron(np.kron(K1, M1), M1)
                 + np.kron(np.kron(M1, K1), M1)
                 + np.kron(np.kron(M1, M1), K1))
            free_list = []
            for i in f1:
                for j in f1:
                    for k in f1:
                        free_list.append(i * n1 * n1 + j * n1 + k)
            free = np.array(free_list, dtype=int)
        else:
            raise ValueError("dim must be 2 or 3")
        Mf = M[np.ix_(free, free)]
        Kf = K[np.ix_(free, free)]
    lam = eigh(Kf, Mf, eigvals_only=True)
    return np.sqrt(np.maximum(lam, 0.0))


def step_and_cutoff(setup):
    omega = spectrum(setup)
    N = len(omega)
    k = int(min(max(miiso2.outlier_count(setup), 1), N - 1))
    i_out = N - k
    w_last_phys = omega[i_out - 1]
    w_first_out = omega[i_out]
    h = (2.0 * np.pi / STEPS_PER_PERIOD) / w_last_phys
    omega_c = h * w_first_out
    return h, omega_c, 0.99 * omega_c, k


def rho_gen_alpha(omega, h, alpha_m, alpha_f, beta):
    out = np.empty(len(omega))
    for i, w in enumerate(omega):
        G = amplification_matrix(w * h, alpha_m, alpha_f, beta)
        out[i] = np.max(np.abs(np.linalg.eigvals(G)))
    return out


def rho_miiso(omega, h, omega0, omega_c, R_blend):
    values = []
    for w in omega:
        values.append(abs(R_blend(1j * (w * h), omega0, omega_c, p=4)))
    return np.array(values)


def panel_title(setup, k):
    problem = setup[0]
    bc = setup[1]
    p = int(setup[2])
    if len(setup) >= 5:
        dim = int(setup[3])
        n = int(setup[4])
        if dim == 3:
            return r"fixed cube $p=%d$, $n=%d$ ($k=%d$)" % (p, n, k)
        if problem == "bar":
            kind = "bar"
        else:
            kind = "beam"
        return r"%s %s $p=%d$, $n=%d$ ($k=%d$)" % (bc, kind, p, n, k)
    if problem == "bar":
        kind = "bar"
    else:
        kind = "beam"
    return r"%s %s $p=%d$ ($k=%d$)" % (bc, kind, p, k)


def _markevery(N):
    if N <= 16:
        return max(2, N // 6)
    return max(3, N // 12)


def plot_one(ax, setup, annotate_outlier=False):
    omega = spectrum(setup)
    N = len(omega)
    h, omega_c, omega0, k = step_and_cutoff(setup)
    n = np.arange(1, N + 1)
    me = _markevery(N)

    x_out = N - k + 0.5
    ax.axvspan(x_out, N + 0.5, color="0.55", alpha=0.10, lw=0, zorder=0)
    ax.axvline(x_out, color="0.45", ls="-", lw=0.7, zorder=1)
    ax.axhline(1.0, color="0.65", ls=":", lw=0.65, zorder=1)

    for _rho_inf, ls, marker, color, label in GA_STYLE:
        am, af, b, _g = classical_gen_alpha(_rho_inf)
        rho = rho_gen_alpha(omega, h, am, af, b)
        ax.plot(
            n, rho, linestyle=ls, marker=marker, ms=3.2, lw=1.15,
            color=color, label=label, markevery=me,
            markeredgecolor="white", markeredgewidth=0.3, zorder=2,
        )

    for label, R_blend, marker, color in MIISO:
        rho = rho_miiso(omega, h, omega0, omega_c, R_blend)
        ax.plot(
            n, rho, linestyle="-", marker=marker, ms=3.8, lw=1.85,
            color=color, label=label, markevery=me,
            markeredgecolor="white", markeredgewidth=0.35, zorder=3,
        )

    if annotate_outlier:
        ax.text(
            x_out + 0.15 * max(N * 0.02, 0.4), 0.08, "outliers",
            color="0.35", fontsize=8, ha="left", va="bottom", clip_on=True,
        )
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlim(0.5, N + 0.5)
    ax.set_title(panel_title(setup, k), pad=4)
    style_axes(ax)
    return h, k, N


def legend_handles():
    handles = []
    for _rho, ls, marker, color, label in GA_STYLE:
        handles.append(Line2D(
            [0], [0], color=color, marker=marker, linestyle=ls,
            lw=1.15, ms=4.0, markeredgecolor="white", markeredgewidth=0.3,
            label=label,
        ))
    for label, _R, marker, color in MIISO:
        handles.append(Line2D(
            [0], [0], color=color, marker=marker, linestyle="-",
            lw=1.85, ms=4.2, markeredgecolor="white", markeredgewidth=0.35,
            label=label,
        ))
    return handles


def _add_shared_legend(fig):
    handles = legend_handles()
    kw = dict(
        loc="upper center", frameon=True,
        fancybox=False, edgecolor="0.55", framealpha=1.0,
        columnspacing=1.6, handletextpad=0.6, borderaxespad=0.45,
        handlelength=2.8, fontsize=13, borderpad=0.55,
    )
    leg_ga = fig.legend(handles=handles[:3], bbox_to_anchor=(0.5, 1.07), ncol=3, **kw)
    leg_mi = fig.legend(handles=handles[3:], bbox_to_anchor=(0.5, 0.995), ncol=5, **kw)
    fig.add_artist(leg_ga)
    for leg in (leg_ga, leg_mi):
        for line in leg.get_lines():
            line.set_linewidth(2.8)
        hs = getattr(leg, "legend_handles", None)
        if not hs:
            hs = getattr(leg, "legendHandles", [])
        for handle in hs:
            if hasattr(handle, "set_markersize"):
                handle.set_markersize(9.5)


def save_group(setups, out_name, show_legend=True):
    n = len(setups)
    if show_legend:
        top = 0.56
    else:
        top = 0.90

    if n == 1:
        if show_legend:
            height = 4.3
        else:
            height = 3.5
        fig = plt.figure(figsize=(5.6, height))
        ax = fig.add_subplot(1, 1, 1)
        plot_one(ax, setups[0], annotate_outlier=True)
        axes = [ax]
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=top)
    elif n == 2:
        if show_legend:
            w = 9.6
            h = 4.7
        else:
            w = 10.2
            h = 3.45
        fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
        for i, setup in enumerate(setups):
            plot_one(axes[i], setup, annotate_outlier=(i == 0))
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=top, wspace=0.14)
    elif n == 3:
        if show_legend:
            w = 10.8
            h = 5.0
        else:
            w = 12.8
            h = 3.45
        fig, axes = plt.subplots(1, 3, figsize=(w, h), sharey=True)
        for i, setup in enumerate(setups):
            plot_one(axes[i], setup, annotate_outlier=(i == 0))
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.11, top=top, wspace=0.12)
    else:
        raise ValueError("save_group expects 1--3 panels, got %d" % n)

    for ax in axes:
        ax.set_xlabel("mode index")
    axes[0].set_ylabel(r"spectral radius $\rho$")
    if show_legend:
        _add_shared_legend(fig)
    path = OUTDIR / out_name
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=== modal gallery Fig.5–10 ===")
    for i, (out_name, setups) in enumerate(GROUPS):
        path = save_group(setups, out_name, show_legend=(i == 0))
        print("  wrote %s" % path)
    print("=== Done ===")


if __name__ == "__main__":
    main()
