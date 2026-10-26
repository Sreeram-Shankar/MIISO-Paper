#makes the modal gallery figures for figure 7-12 in the paper. DISCLAIMER:
# this file is mostly AI generated, because it 
# is almost all mechanical plotting, and whatever
# could be messed up was verified.
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import eigh

import fig_style as sty
sys.path.insert(0, sty.CODES)
import miiso2, miiso3, miiso4, miiso5, miiso6
from generalized_alpha import amplification_matrix, classical_gen_alpha

sty.apply_style()
os.makedirs(sty.OUTDIR, exist_ok=True)

n_el_1d = 20
steps_per_period = 0.5

miiso = [
    ("MIISO2", miiso2.R_blend, "D", sty.COLORS["MIISO2"]),
    ("MIISO3", miiso3.R_blend, "s", sty.COLORS["MIISO3"]),
    ("MIISO4", miiso4.R_blend, "o", sty.COLORS["MIISO4"]),
    ("MIISO5", miiso5.R_blend, "^", sty.COLORS["MIISO5"]),
    ("MIISO6", miiso6.R_blend, "v", sty.COLORS["MIISO6"]),
]
ga_style = [
    (0.0, "-", "o", "#56B4E9", r"gen-$\alpha$, $\rho_\infty=0$"),
    (0.5, "--", "v", "#999999", r"gen-$\alpha$, $\rho_\infty=0.5$"),
    (1.0, ":", "^", "#000000", r"gen-$\alpha$, $\rho_\infty=1$"),
]

groups = [
    ("Fig7.png", [
        ("bar", "fixed", 3), ("bar", "fixed", 4), ("bar", "fixed", 5),
    ]),
    ("Fig8.png", [
        ("bar", "free", 3), ("bar", "free", 4), ("bar", "free", 5),
    ]),
    ("Fig9.png", [
        ("bar", "fixed", 4, 2, 12), ("bar", "fixed", 4, 2, 24),
    ]),
    ("Fig10.png", [
        ("bar", "free", 4, 2, 12),
    ]),
    ("Fig11.png", [
        ("beam", "fixed", 4), ("beam", "pinned", 4), ("beam", "free", 4),
    ]),
    ("Fig12.png", [
        ("bar", "fixed", 3, 3, 12), ("bar", "fixed", 4, 3, 12),
    ]),
]

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
            j2 = k - 1 if r - 1 <= pk else p - r
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

def free_slice_1d(problem, bc, n1):
    if problem == "bar":
        end = 1 if bc == "fixed" else 0
    elif bc == "fixed":
        end = 2
    elif bc == "pinned":
        end = 1
    else:
        end = 0
    return np.arange(end, n1 - end)

def spectrum(setup):
    problem, bc, p = setup[0], setup[1], int(setup[2])
    if len(setup) == 3:
        order = 1 if problem == "bar" else 2
        M, K, n1 = assemble_1d(n_el_1d, p, order)
        free = free_slice_1d(problem, bc, n1)
    else:
        dim, n = int(setup[3]), int(setup[4])
        n_el = n - p
        M1, K1, n1 = assemble_1d(n_el, p, 1)
        f1 = free_slice_1d(problem, bc, n1)
        if dim == 2:
            M = np.kron(M1, M1)
            K = np.kron(K1, M1) + np.kron(M1, K1)
            free = np.array([i * n1 + j for i in f1 for j in f1], dtype=int)
        else:
            M = np.kron(np.kron(M1, M1), M1)
            K = (np.kron(np.kron(K1, M1), M1)
                 + np.kron(np.kron(M1, K1), M1)
                 + np.kron(np.kron(M1, M1), K1))
            free = np.array(
                [i * n1 * n1 + j * n1 + kk for i in f1 for j in f1 for kk in f1],
                dtype=int,
            )
    Mf = M[np.ix_(free, free)]
    Kf = K[np.ix_(free, free)]
    lambda_ = eigh(Kf, Mf, eigvals_only=True)
    return np.sqrt(np.maximum(lambda_, 0.0))

def plot_one(ax, setup, annotate_outlier=False):
    omega = spectrum(setup)
    N = len(omega)
    k = int(min(max(miiso2.outlier_count(setup), 1), N - 1))
    i_out = N - k
    h = (2.0 * np.pi / steps_per_period) / omega[i_out - 1]
    omega_c = h * omega[i_out]
    omega0 = 0.99 * omega_c
    n = np.arange(1, N + 1)
    me = max(2, N // 6) if N <= 16 else max(3, N // 12)
    x_out = N - k + 0.5
    ax.axvspan(x_out, N + 0.5, color="0.55", alpha=0.10, lw=0, zorder=0)
    ax.axvline(x_out, color="0.45", ls="-", lw=0.7, zorder=1)
    ax.axhline(1.0, color="0.65", ls=":", lw=0.65, zorder=1)
    for rho_inf, ls, marker, color, label in ga_style:
        am, af, b, _g = classical_gen_alpha(rho_inf)
        rho = np.empty(N)
        for i, w in enumerate(omega):
            G = amplification_matrix(w * h, am, af, b)
            rho[i] = np.max(np.abs(np.linalg.eigvals(G)))
        ax.plot(n, rho, linestyle=ls, marker=marker, ms=3.2, lw=1.15,
                color=color, label=label, markevery=me,
                markeredgecolor="white", markeredgewidth=0.3, zorder=2)
    for label, R_blend, marker, color in miiso:
        rho = np.array([abs(R_blend(1j * (w * h), omega0, omega_c, p=4)) for w in omega])
        ax.plot(n, rho, linestyle="-", marker=marker, ms=3.8, lw=1.85,
                color=color, label=label, markevery=me,
                markeredgecolor="white", markeredgewidth=0.35, zorder=3)
    if annotate_outlier:
        ax.text(x_out + 0.15 * max(N * 0.02, 0.4), 0.08, "outliers",
                color="0.35", fontsize=8, ha="left", va="bottom", clip_on=True)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlim(0.5, N + 0.5)
    problem, bc, p = setup[0], setup[1], int(setup[2])
    kind = "bar" if problem == "bar" else "beam"
    if len(setup) >= 5:
        dim, nn = int(setup[3]), int(setup[4])
        if dim == 3:
            title = r"fixed cube $p=%d$, $n=%d$ ($k=%d$)" % (p, nn, k)
        else:
            title = r"%s %s $p=%d$, $n=%d$ ($k=%d$)" % (bc, kind, p, nn, k)
    else:
        title = r"%s %s $p=%d$ ($k=%d$)" % (bc, kind, p, k)
    ax.set_title(title, pad=4)
    sty.style_axes(ax)

def add_shared_legend(fig):
    handles = []
    for _rho, ls, marker, color, label in ga_style:
        handles.append(Line2D(
            [0], [0], color=color, marker=marker, linestyle=ls,
            lw=1.15, ms=4.0, markeredgecolor="white", markeredgewidth=0.3,
            label=label,
        ))
    for label, _R, marker, color in miiso:
        handles.append(Line2D(
            [0], [0], color=color, marker=marker, linestyle="-",
            lw=1.85, ms=4.2, markeredgecolor="white", markeredgewidth=0.35,
            label=label,
        ))
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
        for handle in leg.legend_handles:
            handle.set_markersize(9.5)

for gi, (out_name, setups) in enumerate(groups):
    show_legend = (gi == 0)
    n = len(setups)
    top = 0.56 if show_legend else 0.90
    if n == 1:
        fig = plt.figure(figsize=(5.6, 4.3 if show_legend else 3.5))
        axes = [fig.add_subplot(1, 1, 1)]
        plot_one(axes[0], setups[0], annotate_outlier=True)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=top)
    elif n == 2:
        w, h = (9.6, 4.7) if show_legend else (10.2, 3.45)
        fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
        for i, setup in enumerate(setups):
            plot_one(axes[i], setup, annotate_outlier=(i == 0))
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=top, wspace=0.14)
    else:
        w, h = (10.8, 5.0) if show_legend else (12.8, 3.45)
        fig, axes = plt.subplots(1, 3, figsize=(w, h), sharey=True)
        for i, setup in enumerate(setups):
            plot_one(axes[i], setup, annotate_outlier=(i == 0))
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.11, top=top, wspace=0.12)
    for ax in axes:
        ax.set_xlabel("mode index")
    axes[0].set_ylabel(r"spectral radius $\rho$")
    if show_legend:
        add_shared_legend(fig)
    fig.savefig(os.path.join(sty.OUTDIR, out_name))
    plt.close(fig)
