#generates the 3d figures of the amplication plots as
#shown in figure 6 in the paper. DISCLAIMER:
# this file is mostly AI generated, because it 
# is almost all mechanical plotting, and whatever
# could be messed up was verified.
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

p, n_el, steps = 4, 20, 2.0
M, K, n1 = assemble_1d(n_el, p, 1)
free = np.arange(1, n1 - 1)
lambda_ = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True)
omega = np.sqrt(np.maximum(lambda_, 0.0))
N = len(omega)
setup = ("bar", "fixed", 4)
k = int(min(max(miiso2.outlier_count(setup), 1), N - 1))
i_out = N - k
h = (2.0 * np.pi / steps) / omega[i_out - 1]
omega_c = h * omega[i_out]
omega0 = 0.99 * omega_c

pts = 32
re = np.linspace(-6.0, 0.0, int(round(6.0 * pts)))
im = np.linspace(-6.0, 6.0, int(round(12.0 * pts)))
X, Y = np.meshgrid(re, im)
Z = (X + 1j * Y).ravel()
order = np.argsort(np.abs(Z))

methods = [
    ("MIISO2", miiso2.R_blend),
    ("MIISO3", miiso3.R_blend),
    ("MIISO4", miiso4.R_blend),
    ("MIISO5", miiso5.R_blend),
    ("MIISO6", miiso6.R_blend),
]

fig = plt.figure(figsize=(15.5, 9.0))
gs_top = fig.add_gridspec(1, 3, left=0.04, right=0.96, top=0.93, bottom=0.52, wspace=0.22)
gs_bot = fig.add_gridspec(1, 2, left=0.20, right=0.80, top=0.46, bottom=0.14, wspace=0.28)
slots = [gs_top[0, 0], gs_top[0, 1], gs_top[0, 2], gs_bot[0, 0], gs_bot[0, 1]]

for i, (label, R_blend) in enumerate(methods):
    Mag = np.empty(Z.size, float)
    Mag[order] = np.abs(R_blend(Z[order], omega0, omega_c))
    Mag = np.nan_to_num(Mag.reshape(X.shape), nan=np.inf, posinf=np.inf)
    ax = fig.add_subplot(slots[i], projection="3d")
    Xs, Ys = X[::2, ::2], Y[::2, ::2]
    Ms = np.clip(Mag[::2, ::2], 0.0, 1.0)
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.78, 0.78, 0.78, 1.0))
        axis.pane.set_linewidth(0.55)
        axis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.45
    ax.plot_surface(
        Xs, Ys, Ms, cmap="magma", vmin=0.0, vmax=1.0,
        rstride=1, cstride=1, linewidth=0.0, antialiased=True,
        shade=True, edgecolor="none",
    )
    ax.plot_wireframe(
        Xs[::10, ::10], Ys[::10, ::10], np.ones_like(Ms[::10, ::10]),
        color="0.55", linewidth=0.25, alpha=0.22,
    )
    ax.set_xlabel(r"$\mathrm{Re}\,z$", labelpad=10, fontsize=16)
    ax.set_ylabel(r"$\mathrm{Im}\,z$", labelpad=10, fontsize=16)
    ax.set_zlabel(r"$|R(z)|$" if i == 0 else "", labelpad=8, fontsize=16)
    ax.set_xlim(float(Xs.min()), float(Xs.max()))
    ax.set_ylim(float(Ys.min()), float(Ys.max()))
    ax.set_zlim(0.0, 1.0)
    ax.set_zticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=13, pad=2)
    ax.view_init(elev=22, azim=135)
    ax.set_box_aspect((1.0, 1.25, 0.62))
    ax.set_title(r"(%s) %s" % (chr(ord("a") + i), label), fontsize=18, pad=6)

mappable = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0.0, 1.0))
mappable.set_array([])
cax = fig.add_axes([0.30, 0.045, 0.40, 0.028])
cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
cbar.set_label(r"$|R(z)|$", fontsize=16, labelpad=6)
cbar.ax.tick_params(labelsize=13)

path = os.path.join(sty.OUTDIR, "Fig6.png")
fig.savefig(path)
plt.close(fig)
paper = os.path.join(os.path.dirname(sty.HERE), "paper", "figures", "Fig6.png")
open(paper, "wb").write(open(path, "rb").read())
