"""Paper figure: five nonlinear-membrane displacement snapshots (MIISO4).

Layout and styling copied from fig_miiso_surfaces.py (Fig.6).
Reads test/membrance/out_video/export → figures/Fig18.png (+ paper/figures).
"""

from __future__ import annotations

import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from fig_style import OUTDIR, apply_style

EXP = Path(__file__).resolve().parents[1] / "test" / "membrance" / "out_video" / "export"
PAPER_FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"


def read_f64(path: Path):
    with open(path, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{nd}q", f.read(8 * nd))
        return np.frombuffer(f.read(), dtype="<f8").reshape(shape, order="F")


def pick_indices(times, U):
    """Five frames evenly spaced over the saved window (corner-bump IC)."""
    n = len(times)
    targets = np.linspace(times[0], times[-1], 5)
    idxs = []
    used = set()
    for t in targets:
        i = int(np.argmin(np.abs(times - t)))
        if i in used:
            for d in range(1, n):
                found = False
                for cand in (i - d, i + d):
                    if 0 <= cand < n and cand not in used:
                        i = cand
                        found = True
                        break
                if found:
                    break
        used.add(i)
        idxs.append(i)
    return idxs


def _draw_u(ax, X, Y, Uk, umax):
    """Same pane / surface / tick recipe as fig_miiso_surfaces._draw_surface."""
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.78, 0.78, 0.78, 1.0))
        axis.pane.set_linewidth(0.55)
        axis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.45
    ax.plot_surface(
        X, Y, Uk, cmap="coolwarm", vmin=-umax, vmax=umax,
        rstride=1, cstride=1, linewidth=0.0, antialiased=True,
        shade=True, edgecolor="none",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(-umax, umax)
    ax.set_xlabel(r"$x$", labelpad=10, fontsize=16)
    ax.set_ylabel(r"$y$", labelpad=10, fontsize=16)
    ax.set_zlabel(r"$u$", labelpad=8, fontsize=16)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    zticks = np.linspace(-umax, umax, 3)
    ax.set_zticks(zticks)
    ax.tick_params(labelsize=13, pad=2)
    ax.view_init(elev=22, azim=-60)
    # Square floor. z visual weight matched to Fig.6 (not data-true pancake).
    try:
        ax.set_box_aspect((1.0, 1.0, 0.62))
    except Exception:
        pass


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    info = {}
    for line in open(EXP / "info.txt"):
        key, value = line.strip().split("=", 1)
        info[key] = value
    times = read_f64(EXP / "times.bin")
    X = read_f64(EXP / "X.bin")
    Y = read_f64(EXP / "Y.bin")
    U = read_f64(EXP / "U.bin")
    umax = float(info["umax"])
    idxs = pick_indices(times, U)
    print("selected frames:")
    for k, i in enumerate(idxs):
        ic = max(1, U.shape[1] // 16)
        jc = max(1, U.shape[2] // 16)
        print(f"  ({chr(ord('a')+k)}) i={i}  t={times[i]:.4f}  "
              f"u_corner={U[i, ic, jc]:+.4f}")

    if X.shape[0] > 64:
        step = 2
    else:
        step = 1
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]

    path = OUTDIR / "Fig18.png"
    fig = plt.figure(figsize=(15.5, 9.0))
    gs_top = fig.add_gridspec(
        1, 3, left=0.04, right=0.96, top=0.93, bottom=0.52, wspace=0.22,
    )
    gs_bot = fig.add_gridspec(
        1, 2, left=0.20, right=0.80, top=0.46, bottom=0.14, wspace=0.28,
    )
    slots = [gs_top[0, 0], gs_top[0, 1], gs_top[0, 2], gs_bot[0, 0], gs_bot[0, 1]]
    for k, i in enumerate(idxs):
        ax = fig.add_subplot(slots[k], projection="3d")
        _draw_u(ax, Xs, Ys, U[i, ::step, ::step], umax)
        ax.set_title(
            rf"$({chr(ord('a')+k)})$  $t={times[i]:.3f}$",
            fontsize=18, pad=6,
        )
        if k > 0:
            ax.set_zlabel("")

    mappable = cm.ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=-umax, vmax=umax))
    mappable.set_array([])
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.028])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(r"$u$", fontsize=16, labelpad=6)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_ticks(np.linspace(-umax, umax, 5))

    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", path)
    print("wrote", path.with_suffix(".pdf"))
    # Do not overwrite paper/figures until explicitly requested.
    # (PAPER_FIG / "Fig18.png").write_bytes(path.read_bytes())
    # print("copied", PAPER_FIG / "Fig18.png")
    print("skipped paper copy (keep paper/figures/Fig18.png unchanged)")


if __name__ == "__main__":
    main()
