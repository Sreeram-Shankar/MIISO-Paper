"""Paper figures. MIISO4 vs exprb43 at the blow-up time (corner-bump IC).

Two figures (document order):
  Fig16.png  3D displacement u
  Fig17.png  outlier heatmaps u_out

  python3 scripts/fig_corner_blowup.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from fig_style import OUTDIR, apply_style

EXP = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "membrance"
    / "outlier_visual"
    / "out"
    / "export"
)
META = EXP.parent / "frames.json"
PAPER_FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
LOCAL_OUT = META.parent
T_TARGET = 1.4


def read_f64(path: Path):
    with open(path, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{nd}q", f.read(8 * nd))
        return np.frombuffer(f.read(), dtype="<f8").reshape(shape, order="F")


def nearest_index(times, t):
    return int(np.argmin(np.abs(np.asarray(times, float) - t)))


def nice_lim(m):
    m = float(max(m, 1e-16))
    exp = int(np.floor(np.log10(m)))
    base = 10.0 ** exp
    for f in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if f * base >= m:
            return f * base
    return 10.0 * base


def fmt_tick(x, _pos=None):
    ax = abs(x)
    if ax == 0.0:
        return r"$0$"
    if ax >= 10:
        return rf"${x:.0f}$"
    if ax >= 1:
        return rf"${x:.1f}$"
    if ax >= 0.01:
        return rf"${x:.2f}$"
    return rf"${x:.0e}$".replace("e-0", "e-").replace("e+0", "e+")


def draw_u(ax, X, Y, Uk, color_max, z_max=None):
    if z_max is None:
        z_max = color_max
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.78, 0.78, 0.78, 1.0))
        axis.pane.set_linewidth(0.55)
        axis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.45
    ax.plot_surface(
        X,
        Y,
        Uk,
        cmap="coolwarm",
        vmin=-color_max,
        vmax=color_max,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        edgecolor="none",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(-z_max, z_max)
    ax.set_xlabel(r"$x$", labelpad=12, fontsize=20)
    ax.set_ylabel(r"$y$", labelpad=12, fontsize=20)
    ax.set_zlabel(r"$u$", labelpad=10, fontsize=20)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks(np.linspace(-z_max, z_max, 3))
    ax.zaxis.set_major_formatter(FuncFormatter(fmt_tick))
    ax.tick_params(labelsize=16, pad=3)
    ax.view_init(elev=22, azim=-60)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.62))
    except Exception:
        pass


def write_all(fig, stem):
    for path in (OUTDIR / f"{stem}.png", PAPER_FIG / f"{stem}.png", LOCAL_OUT / f"{stem}.png"):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    fig.savefig(OUTDIR / f"{stem}.pdf")
    print("wrote", OUTDIR / f"{stem}.png")
    print("wrote", OUTDIR / f"{stem}.pdf")


def fig_u(Xs, Ys, UmT, UeT, um_m, ue_cap, ue_zmax):
    fig = plt.figure(figsize=(14.5, 6.2))
    ax0 = fig.add_axes([0.03, 0.18, 0.44, 0.78], projection="3d")
    ax1 = fig.add_axes([0.52, 0.18, 0.44, 0.78], projection="3d")
    draw_u(ax0, Xs, Ys, UmT, um_m)
    draw_u(ax1, Xs, Ys, UeT, ue_cap, ue_zmax)
    ax0.set_title(r"$(a)$", fontsize=22, pad=8)
    ax1.set_title(r"$(b)$", fontsize=22, pad=8)

    for i, vmax in enumerate((um_m, ue_cap)):
        cax = fig.add_axes([0.08 + i * 0.49, 0.08, 0.34, 0.035])
        mappable = cm.ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=-vmax, vmax=vmax))
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.set_label(r"$u$", fontsize=18, labelpad=6)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_ticks(np.linspace(-vmax, vmax, 5))
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(fmt_tick))

    write_all(fig, "Fig16")
    plt.close(fig)


def fig_uout(UomT, UoeT, uo_vmax):
    norm = SymLogNorm(linthresh=1e-3, vmin=-uo_vmax, vmax=uo_vmax)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.9))
    for ax, U, letter in (
        (axes[0], UomT, "a"),
        (axes[1], UoeT, "b"),
    ):
        im = ax.imshow(
            U,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            norm=norm,
            aspect="equal",
            interpolation="bilinear",
        )
        ax.set_xlabel(r"$x$", fontsize=20)
        ax.set_ylabel(r"$y$", fontsize=20)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["", r"$0.5$", r"$1.0$"])
        ax.tick_params(labelsize=16)
        ax.set_title(rf"$({letter})$", fontsize=22, pad=10)

    fig.subplots_adjust(bottom=0.24)
    cax = fig.add_axes([0.16, 0.1, 0.68, 0.04])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(r"$u_{\mathrm{out}}$ (log)", fontsize=18, labelpad=6)
    cbar.ax.tick_params(labelsize=15)
    ticks = [-10.0, -1.0, -0.1, -0.01, -1e-3, 1e-3, 0.01, 0.1, 1.0, 10.0]
    ticks = [t for t in ticks if -uo_vmax < t < uo_vmax]
    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(fmt_tick))

    write_all(fig, "Fig17")
    plt.close(fig)


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    data = json.loads(META.read_text())
    X = read_f64(EXP / "X.bin")
    Y = read_f64(EXP / "Y.bin")
    Um = read_f64(EXP / "U_miiso.bin")
    Ue = read_f64(EXP / "U_exprb.bin")
    Uom = read_f64(EXP / "Uout_miiso.bin")
    Uoe = read_f64(EXP / "Uout_exprb.bin")
    tm = read_f64(EXP / "times_miiso.bin")
    te = read_f64(EXP / "times_exprb.bin")

    im = nearest_index(tm, T_TARGET)
    ie = nearest_index(te, T_TARGET)
    print(f"frame t={tm[im]:.6g}  (exprb t={te[ie]:.6g})")
    print(
        f"E_out  MIISO4={data['E_out_miiso'][im]:.4e}  "
        f"exprb43={data['E_out_exprb'][ie]:.4e}"
    )

    if X.shape[0] > 64:
        step = 2
    else:
        step = 1
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    UmT = Um[im, ::step, ::step]
    UeT = Ue[ie, ::step, ::step]
    # Full fine grid for heatmaps (no downsample).
    UomT = Uom[im]
    UoeT = Uoe[ie]

    um_m = nice_lim(np.max(np.abs(UmT)))
    ue_cap = 10.0
    ue_zmax = nice_lim(np.max(np.abs(UeT)))
    uo_vmax = nice_lim(np.max(np.abs(UoeT)))
    print(f"|u| MIISO4 scale ±{um_m:g}  exprb43 color cap ±{ue_cap:g} on z ±{ue_zmax:g}")
    print(f"|u_out| shared log scale ±{uo_vmax:g}  (exprb max {np.abs(UoeT).max():.6g})")

    fig_u(Xs, Ys, UmT, UeT, um_m, ue_cap, ue_zmax)
    fig_uout(UomT, UoeT, uo_vmax)
    for src, dst in (("Fig16", "Figure8"), ("Fig17", "Figure9")):
        (PAPER_FIG / f"{dst}.png").write_bytes((OUTDIR / f"{src}.png").read_bytes())
        (PAPER_FIG / f"{dst}.pdf").write_bytes((OUTDIR / f"{src}.pdf").read_bytes())
        print("copied", src, "->", dst, "in paper/figures")


if __name__ == "__main__":
    main()
