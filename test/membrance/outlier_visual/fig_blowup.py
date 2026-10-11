#!/usr/bin/env python3
"""Dramatic MIISO4 vs exprb43 figure: full-field u with independent scales.

Shared color limits hide MIISO when exprb has blown up. Separate |u|_∞
scales make the unphysical growth obvious.

  python3 test/membrance/outlier_visual/fig_blowup.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = Path(__file__).resolve().parent / "out"
EXP = OUT / "export"
PAPER = Path(__file__).resolve().parents[3] / "paper" / "figures"


def read_f64(path: Path):
    with open(path, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{nd}q", f.read(8 * nd))
        return np.frombuffer(f.read(), dtype="<f8").reshape(shape, order="F")


def draw_u(ax, X, Y, Uk, umax, title):
    ax.plot_surface(
        X,
        Y,
        Uk,
        cmap="coolwarm",
        vmin=-umax,
        vmax=umax,
        linewidth=0,
        edgecolor="none",
        shade=True,
        antialiased=True,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-umax, umax)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$u$")
    ax.view_init(elev=24, azim=-60)
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except Exception:
        pass
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_title(title, fontsize=11)


def main():
    data = json.loads((OUT / "frames.json").read_text())
    X = read_f64(EXP / "X.bin")
    Y = read_f64(EXP / "Y.bin")
    Um = read_f64(EXP / "U_miiso.bin")
    Ue = read_f64(EXP / "U_exprb.bin")
    tm = read_f64(EXP / "times_miiso.bin")
    te = read_f64(EXP / "times_exprb.bin")

    # Last common-ish times (each method's final saved frame).
    im = Um.shape[0] - 1
    ie = Ue.shape[0] - 1
    UmT = Um[im]
    UeT = Ue[ie]
    t_m = float(tm[im])
    t_e = float(te[ie])
    um_m = float(np.max(np.abs(UmT)))
    um_e = float(np.max(np.abs(UeT)))
    Eo_m = float(data["E_out_miiso"][im])
    Eo_e = float(data["E_out_exprb"][ie])
    Er_m = float(data["E_res_miiso"][im])
    Er_e = float(data["E_res_exprb"][ie])

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )

    # Independent z-scales: the point is the amplitude gap.
    fig = plt.figure(figsize=(10.2, 4.4))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_u(
        ax0,
        X,
        Y,
        UmT,
        max(um_m, 1e-12),
        rf"MIISO4  $t={t_m:.3g}$"
        "\n"
        + rf"$|u|_\infty={um_m:.3g}$,  $E_{{\mathrm{{out}}}}={Eo_m:.2e}$,  $E_{{\mathrm{{res}}}}={Er_m:.2e}$",
    )
    draw_u(
        ax1,
        X,
        Y,
        UeT,
        max(um_e, 1e-12),
        rf"exprb43  $t={t_e:.3g}$"
        "\n"
        + rf"$|u|_\infty={um_e:.3g}$,  $E_{{\mathrm{{out}}}}={Eo_e:.2e}$,  $E_{{\mathrm{{res}}}}={Er_e:.2e}$",
    )
    fig.suptitle(
        rf"Same mesh, step, and corner-bump IC "
        rf"({data['nel']}$\times${data['nel']}, $k={data['k']}$, $h={data['h']:.4g}$)",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUT / "blowup_u_final.png"
    fig.savefig(path)
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / "Fig_corner_blowup.png")
    print(f"wrote {path}")
    print(f"wrote {PAPER / 'Fig_corner_blowup.png'}")
    print(f"|u|_∞  MIISO4={um_m:.6g}  exprb43={um_e:.6g}  ratio={um_e / max(um_m, 1e-30):.3g}")
    plt.close(fig)

    # Companion: band energies (quantitative blow-up).
    fig2, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for ax, key, ylabel, title, logy in (
        (axes[0], "E_out", r"$E_{\mathrm{out}}$", "Outlier band", True),
        (axes[1], "E_res", r"$E_{\mathrm{res}}$", "Resolved band", False),
    ):
        for prefix, label, style in (("miiso", "MIISO4", "o-"), ("exprb", "exprb43", "s-")):
            t = np.asarray(data[f"times_{prefix}"], float)
            y = np.asarray(data[f"{key}_{prefix}"], float)
            if t.size == 0:
                continue
            if logy:
                ax.semilogy(t, y, style, ms=4, label=label)
            else:
                ax.plot(t, y, style, ms=4, label=label)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    path2 = OUT / "blowup_bands.png"
    fig2.savefig(path2)
    fig2.savefig(PAPER / "Fig_corner_blowup_bands.png")
    print(f"wrote {path2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
