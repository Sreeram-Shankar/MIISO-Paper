#!/usr/bin/env python3
"""MIISO4 vs exprb43: u_out, E_out, E_res."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"


def series(data, prefix):
    tc = data.get(f"tcrash_{prefix}", None)
    if tc is None:
        tcrash = float("nan")
    else:
        tcrash = float(tc)
    return (
        np.asarray(data[f"times_{prefix}"], float),
        np.asarray(data[f"E_out_{prefix}"], float),
        np.asarray(data[f"E_res_{prefix}"], float),
        bool(data[f"crash_{prefix}"]),
        tcrash,
    )


def main():
    data = json.loads((OUT / "frames.json").read_text())
    m1 = int(data["m1"])
    methods = (
        ("miiso", "MIISO4", "o-"),
        ("exprb", "exprb43", "s-"),
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for ax, kind, ylabel, title, logy in (
        (axes[0], "E_out", r"$E_{\mathrm{out}}$", "Outlier band", True),
        (axes[1], "E_res", r"$E_{\mathrm{res}}$", "Resolved band", False),
    ):
        for prefix, label, style in methods:
            t, Eo, Er, crashed, tcrash = series(data, prefix)
            if kind == "E_out":
                y = Eo
            else:
                y = Er
            if t.size == 0:
                continue
            if crashed:
                curve_label = label + " (crash)"
            else:
                curve_label = label
            if logy:
                ax.semilogy(t, y, style, ms=4, label=curve_label)
            else:
                ax.plot(t, y, style, ms=4, label=curve_label)
            if crashed and np.isfinite(tcrash):
                ax.axvline(tcrash, color="C3", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    path = OUT / "band_energies.png"
    fig.savefig(path)
    print(f"wrote {path}")
    plt.close(fig)

    for kind, ylabel, title, fname, logy in (
        ("E_out", r"$E_{\mathrm{out}}$", "Outlier band energy", "E_out_history.png", True),
        ("E_res", r"$E_{\mathrm{res}}$", "Resolved band energy", "E_res_history.png", False),
    ):
        fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
        for prefix, label, style in methods:
            t, Eo, Er, crashed, _ = series(data, prefix)
            if kind == "E_out":
                y = Eo
            else:
                y = Er
            if t.size == 0:
                continue
            if crashed:
                curve_label = label + " (crash)"
            else:
                curve_label = label
            if logy:
                ax.semilogy(t, y, style, ms=4, label=curve_label)
            else:
                ax.plot(t, y, style, ms=4, label=curve_label)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        path = OUT / fname
        fig.savefig(path)
        print(f"wrote {path}")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.8), constrained_layout=True)
    vmax = 1e-12
    packed = []
    for prefix, label, _ in methods:
        uouts = data[f"uout_{prefix}"]
        times = np.asarray(data[f"times_{prefix}"], float)
        if len(uouts) == 0:
            packed.append((label, [], [], [], prefix))
            continue
        idxs = [0, len(uouts) // 2, len(uouts) - 1]
        grids = []
        for i in idxs:
            grids.append(np.asarray(uouts[i], float).reshape(m1, m1))
        eouts = []
        for i in idxs:
            eouts.append(data[f"E_out_{prefix}"][i])
        for g in grids:
            vmax = max(vmax, np.max(np.abs(g)))
        packed.append((label, times[idxs], grids, eouts, prefix))

    im = None
    for row, (label, ts, grids, eouts, prefix) in enumerate(packed):
        for j in range(3):
            ax = axes[row, j]
            if j >= len(grids):
                ax.set_axis_off()
                continue
            im = ax.imshow(
                grids[j].T,
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                aspect="equal",
            )
            crashed = bool(data[f"crash_{prefix}"])
            if crashed and j == len(grids) - 1:
                lab = label + " CRASHED"
            else:
                lab = label
            ax.set_title(rf"{lab}  $u_{{\mathrm{{out}}}}$  $t={ts[j]:.3g}$", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.02,
                0.96,
                rf"$E_{{\mathrm{{out}}}}={eouts[j]:.2e}$",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="0.15",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
            )
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label=r"$u_{\mathrm{out}}$")
    ic = data.get("ic", "corner_bump")
    fig.suptitle(
        rf"Outlier projection, {ic} IC, $\alpha={data['alpha']}$ "
        rf"({data['nel']}$\times${data['nel']}, $k={data['k']}$, $h={data['h']:.4g}$)",
        fontsize=11,
    )
    path = OUT / "outlier_visual.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    print(f"wrote {path}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    for ax, (label, ts, grids, eouts, _) in zip(axes2, packed):
        if len(grids) < 2:
            ax.set_axis_off()
            continue
        ax.imshow(
            grids[1].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(rf"{label}  $E_{{\mathrm{{out}}}}={eouts[1]:.2e}$")
        ax.set_xticks([])
        ax.set_yticks([])
    if packed[0][1] is not None and len(packed[0][1]) > 1:
        fig2.suptitle(rf"$u_{{\mathrm{{out}}}}$ at $t={packed[0][1][1]:.3g}$", fontsize=11)
    still = OUT / "uout_mid.png"
    fig2.savefig(still, dpi=200)
    print(f"wrote {still}")
    plt.close("all")


if __name__ == "__main__":
    main()
