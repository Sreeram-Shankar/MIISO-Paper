#!/usr/bin/env python3
"""Paper-style plots for 2D membrane cutoff tracking.

  gap / leakage (small mesh, A∈{0.20,0.30,0.35})
  exact vs Rayleigh ω_c (finer mesh, A=0.2)

  python3 test/cutoff_tracking/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "out"

AMP_STYLE = {
    0.20: ("#0072B2", "-", r"$A=0.20$"),
    0.30: ("#E69F00", "--", r"$A=0.30$"),
    0.35: ("#D55E00", "-", r"$A=0.35$"),
}


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.03,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.15")
    ax.tick_params(which="both", color="0.15", length=3.5)


def plot_gap_leakage():
    data = json.loads((OUT / "outlier_tracking_validity.json").read_text())
    k = int(data["meta"]["k"])
    amps = []
    for a in data["meta"]["amplitudes"]:
        amps.append(float(a))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.55))
    handles = []
    ymax = 0.0
    for A in amps:
        color, ls, label = AMP_STYLE.get(A, ("#333333", "-", rf"$A={A:g}$"))
        hist = data["by_A"][f"{A:g}"]
        t = np.asarray(hist["t"], dtype=float)
        gap = np.asarray(hist["gap_ratio"], dtype=float)
        leak = np.asarray(hist["leakage"], dtype=float)
        axes[0].plot(t, gap, color=color, ls=ls, lw=1.9)
        axes[1].plot(t, leak, color=color, ls=ls, lw=1.9)
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=2.0, label=label))
        ymax = max(ymax, float(leak.max()))

    T_meta = data["meta"]["T"]
    if isinstance(T_meta, dict):
        t_values = []
        for v in T_meta.values():
            t_values.append(float(v))
        tmax = max(t_values)
    else:
        tmax = float(T_meta)

    axes[0].axhline(1.0, color="0.45", ls=":", lw=0.9)
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"gap ratio $\omega_{N-k+1}/\omega_{N-k}$")
    axes[0].set_xlim(0.0, tmax)
    axes[0].set_ylim(0.95, None)

    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"subspace leakage $\sum_j\sin^2\theta_j$")
    axes[1].set_xlim(0.0, tmax)
    axes[1].set_ylim(-0.03, min(k, ymax) * 1.12)

    for ax in axes:
        style_axes(ax)

    fig.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.02), columnspacing=1.8, handlelength=2.6,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.86, wspace=0.28)
    for ext in (".png", ".pdf"):
        path = OUT / f"cutoff_failure_gap_leakage{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


def plot_rayleigh_A02():
    """A=0.2: exact ω_c vs Rayleigh-tracked ω_c, plus relative error."""
    data = json.loads((OUT / "rayleigh_error_A02.json").read_text())
    t = np.asarray(data["t"], dtype=float)
    we = np.asarray(data["omega_c_exact"], dtype=float)
    wr = np.asarray(data["omega_c_rayleigh"], dtype=float)
    rel = np.maximum(np.asarray(data["rel_err"], dtype=float), 1e-16)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.55))
    axes[0].plot(t, we, color="#0072B2", ls="-", lw=1.9, label=r"exact $\omega_c$")
    axes[0].plot(t, wr, color="#D55E00", ls="--", lw=1.9, label=r"Rayleigh $\omega_c$")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$\omega_c$")
    axes[0].set_xlim(0.0, t[-1])
    axes[0].legend(loc="best", handlelength=2.4)

    axes[1].semilogy(t, rel, color="#D55E00", ls="-", lw=1.9)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$|\omega_c^{\mathrm{Rayleigh}}-\omega_c|/\omega_c$")
    axes[1].set_xlim(0.0, t[-1])

    Tp = data.get("T_period")
    if Tp is not None:
        Tp = float(Tp)
        for ax in axes:
            ax.axvline(Tp, color="0.55", ls=":", lw=0.8)
            ax.axvline(2 * Tp, color="0.55", ls=":", lw=0.8)

    for ax in axes:
        style_axes(ax)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.92, wspace=0.30)
    for ext in (".png", ".pdf"):
        path = OUT / f"rayleigh_error_A02{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


def main():
    apply_style()
    plot_gap_leakage()
    plot_rayleigh_A02()


if __name__ == "__main__":
    main()
