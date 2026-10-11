"""Shared paper style for scripts/fig_*.py (Times New Roman, Okabe–Ito)."""

from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parent
CODES = ROOT / "codes"
OUTDIR = ROOT / "figures"

# Match test/membrance/plot.py
COLORS = {
    "MIISO2": "#0072B2",
    "MIISO3": "#E69F00",
    "MIISO4": "#009E73",
    "MIISO5": "#CC79A7",
    "MIISO6": "#D55E00",
    "gen-α": "#000000",
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
            "legend.fontsize": 9,
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
        }
    )


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.15")
    ax.tick_params(which="both", color="0.15", length=3.5)
