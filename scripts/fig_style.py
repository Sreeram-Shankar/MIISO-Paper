#shared colors and stylig for the plots
import os
import matplotlib as mpl

HERE = os.path.dirname(__file__)
CODES = os.path.join(HERE, "codes")
OUTDIR = os.path.join(HERE, "figures")

COLORS = {
    "MIISO2": "#0072B2",
    "MIISO3": "#E69F00",
    "MIISO4": "#009E73",
    "MIISO5": "#CC79A7",
    "MIISO6": "#D55E00",
    "gen-alpha": "#000000",
}

def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })

def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.15")
    ax.tick_params(which="both", color="0.15", length=3.5)
