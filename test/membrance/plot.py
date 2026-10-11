#!/usr/bin/env python3
"""Paper figures for the membrane suite (Times New Roman, Okabe–Ito colors).

  python3 test/membrance/plot.py              # all three
  python3 test/membrance/plot.py cost         # cost vs error
  python3 test/membrance/plot.py scaling      # wall vs N
  python3 test/membrance/plot.py convergence  # error vs h
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
ERR_FLOOR = 1e-11  # drop FP-floor points left of MIISO5/6 meaningful cutoff

COLORS = {
    "MIISO2": "#0072B2",
    "MIISO3": "#E69F00",
    "MIISO4": "#009E73",
    "MIISO5": "#CC79A7",
    "MIISO6": "#D55E00",
    "gen-α": "#000000",
}
MARKERS = {
    "MIISO2": "o",
    "MIISO3": "s",
    "MIISO4": "D",
    "MIISO5": "^",
    "MIISO6": "v",
    "gen-α": "x",
}


def _first_of_pair(pair):
    return pair[0]


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
            "axes.grid": False,
        }
    )


def style_for(name: str):
    return {
        "color": COLORS.get(name, "#333333"),
        "marker": MARKERS.get(name, "o"),
        "label": name,
    }


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    print(f"wrote {path}")
    print(f"wrote {path.with_suffix('.pdf')}")
    plt.close(fig)


def plot_cost():
    data = json.loads((OUT / "cost_vs_error_nel20.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    xmin = np.inf
    ymin = np.inf
    xmax = 0.0
    ymax = 0.0
    for name in data["order"]:
        pts = []
        for r in data["methods"][name]:
            if r["err"] >= ERR_FLOOR:
                pts.append((r["err"], r["wall"]))
        pts.sort(key=_first_of_pair, reverse=True)
        if not pts:
            continue
        err_list = []
        wall_list = []
        for p in pts:
            err_list.append(p[0])
            wall_list.append(p[1])
        err = np.array(err_list)
        wall = np.array(wall_list)
        gen = name.startswith("gen")
        if gen:
            linewidth = 2.2
            markersize = 7
            markeredgewidth = 1.1
        else:
            linewidth = 1.6
            markersize = 5.5
            markeredgewidth = 0.8
        ax.loglog(
            err, wall, linestyle="-", linewidth=linewidth,
            markersize=markersize, markeredgewidth=markeredgewidth,
            **style_for(name),
        )
        xmin, xmax = min(xmin, err.min()), max(xmax, err.max())
        ymin, ymax = min(ymin, wall.min()), max(ymax, wall.max())
    ax.set_xlabel(r"relative $L^2$ error")
    ax.set_ylabel("wall time [s]")
    ax.set_xlim(xmin / 1.8, xmax * 1.8)
    ax.set_ylim(ymin / 1.4, ymax * 1.4)
    ax.legend(loc="lower left", handlelength=1.6)
    ax.set_title("Cost vs accuracy")
    save(fig, OUT / "cost_vs_error_nel20.png")


def plot_scaling():
    data = json.loads((OUT / "scaling_cost.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    Nmax = 0.0
    Wmin, Wmax = np.inf, 0.0
    for name in data["order"]:
        rows = data["methods"][name]
        N_list = []
        wall_list = []
        for r in rows:
            N_list.append(r["N"])
            wall_list.append(r["wall"])
        N = np.array(N_list, dtype=float)
        wall = np.array(wall_list, dtype=float)
        gen = name.startswith("gen")
        if gen:
            linewidth = 2.4
            markersize = 7
            markeredgewidth = 1.1
        else:
            linewidth = 1.6
            markersize = 5.5
            markeredgewidth = 0.8
        ax.semilogy(
            N, wall, linestyle="-", linewidth=linewidth,
            markersize=markersize, markeredgewidth=markeredgewidth,
            **style_for(name),
        )
        Nmax = max(Nmax, float(N.max()))
        Wmin, Wmax = min(Wmin, float(wall.min())), max(Wmax, float(wall.max()))
    ax.set_xlabel(r"$N$ (degrees of freedom)")
    ax.set_ylabel("wall time [s]")
    ax.set_xlim(0.0, Nmax * 1.05)
    ax.set_ylim(Wmin / 1.4, Wmax * 1.4)
    ax.legend(loc="lower right", handlelength=1.6)
    ax.set_title("Scaling cost")
    save(fig, OUT / "scaling_cost.png")


def plot_convergence():
    data = json.loads((OUT / "convergence_nel6.json").read_text())
    hs = np.array(data["hs"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name in data["order"]:
        err = np.array(data["methods"][name]["err"], dtype=float)
        ax.loglog(hs, err, linestyle="-", linewidth=1.6, markersize=5.5,
                  markeredgewidth=0.8, **style_for(name))
    h_ref = np.array([hs[0], hs[-1]])
    for p, y0, lab in (
        (2, 2.0e0, r"$\mathcal{O}(h^2)$"),
        (4, 8.0e-2, r"$\mathcal{O}(h^4)$"),
        (6, 2.5e-3, r"$\mathcal{O}(h^6)$"),
    ):
        ax.loglog(h_ref, y0 * (h_ref / h_ref[0]) ** p,
                  color="0.55", linestyle="--", linewidth=0.9, label=lab)
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.legend(loc="lower right", ncol=2, handlelength=1.6)
    ax.set_title("Convergence")
    save(fig, OUT / "convergence_nel6.png")


def main(argv=None):
    apply_style()
    if argv:
        which = argv
    else:
        which = sys.argv[1:]
    if not which:
        which = ["cost", "scaling", "convergence"]
    for w in which:
        if w == "cost":
            plot_cost()
        elif w == "scaling":
            plot_scaling()
        elif w == "convergence":
            plot_convergence()
        else:
            raise KeyError(w)


if __name__ == "__main__":
    main()
