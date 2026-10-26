# generates the resultsn from the bridge simulation 
# for figure 10 in the paper. DISCLAIMER:
# this file is mostly AI generated, because it 
# is almost all mechanical plotting, and whatever
# could be messed up was verified.
import os, struct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource, LogNorm, PowerNorm
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter, zoom

import fig_style as sty

sty.apply_style()
os.makedirs(sty.OUTDIR, exist_ok=True)

root = os.path.dirname(sty.HERE)
snap = os.path.join(root, "test", "bridge", "out_bridge_nel3266", "export")
paper_fig = os.path.join(root, "paper", "figures")

def read_f64(path):
    with open(path, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack("<%dq" % nd, f.read(8 * nd))
        return np.frombuffer(f.read(), dtype="<f8").reshape(shape, order="F")

def draw(ax, faces, geom, C, wires, outer, xlim, wlim, hlim, aspect,
         norm, cmap, ls, title="", show_ylabels=True):
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))
    rank = {}
    for n in faces:
        if n.startswith("e"):
            rank[n] = 2
        elif n in outer:
            rank[n] = 1
        else:
            rank[n] = 0
    for f in sorted(faces, key=rank.get):
        Xp, Wp, Hp = geom[f]
        rgb = ls.shade_rgb(cmap(norm(C[f])), Hp, blend_mode="soft")
        alpha = 0.95 if (f.startswith("e") or f in outer) else 0.35
        ax.plot_surface(
            Xp, Wp, Hp, facecolors=rgb,
            linewidth=0, edgecolor="none", shade=False, antialiased=True,
            alpha=alpha,
        )
    for i in range(wires.shape[0]):
        ax.plot(
            wires[i, 0, :], wires[i, 2, :], wires[i, 1, :],
            color="0.18", lw=0.8, alpha=0.75, solid_capstyle="round",
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*wlim)
    ax.set_zlim(*hlim)
    ax.set_box_aspect(aspect)
    ax.view_init(elev=30, azim=-52)
    ax.set_xticks([-60, 0, 60])
    ax.set_yticks([-5, 0, 5])
    ax.set_zticks([0, 12, 25])
    ax.tick_params(labelsize=11, pad=0)
    ax.set_xlabel(r"$x$", fontsize=14, labelpad=2)
    ax.set_ylabel(r"$z$", fontsize=14, labelpad=2)
    ax.set_zlabel(r"$y$" if show_ylabels else "", fontsize=14, labelpad=0)
    ax.set_title(title, fontsize=15, pad=2)

info = {}
for line in open(os.path.join(snap, "info.txt")):
    key, value = line.strip().split("=", 1)
    info[key] = value
nF = int(info["nF"])
cmax = float(info["cmax"])
faces = [s.strip() for s in info["faces"].split(",")]
times = read_f64(os.path.join(snap, "times.bin"))
wires = read_f64(os.path.join(snap, "wires.bin"))

nh = sum(1 for f in faces if f.startswith("h"))
nw = sum(1 for f in faces if f.startswith("w"))
outer = {"h1", "h%d" % nh, "w1", "w%d" % nw, "e0", "e1"}

up = 3
geom = {}
Call = {}
for f in faces:
    X = read_f64(os.path.join(snap, "X_%s.bin" % f))
    Y = read_f64(os.path.join(snap, "Y_%s.bin" % f))
    Z = read_f64(os.path.join(snap, "Z_%s.bin" % f))
    geom[f] = (zoom(X, up, order=3), zoom(Z, up, order=3), zoom(Y, up, order=3))
    Call[f] = read_f64(os.path.join(snap, "C_%s.bin" % f))
mmax = np.max(np.stack([Call[f].reshape(nF, -1).max(axis=1) for f in faces]), axis=0)

nS = 12
t_eq = 1.5
sel = [0, int(np.argmin(np.abs(times - t_eq)))]
peaks = [k for k in range(1, nF - 1) if mmax[k] >= mmax[k - 1] and mmax[k] >= mmax[k + 1]]
for _neg, k in sorted([(-mmax[k], k) for k in peaks]):
    if len(sel) >= nS - 1:
        break
    if all(abs(times[k] - times[j]) >= 0.6 for j in sel):
        sel.append(k)
sel.append(nF - 1)
sel = sorted(sel)

snaps = []
peak_frac = []
for k in sel:
    snaps.append({f: gaussian_filter(zoom(Call[f][k], up, order=3), 1.5) for f in faces})
    peak_frac.append(mmax[k] / max(cmax, 1e-16))
times = times[sel]

edir = os.path.join(os.path.dirname(snap), "energy")
einfo = {}
for line in open(os.path.join(edir, "info.txt")):
    key, value = line.strip().split("=", 1)
    einfo[key] = value
psi_max = float(einfo["psi_max"])
efaces = [s.strip() for s in einfo["faces"].split(",")]
etimes = read_f64(os.path.join(edir, "times.bin"))
egeom = {}
Pall = {}
for f in efaces:
    X = read_f64(os.path.join(edir, "X_%s.bin" % f))
    Y = read_f64(os.path.join(edir, "Y_%s.bin" % f))
    Z = read_f64(os.path.join(edir, "Z_%s.bin" % f))
    egeom[f] = (zoom(X, up, order=3), zoom(Z, up, order=3), zoom(Y, up, order=3))
    Pall[f] = np.maximum(read_f64(os.path.join(edir, "P_%s.bin" % f)), 1e-30)
esnaps = []
for t in times:
    ke = int(np.argmin(np.abs(etimes - t)))
    esnaps.append({f: gaussian_filter(zoom(Pall[f][ke], up, order=3), 1.5) for f in efaces})
enorm = LogNorm(vmin=max(psi_max * 1e-4, 1e-30), vmax=psi_max)
ecmap = cm.inferno

xs = np.concatenate([geom[f][0].ravel() for f in faces] + [wires[:, 0, :].ravel()])
ws = np.concatenate([geom[f][1].ravel() for f in faces] + [wires[:, 2, :].ravel()])
hs = np.concatenate([geom[f][2].ravel() for f in faces] + [wires[:, 1, :].ravel()])
pad = 0.08
xlim = (xs.min() - pad * np.ptp(xs), xs.max() + pad * np.ptp(xs))
wlim = (ws.min() - pad * np.ptp(ws), ws.max() + pad * np.ptp(ws))
hlim = (hs.min() - pad * np.ptp(hs), hs.max() + pad * np.ptp(hs))
aspect = (np.ptp(xs), 3.0 * np.ptp(ws), 1.85 * np.ptp(hs))

norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=max(cmax, 1e-16))
cmap = cm.turbo
ls = LightSource(azdeg=315, altdeg=45)

for pair in range(6):
    fig = plt.figure(figsize=(13.5, 10.5))
    gs = fig.add_gridspec(
        2, 2, left=0.02, right=0.98, top=0.95, bottom=0.02,
        wspace=0.10, hspace=0.02,
    )
    for c in range(2):
        k = 2 * pair + c
        ax = fig.add_subplot(gs[0, c], projection="3d")
        title = r"$(%s)$  $t=%.2f\,\mathrm{s}$" % (chr(ord("a") + k), times[k])
        draw(ax, faces, geom, snaps[k], wires, outer, xlim, wlim, hlim, aspect,
             norm, cmap, ls, title=title, show_ylabels=(c == 0))
        ax = fig.add_subplot(gs[1, c], projection="3d")
        draw(ax, efaces, egeom, esnaps[k], wires, set(efaces), xlim, wlim, hlim, aspect,
             enorm, ecmap, ls, title=r"$\psi$", show_ylabels=(c == 0))
    stem = "Figure10%s" % chr(ord("a") + pair)
    for ext in (".png", ".pdf"):
        path = os.path.join(sty.OUTDIR, stem + ext)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        open(os.path.join(paper_fig, stem + ext), "wb").write(open(path, "rb").read())
    plt.close(fig)

fig = plt.figure(figsize=(12.0, 3.8))
gs = fig.add_gridspec(
    2, 1, height_ratios=[1.0, 0.8],
    left=0.06, right=0.98, top=0.92, bottom=0.22, hspace=0.9,
)
ax_rail = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[1, 0])
ax_rail.set_xlim(-0.3, 11.8)
ax_rail.set_ylim(0.0, 1.0)
ax_rail.set_yticks([])
for spine in ("top", "right", "left"):
    ax_rail.spines[spine].set_visible(False)
ax_rail.spines["bottom"].set_linewidth(1.1)
ax_rail.spines["bottom"].set_color("0.2")
ax_rail.add_patch(FancyBboxPatch(
    (0.0, 0.42), 11.5, 0.22, boxstyle="round,pad=0.02,rounding_size=0.08",
    linewidth=0, facecolor="0.92", transform=ax_rail.transData, clip_on=False, zorder=0,
))
ax_rail.axvspan(0.0, 1.0, ymin=0.40, ymax=0.68, color="#0072B2", alpha=0.16, lw=0, zorder=1)
ax_rail.axvspan(1.5, 11.5, ymin=0.40, ymax=0.68, color="#D55E00", alpha=0.12, lw=0, zorder=1)
for t, fr in zip(times, peak_frac):
    col = cmap(float(np.clip(fr, 0.0, 1.0)))
    ax_rail.plot(t, 0.53, "o", ms=10, color=col,
                 markeredgecolor="0.12", markeredgewidth=0.7, zorder=5)
ax_rail.set_xticks(list(times))
ax_rail.set_xticklabels(["%.2f" % t for t in times], fontsize=10, rotation=30, ha="right")
ax_rail.set_xlabel(r"$t$", fontsize=14, labelpad=2)
ax_rail.tick_params(axis="x", length=5, width=1.0, colors="0.15")
mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
cbar.set_label(r"$|u|$ [m]", fontsize=13, labelpad=2)
cbar.ax.tick_params(labelsize=11)
for ext in (".png", ".pdf"):
    path = os.path.join(sty.OUTDIR, "Figure10_key" + ext)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    open(os.path.join(paper_fig, "Figure10_key" + ext), "wb").write(open(path, "rb").read())
plt.close(fig)
