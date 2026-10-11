"""Paper Fig.10: continuum-arch |u| snapshots as separate panels.

Reads test/bridge/out_bridge_nel3266/export/ and picks 12 frames at
peak |u| over the run.
Writes figures/Figure10a–f.png (+ Figure10_key) and copies to paper/figures.
"""

from __future__ import annotations

import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource, LogNorm, PowerNorm
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter, zoom

from fig_style import OUTDIR, apply_style

SNAP = Path(__file__).resolve().parents[1] / "test" / "bridge" / "out_bridge_nel3266" / "export"
PAPER_FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"


def read_f64(path: Path):
    with open(path, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{nd}q", f.read(8 * nd))
        return np.frombuffer(f.read(), dtype="<f8").reshape(shape, order="F")


def _limits(geom, faces, wires):
    xs = []
    ws = []
    hs = []
    for f in faces:
        xs.append(geom[f][0].ravel())
        ws.append(geom[f][1].ravel())
        hs.append(geom[f][2].ravel())
    xs.append(wires[:, 0, :].ravel())
    ws.append(wires[:, 2, :].ravel())
    hs.append(wires[:, 1, :].ravel())
    xs = np.concatenate(xs)
    ws = np.concatenate(ws)
    hs = np.concatenate(hs)
    pad = 0.08
    xlim = (xs.min() - pad * np.ptp(xs), xs.max() + pad * np.ptp(xs))
    wlim = (ws.min() - pad * np.ptp(ws), ws.max() + pad * np.ptp(ws))
    hlim = (hs.min() - pad * np.ptp(hs), hs.max() + pad * np.ptp(hs))
    aspect = (np.ptp(xs), 3.0 * np.ptp(ws), 1.85 * np.ptp(hs))
    return xlim, wlim, hlim, aspect


def _draw(ax, faces, geom, C, wires, outer, xlim, wlim, hlim, aspect, norm, cmap, ls,
          title="", show_ylabels=True):
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
    order = sorted(faces, key=rank.get)
    for f in order:
        Xp = geom[f][0]
        Wp = geom[f][1]
        Hp = geom[f][2]
        rgb = ls.shade_rgb(cmap(norm(C[f])), Hp, blend_mode="soft")
        if f.startswith("e") or f in outer:
            alpha = 0.95
        else:
            alpha = 0.35
        ax.plot_surface(
            Xp, Wp, Hp, facecolors=rgb,
            linewidth=0, edgecolor="none", shade=False, antialiased=True,
            alpha=alpha,
        )
    Wspan = wires[:, 0, :]
    Wwidth = wires[:, 2, :]
    Wheight = wires[:, 1, :]
    for i in range(wires.shape[0]):
        ax.plot(
            Wspan[i], Wwidth[i], Wheight[i],
            color="0.18", lw=0.8, alpha=0.75, solid_capstyle="round",
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*wlim)
    ax.set_zlim(*hlim)
    try:
        ax.set_box_aspect(aspect)
    except Exception:
        pass
    ax.view_init(elev=30, azim=-52)
    ax.set_xticks([-60, 0, 60])
    ax.set_yticks([-5, 0, 5])
    ax.set_zticks([0, 12, 25])
    ax.tick_params(labelsize=11, pad=0)
    # Native 3D labels with pads tuned for this camera (no floating text2D).
    ax.set_xlabel(r"$x$", fontsize=14, labelpad=2)
    ax.set_ylabel(r"$z$", fontsize=14, labelpad=2)
    if show_ylabels:
        ax.set_zlabel(r"$y$", fontsize=14, labelpad=0)
    else:
        ax.set_zlabel("")
    if title:
        ax.set_title(title, fontsize=15, pad=2)


def _draw_key(ax_rail, cax, times, peak_frac, norm, cmap):
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
        ax_rail.plot(
            t, 0.53, "o", ms=10, color=col,
            markeredgecolor="0.12", markeredgewidth=0.7, zorder=5,
        )
    ax_rail.set_xticks(list(times))
    tick_labels = []
    for t in times:
        tick_labels.append(f"{t:.2f}")
    ax_rail.set_xticklabels(tick_labels, fontsize=10, rotation=30, ha="right")
    ax_rail.set_xlabel(r"$t$", fontsize=14, labelpad=2)
    ax_rail.tick_params(axis="x", length=5, width=1.0, colors="0.15")

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(r"$|u|$ [m]", fontsize=13, labelpad=2)
    cbar.ax.tick_params(labelsize=11)


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    info = {}
    for line in open(SNAP / "info.txt"):
        key, value = line.strip().split("=", 1)
        info[key] = value
    nF = int(info["nF"])
    cmax = float(info["cmax"])
    faces = []
    for s in info["faces"].split(","):
        if s.strip():
            faces.append(s.strip())
    times = read_f64(SNAP / "times.bin")
    wires = read_f64(SNAP / "wires.bin")
    nh = 0
    nw = 0
    for f in faces:
        if f.startswith("h"):
            nh += 1
        if f.startswith("w"):
            nw += 1
    outer = {f"h1", f"h{nh}", f"w1", f"w{nw}", "e0", "e1"}

    up = 3
    geom = {}
    for f in faces:
        X = read_f64(SNAP / f"X_{f}.bin")
        Y = read_f64(SNAP / f"Y_{f}.bin")
        Z = read_f64(SNAP / f"Z_{f}.bin")
        Xz = zoom(X, up, order=3)
        Zz = zoom(Z, up, order=3)
        Yz = zoom(Y, up, order=3)
        geom[f] = (Xz, Zz, Yz)

    Call = {}
    for f in faces:
        Call[f] = read_f64(SNAP / f"C_{f}.bin")
    per_face_max = []
    for f in faces:
        per_face_max.append(Call[f].reshape(nF, -1).max(axis=1))
    mmax = np.max(np.stack(per_face_max), axis=0)

    # Pick 12 frames: t=0, the last frame, then the largest local peaks of
    # the global |u| curve, subject to a 0.6 s minimum separation.
    nS = 12
    t_eq = float(info.get("t_eq", 1.5))
    sel = [0, int(np.argmin(np.abs(times - t_eq)))]
    peaks = []
    for k in range(1, nF - 1):
        if mmax[k] >= mmax[k - 1] and mmax[k] >= mmax[k + 1]:
            peaks.append(k)
    ranked = []
    for k in peaks:
        ranked.append((-mmax[k], k))
    ranked.sort()
    for _neg_mmax, k in ranked:
        if len(sel) >= nS - 1:
            break
        far_enough = True
        for j in sel:
            if abs(times[k] - times[j]) < 0.6:
                far_enough = False
                break
        if far_enough:
            sel.append(k)
    sel.append(nF - 1)
    sel = sorted(sel)

    snaps = []
    peak_frac = []
    for k in sel:
        Cs = {}
        for f in faces:
            Cs[f] = gaussian_filter(zoom(Call[f][k], up, order=3), 1.5)
        snaps.append(Cs)
        peak_frac.append(mmax[k] / max(cmax, 1e-16))
    times = times[sel]

    # Strain energy at the same times (own face set, log scale).
    EDIR = SNAP.parent / "energy"
    einfo = {}
    for line in open(EDIR / "info.txt"):
        key, value = line.strip().split("=", 1)
        einfo[key] = value
    psi_max = float(einfo["psi_max"])
    efaces = []
    for s in einfo["faces"].split(","):
        if s.strip():
            efaces.append(s.strip())
    etimes = read_f64(EDIR / "times.bin")
    egeom = {}
    for f in efaces:
        X = read_f64(EDIR / f"X_{f}.bin")
        Y = read_f64(EDIR / f"Y_{f}.bin")
        Z = read_f64(EDIR / f"Z_{f}.bin")
        Xz = zoom(X, up, order=3)
        Zz = zoom(Z, up, order=3)
        Yz = zoom(Y, up, order=3)
        egeom[f] = (Xz, Zz, Yz)
    Pall = {}
    for f in efaces:
        Pall[f] = np.maximum(read_f64(EDIR / f"P_{f}.bin"), 1e-30)
    esnaps = []
    for t in times:
        ke = int(np.argmin(np.abs(etimes - t)))
        frame = {}
        for f in efaces:
            frame[f] = gaussian_filter(zoom(Pall[f][ke], up, order=3), 1.5)
        esnaps.append(frame)
    enorm = LogNorm(vmin=max(psi_max * 1e-4, 1e-30), vmax=psi_max)
    ecmap = cm.inferno

    xlim, wlim, hlim, aspect = _limits(geom, faces, wires)
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=max(cmax, 1e-16))
    cmap = cm.turbo
    ls = LightSource(azdeg=315, altdeg=45)

    # Six 2x2 panel files: |u| on top, psi below. Figure10a–f + key.
    for pair in range(6):
        fig = plt.figure(figsize=(13.5, 10.5))
        gs = fig.add_gridspec(
            2, 2, left=0.02, right=0.98, top=0.95, bottom=0.02,
            wspace=0.10, hspace=0.02,
        )
        for c in range(2):
            k = 2 * pair + c
            ax = fig.add_subplot(gs[0, c], projection="3d")
            title = rf"$({chr(ord('a')+k)})$  $t={times[k]:.2f}\,\mathrm{{s}}$"
            _draw(
                ax, faces, geom, snaps[k], wires, outer, xlim, wlim, hlim, aspect, norm, cmap, ls,
                title=title, show_ylabels=(c == 0),
            )
            ax = fig.add_subplot(gs[1, c], projection="3d")
            _draw(
                ax, efaces, egeom, esnaps[k], wires, set(efaces), xlim, wlim, hlim, aspect,
                enorm, ecmap, ls, title=r"$\psi$", show_ylabels=(c == 0),
            )
        stem = f"Figure10{chr(ord('a')+pair)}"
        for ext in (".png", ".pdf"):
            path = OUTDIR / f"{stem}{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
            (PAPER_FIG / f"{stem}{ext}").write_bytes(path.read_bytes())
            print("wrote", path)
        plt.close(fig)

    fig = plt.figure(figsize=(12.0, 3.8))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.8],
        left=0.06, right=0.98, top=0.92, bottom=0.22, hspace=0.9,
    )
    ax_rail = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])
    _draw_key(ax_rail, cax, times, peak_frac, norm, cmap)
    for ext in (".png", ".pdf"):
        path = OUTDIR / f"Figure10_key{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        (PAPER_FIG / f"Figure10_key{ext}").write_bytes(path.read_bytes())
        print("wrote", path)
    plt.close(fig)

    for k, t in enumerate(times):
        print(f"  ({chr(ord('a')+k)}) t={t:.2f}  frac={peak_frac[k]:.2f}")


if __name__ == "__main__":
    main()
