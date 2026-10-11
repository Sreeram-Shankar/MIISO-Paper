"""Paper Fig.1–4: Padé [0/N] instability wedges with repair (N = 3,4,5,6).

Each figure is a 1×3 triptych in the original style.
Writes figures/Fig1.png … Fig4.png (and copies into paper/figures/).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from fig_style import OUTDIR, apply_style

PAD = 0.10
CYAN = "#00e5ff"
PAPER_FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"


def plot_triptych(X, Y, panels, curves, xlim, ylim, path):
    """curves: list of (x, y, ls) overlays drawn on every panel."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), sharey=True)
    clipped_mins = []
    for M, _ in panels:
        clipped_mins.append(float(np.min(np.clip(M, 0.0, 1.0))))
    vmin = min(clipped_mins)
    pcm = None
    for ax, (Mag, title) in zip(axes, panels):
        pcm = ax.pcolormesh(
            X, Y, np.clip(Mag, 0.0, 1.0), shading="auto",
            cmap="inferno", vmin=vmin, vmax=1.0,
        )
        bad = (Mag > 1.0) & (X <= 0.0)
        if np.any(bad):
            ax.pcolormesh(
                X, Y, np.ma.masked_where(~bad, np.ones_like(Mag)),
                shading="auto", cmap=ListedColormap([CYAN]),
                vmin=0.0, vmax=1.0,
            )
        for xc, yc, ls in curves:
            if ls == "-":
                lw = 1.8
            else:
                lw = 1.4
            ax.plot(xc, yc, color="w", ls=ls, lw=lw)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel(r"$\mathrm{Re}\,z$", fontsize=22)
        ax.set_title(title, fontsize=22, pad=10)
        ax.tick_params(direction="in", top=True, right=True, labelsize=18, length=6, width=1.1)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(18)
    axes[0].set_ylabel(r"$\mathrm{Im}\,z$", fontsize=22)
    cb = fig.colorbar(pcm, ax=axes, pad=0.018, fraction=0.032)
    cb.set_label(r"$|R(z)|$", fontsize=22)
    cb.ax.tick_params(labelsize=18)
    fig.savefig(path)
    plt.close(fig)


def mag_arg_blend(r0, r1, th):
    mag = (np.abs(r1) ** th) * (np.abs(r0) ** (1.0 - th))
    phase = (1.0 - th) * np.angle(r0) + th * np.angle(r1)
    return mag * np.exp(1j * phase)


def save(path):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    return path


# ---- Padé [0/3] repaired by [1/2] -------------------------------------------

SQRT3 = np.sqrt(3.0)


def R03(z):
    z = np.asarray(z, complex)
    return 1.0 / (1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0)


def R12(z):
    z = np.asarray(z, complex)
    return (1.0 + z / 3.0) / (1.0 - 2.0 * z / 3.0 + z ** 2 / 6.0)


def F03(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    re = 1.0 - x + 0.5 * (x * x - y * y) - (x ** 3 - 3.0 * x * y * y) / 6.0
    im = -y + x * y - (3.0 * x * x * y - y ** 3) / 6.0
    return re * re + im * im - 1.0


def x_boundary_03(y, xlo=-0.2, n=4001):
    y, ay = float(y), abs(float(y))
    if ay <= 0.0 or ay >= SQRT3:
        return 0.0
    xs = np.linspace(xlo, 0.0, n)
    v = F03(xs, y)
    hits = np.where(np.diff(np.signbit(v)))[0]
    if hits.size == 0:
        return 0.0
    i = hits[-1]
    a, b = float(xs[i]), float(xs[i + 1])
    for _ in range(60):
        m = 0.5 * (a + b)
        if F03(a, y) * F03(m, y) <= 0.0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


_Y03 = np.linspace(0.0, SQRT3, 2001)
_XB03_vals = []
for y in _Y03:
    _XB03_vals.append(x_boundary_03(y))
_XB03 = np.array(_XB03_vals)


def theta_03(z, pad=PAD):
    z = np.asarray(z, complex)
    x, y = np.real(z).ravel(), np.imag(z).ravel()
    ay = np.abs(y)
    y_max = (1.0 + pad) * SQRT3
    out = np.zeros(x.size, float)
    m = (ay < SQRT3) & (x <= 0.0)
    if np.any(m):
        xb = np.interp(ay[m], _Y03, _XB03, left=0.0, right=0.0)
        x_outer = (1.0 + pad) * xb
        xi = x[m]
        t = np.zeros(xi.size, float)
        wide = xb < -1e-14
        if np.any(wide):
            xo, xb_w, xw = x_outer[wide], xb[wide], xi[wide]
            tw = (xw - xo) / (xb_w - xo)
            tw = np.where(xw >= xb_w, 1.0, tw)
            tw = np.where(xw <= xo, 0.0, tw)
            t[wide] = np.clip(tw, 0.0, 1.0)
        t[(~wide) & (xi >= -1e-14) & (ay[m] > 0.0)] = 1.0
        out[m] = t
    tip = (ay >= SQRT3) & (ay < y_max) & (x <= 0.0)
    if np.any(tip):
        s = (y_max - ay[tip]) / (pad * SQRT3)
        xo = -pad * 0.04 * s
        xi = x[tip]
        out[tip] = np.where(xi <= xo, 0.0, np.where(xi >= 0.0, s, s * (xi - xo) / (0.0 - xo)))
    return out.reshape(np.shape(z))


def R_D03(z):
    return mag_arg_blend(R03(z), R12(z), theta_03(z))


def make_fig1():
    y_tab = np.linspace(0.0, SQRT3, 401)
    xb_tab_vals = []
    for y in y_tab:
        xb_tab_vals.append(x_boundary_03(y))
    xb_tab = np.array(xb_tab_vals)
    re_lo = (1.0 + PAD) * float(xb_tab.min()) - 0.02
    im_hi = (1.0 + PAD) * SQRT3 + 0.15
    re = np.linspace(re_lo, 0.0, 500)
    im = np.linspace(-im_hi, im_hi, 700)
    X, Y = np.meshgrid(re, im)
    Z = X + 1j * Y
    yb = np.linspace(1e-4, SQRT3 - 1e-4, 500)
    xb_vals = []
    for y in yb:
        xb_vals.append(x_boundary_03(y))
    xb = np.array(xb_vals)
    curves = [
        (xb, yb, "-"), (xb, -yb, "-"),
        ((1 + PAD) * xb, yb, "--"), ((1 + PAD) * xb, -yb, "--"),
    ]
    path = OUTDIR / "Fig1.png"
    plot_triptych(
        X, Y,
        [(np.abs(R03(Z)), r"raw $R_{0,3}$"),
         (np.abs(R12(Z)), r"raw $R_{1,2}$"),
         (np.abs(R_D03(Z)), r"stitched $R_D$")],
        curves, (re_lo, 0.0), (-im_hi, im_hi), path,
    )
    print("wrote", path)


# ---- Padé [0/4] repaired by [1/3] -------------------------------------------

SQRT8 = np.sqrt(8.0)


def R04(z):
    z = np.asarray(z, complex)
    return 1.0 / (1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0)


def R13(z):
    z = np.asarray(z, complex)
    return (1.0 + 0.25 * z) / (1.0 - 0.75 * z + 0.25 * z ** 2 - z ** 3 / 24.0)


def F04(x, y):
    z = np.asarray(x, float) + 1j * np.asarray(y, float)
    return np.abs(1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0) ** 2 - 1.0


def x_boundary_04(y, xlo=-0.5, n=4001):
    y, ay = float(y), abs(float(y))
    if ay <= 0.0 or ay >= SQRT8:
        return 0.0
    xs = np.linspace(xlo, 0.0, n)
    v = F04(xs, y)
    hits = np.where(np.diff(np.signbit(v)))[0]
    if hits.size == 0:
        return 0.0
    i = int(hits[-1])
    a, b = float(xs[i]), float(xs[i + 1])
    for _ in range(60):
        m = 0.5 * (a + b)
        if float(F04(a, y)) * float(F04(m, y)) <= 0.0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


_Y04 = np.linspace(0.0, SQRT8, 2001)
_XB04_vals = []
for y in _Y04:
    _XB04_vals.append(x_boundary_04(y))
_XB04 = np.array(_XB04_vals)


def theta_04(z, pad=PAD):
    z = np.asarray(z, complex)
    x, y = np.real(z).ravel(), np.imag(z).ravel()
    ay = np.abs(y)
    y_max = (1.0 + pad) * SQRT8
    out = np.zeros(x.size, float)
    m = (ay < SQRT8) & (x <= 0.0)
    if np.any(m):
        xb = np.interp(ay[m], _Y04, _XB04, left=0.0, right=0.0)
        x_outer = (1.0 + pad) * xb
        xi = x[m]
        t = np.zeros(xi.size, float)
        wide = xb < -1e-14
        if np.any(wide):
            xo, xb_w, xw = x_outer[wide], xb[wide], xi[wide]
            tw = (xw - xo) / (xb_w - xo)
            tw = np.where(xw >= xb_w, 1.0, tw)
            tw = np.where(xw <= xo, 0.0, tw)
            t[wide] = np.clip(tw, 0.0, 1.0)
        t[(~wide) & (xi >= -1e-14) & (ay[m] > 0.0)] = 1.0
        out[m] = t
    tip = (ay >= SQRT8) & (ay < y_max) & (x <= 0.0)
    if np.any(tip):
        s = (y_max - ay[tip]) / (pad * SQRT8)
        xo = -pad * 0.25 * s
        xi = x[tip]
        out[tip] = np.where(xi <= xo, 0.0, np.where(xi >= 0.0, s, s * (xi - xo) / (0.0 - xo)))
    return out.reshape(np.shape(z))


def R_D04(z):
    return mag_arg_blend(R04(z), R13(z), theta_04(z))


def make_fig2():
    y_tab = np.linspace(0.0, SQRT8, 401)
    xb_tab_vals = []
    for y in y_tab:
        xb_tab_vals.append(x_boundary_04(y))
    xb_tab = np.array(xb_tab_vals)
    re_lo = (1.0 + PAD) * float(xb_tab.min()) - 0.03
    im_hi = (1.0 + PAD) * SQRT8 + 0.2
    re = np.linspace(re_lo, 0.0, 500)
    im = np.linspace(-im_hi, im_hi, 700)
    X, Y = np.meshgrid(re, im)
    Z = X + 1j * Y
    yb = np.linspace(1e-4, SQRT8 - 1e-4, 500)
    xb_vals = []
    for y in yb:
        xb_vals.append(x_boundary_04(y))
    xb = np.array(xb_vals)
    curves = [
        (xb, yb, "-"), (xb, -yb, "-"),
        ((1 + PAD) * xb, yb, "--"), ((1 + PAD) * xb, -yb, "--"),
    ]
    path = OUTDIR / "Fig2.png"
    plot_triptych(
        X, Y,
        [(np.abs(R04(Z)), r"raw $R_{0,4}$"),
         (np.abs(R13(Z)), r"raw $R_{1,3}$"),
         (np.abs(R_D04(Z)), r"stitched $R_D$")],
        curves, (re_lo, 0.0), (-im_hi, im_hi), path,
    )
    print("wrote", path)


# ---- Padé [0/5] repaired by [2/3] -------------------------------------------

def R05(z):
    z = np.asarray(z, complex)
    return 1.0 / (1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0 - z ** 5 / 120.0)


def R23(z):
    z = np.asarray(z, complex)
    return (1.0 + 0.4 * z + 0.05 * z ** 2) / (1.0 - 0.6 * z + 0.15 * z ** 2 - z ** 3 / 60.0)


def F05(z):
    z = np.asarray(z, complex)
    return np.abs(1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0 - z ** 5 / 120.0) ** 2 - 1.0


Y_LO = np.sqrt((15 - np.sqrt(65)) / 2)
Y_HI = np.sqrt((15 + np.sqrt(65)) / 2)
Y_C = 0.5 * (Y_LO + Y_HI)
C05 = 1j * Y_C


def ray_boundary_05(alpha, rmax=1.5, n=2001):
    d = np.exp(1j * alpha)
    rs = np.linspace(0.0, rmax, n)
    Fs = F05(C05 + rs * d)
    i = np.where((Fs[:-1] < 0) & (Fs[1:] >= 0))[0]
    if i.size == 0:
        return 0.0
    j = int(i[0])
    a, b = float(rs[j]), float(rs[j + 1])
    for _ in range(50):
        m = 0.5 * (a + b)
        if F05(C05 + m * d) < 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


_A05 = np.linspace(0.5 * np.pi, 1.5 * np.pi, 801)
_R05_vals = []
for a in _A05:
    _R05_vals.append(ray_boundary_05(a))
_R05 = np.array(_R05_vals)


def theta_05(z, pad=PAD):
    z = np.asarray(z, complex)
    x = np.minimum(np.real(z), 0.0).ravel()
    ay = np.abs(np.imag(z)).ravel()
    d = (x + 1j * ay) - C05
    r = np.abs(d)
    out = np.zeros(x.size, float)
    tiny = r < 1e-14
    out[tiny] = 1.0
    ok = ~tiny
    if np.any(ok):
        alpha = np.angle(d[ok])
        alpha = np.where(alpha < 0, alpha + 2 * np.pi, alpha)
        alpha = np.clip(alpha, 0.5 * np.pi, 1.5 * np.pi)
        rb = np.interp(alpha, _A05, _R05)
        s = np.where(rb > 0, r[ok] / rb, np.inf)
        out[ok] = np.where(s <= 1.0, 1.0, np.where(s >= 1.0 + pad, 0.0, (1.0 + pad - s) / pad))
    return out.reshape(np.shape(z))


def R_D05(z):
    return mag_arg_blend(R05(z), R23(z), theta_05(z))


def closed_from_rays(center, angles, radii, pad=0.0):
    pts = center + (1.0 + pad) * radii * np.exp(1j * angles)
    if np.isclose(center.real, 0.0) and center.imag > 0 and angles[0] >= 0.5 * np.pi - 1e-9:
        pts = np.r_[pts, np.conj(pts[::-1])]
    return np.real(pts), np.imag(pts)


def make_fig3():
    # Zoom on the upper imag-axis band (conjugate-symmetric). Keep Re≤0.
    xb, yb = closed_from_rays(C05, _A05, _R05, 0.0)
    xo, yo = closed_from_rays(C05, _A05, _R05, PAD)
    # keep only upper half of the closed curves
    up_b = yb >= 0
    up_o = yo >= 0
    xb, yb = xb[up_b], yb[up_b]
    xo, yo = xo[up_o], yo[up_o]
    pad_box = 0.20
    re_lo = min(float(np.min(xo)), float(np.min(xb))) - pad_box
    re_hi = 0.0
    im_lo = max(0.0, float(np.min(np.r_[yb, yo])) - pad_box)
    im_hi = float(np.max(np.r_[yb, yo])) + pad_box
    re = np.linspace(re_lo, re_hi, 500)
    im = np.linspace(im_lo, im_hi, 500)
    X, Y = np.meshgrid(re, im)
    Z = X + 1j * Y
    curves = [(xb, yb, "-"), (xo, yo, "--")]
    path = OUTDIR / "Fig3.png"
    plot_triptych(
        X, Y,
        [(np.abs(R05(Z)), r"raw $R_{0,5}$"),
         (np.abs(R23(Z)), r"raw $R_{2,3}$"),
         (np.abs(R_D05(Z)), r"stitched $R_D$")],
        curves, (re_lo, re_hi), (im_lo, im_hi), path,
    )
    print("wrote", path)


# ---- Padé [0/6] repaired by [2/4] -------------------------------------------

def Q06(z):
    z = np.asarray(z, complex)
    return 1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0 - z ** 5 / 120.0 + z ** 6 / 720.0


def Q05(z):
    z = np.asarray(z, complex)
    return 1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0 - z ** 5 / 120.0


def R06(z):
    return 1.0 / Q06(z)


def R24(z):
    z = np.asarray(z, complex)
    return (1.0 + z / 3.0 + z ** 2 / 30.0) / (
        1.0 - 2.0 * z / 3.0 + z ** 2 / 5.0 - z ** 3 / 30.0 + z ** 4 / 360.0
    )


def F06(z):
    return np.abs(Q06(z)) ** 2 - 1.0


def pole06(z0=-0.8 + 3.7j):
    z = complex(z0)
    for _ in range(50):
        dz = Q06(z) / (-Q05(z))
        z -= dz
        if abs(dz) < 1e-15:
            break
    return z


P06 = pole06()


def ray_boundary_06(alpha, rmax=1.0, n=3001):
    d = np.exp(1j * alpha)
    rs = np.linspace(0.0, rmax, n)
    Fs = F06(P06 + rs * d)
    i = np.where((Fs[:-1] < 0) & (Fs[1:] >= 0))[0]
    if i.size == 0:
        return 0.0
    j = int(i[0])
    a, b = float(rs[j]), float(rs[j + 1])
    for _ in range(50):
        m = 0.5 * (a + b)
        if F06(P06 + m * d) < 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


_A06 = np.linspace(0.0, 2 * np.pi, 801)
_R06_vals = []
for a in _A06:
    _R06_vals.append(ray_boundary_06(a))
_R06 = np.array(_R06_vals)


def theta_06(z, pad=PAD):
    z = np.asarray(z, complex)
    d = (np.real(z) + 1j * np.abs(np.imag(z))) - P06
    d = d.ravel()
    r = np.abs(d)
    out = np.zeros(d.size, float)
    tiny = r < 1e-14
    out[tiny] = 1.0
    ok = ~tiny
    if np.any(ok):
        alpha = np.mod(np.angle(d[ok]), 2 * np.pi)
        rb = np.interp(alpha, _A06, _R06, period=2 * np.pi)
        s = np.where(rb > 0, r[ok] / rb, np.inf)
        out[ok] = np.where(s <= 1.0, 1.0, np.where(s >= 1.0 + pad, 0.0, (1.0 + pad - s) / pad))
    return out.reshape(np.shape(z))


def R_D06(z):
    return mag_arg_blend(R06(z), R24(z), theta_06(z))


def island_curve(center, angles, radii, pad=0.0):
    pts = center + (1.0 + pad) * radii * np.exp(1j * angles)
    return np.real(pts), np.imag(pts)


def make_fig4():
    # Zoom on the upper LHP island (conjugate-symmetric).
    xu, yu = island_curve(P06, _A06, _R06, 0.0)
    xo, yo = island_curve(P06, _A06, _R06, PAD)
    pad_box = 0.25
    xs = np.r_[xu, xo]
    ys = np.r_[yu, yo]
    re_lo = float(np.min(xs)) - pad_box
    re_hi = min(0.0, float(np.max(xs)) + pad_box)
    im_lo = float(np.min(ys)) - pad_box
    im_hi = float(np.max(ys)) + pad_box
    re = np.linspace(re_lo, re_hi, 500)
    im = np.linspace(im_lo, im_hi, 500)
    X, Y = np.meshgrid(re, im)
    Z = X + 1j * Y
    curves = [(xu, yu, "-"), (xo, yo, "--")]
    path = OUTDIR / "Fig4.png"
    plot_triptych(
        X, Y,
        [(np.abs(R06(Z)), r"raw $R_{0,6}$"),
         (np.abs(R24(Z)), r"raw $R_{2,4}$"),
         (np.abs(R_D06(Z)), r"stitched $R_D$")],
        curves, (re_lo, re_hi), (im_lo, im_hi), path,
    )
    print("wrote", path)


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    for name in ("Fig1.png", "Fig2.png", "Fig3.png", "Fig4.png"):
        src = OUTDIR / name
        dst = PAPER_FIG / name
        dst.write_bytes(src.read_bytes())
        print("copied", dst)


if __name__ == "__main__":
    main()
