#figuers 1 to 4 show where the Pade approximants 
# are not A stable and the repair we use. DISCLAIMER:
# this file is mostly AI generated, because it 
# is almost all mechanical plotting, and whatever
# could be messed up was verified.
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import fig_style as sty

sty.apply_style()
os.makedirs(sty.OUTDIR, exist_ok=True)
paper_fig = os.path.join(os.path.dirname(sty.HERE), "paper", "figures")

pad = 0.10
cyan = ListedColormap(["#00e5ff"])

def mag_arg_blend(r0, r1, theta):
    mag = (np.abs(r1) ** theta) * (np.abs(r0) ** (1.0 - theta))
    phase = (1.0 - theta) * np.angle(r0) + theta * np.angle(r1)
    return mag * np.exp(1j * phase)

def plot_triptych(X, Y, panels, curves, xlim, ylim, path):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), sharey=True)
    vmin = min(float(np.min(np.clip(M, 0.0, 1.0))) for M, _ in panels)
    pcm = None
    for ax, (Mag, title) in zip(axes, panels):
        pcm = ax.pcolormesh(
            X, Y, np.clip(Mag, 0.0, 1.0), shading="auto",
            cmap="inferno", vmin=vmin, vmax=1.0,
        )
        bad = (Mag > 1.0) & (X <= 0.0)
        ax.pcolormesh(
            X, Y, np.ma.masked_where(~bad, np.ones_like(Mag)),
            shading="auto", cmap=cyan, vmin=0.0, vmax=1.0,
        )
        for xc, yc, ls in curves:
            ax.plot(xc, yc, color="w", ls=ls, lw=(1.8 if ls == "-" else 1.4))
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

# ---- Fig1: [0/3] repaired by [1/2] ----
sqrt3 = np.sqrt(3.0)

def R03(z):
    return 1.0 / (1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0)

def R12(z):
    return (1.0 + z / 3.0) / (1.0 - 2.0 * z / 3.0 + z ** 2 / 6.0)

def F03(x, y):
    re = 1.0 - x + 0.5 * (x * x - y * y) - (x ** 3 - 3.0 * x * y * y) / 6.0
    im = -y + x * y - (3.0 * x * x * y - y ** 3) / 6.0
    return re * re + im * im - 1.0

def x_boundary_03(y):
    y, ay = float(y), abs(float(y))
    if ay <= 0.0 or ay >= sqrt3:
        return 0.0
    xs = np.linspace(-0.2, 0.0, 4001)
    v = F03(xs, y)
    hits = np.where(np.diff(np.signbit(v)))[0]
    a, b = float(xs[hits[-1]]), float(xs[hits[-1] + 1])
    for _ in range(60):
        m = 0.5 * (a + b)
        if F03(a, y) * F03(m, y) <= 0.0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)

_Y03 = np.linspace(0.0, sqrt3, 2001)
_XB03 = np.array([x_boundary_03(y) for y in _Y03])

def theta_03(z):
    x, y = np.real(z).ravel(), np.imag(z).ravel()
    ay = np.abs(y)
    y_max = (1.0 + pad) * sqrt3
    out = np.zeros(x.size, float)
    m = (ay < sqrt3) & (x <= 0.0)
    xb = np.interp(ay[m], _Y03, _XB03, left=0.0, right=0.0)
    x_outer = (1.0 + pad) * xb
    x_m = x[m]
    t = np.zeros(x_m.size, float)
    wide = xb < -1e-14
    xo, xb_w, xw = x_outer[wide], xb[wide], x_m[wide]
    tw = (xw - xo) / (xb_w - xo)
    tw = np.where(xw >= xb_w, 1.0, tw)
    tw = np.where(xw <= xo, 0.0, tw)
    t[wide] = np.clip(tw, 0.0, 1.0)
    t[(~wide) & (x_m >= -1e-14) & (ay[m] > 0.0)] = 1.0
    out[m] = t
    tip = (ay >= sqrt3) & (ay < y_max) & (x <= 0.0)
    s = (y_max - ay[tip]) / (pad * sqrt3)
    xo = -pad * 0.04 * s
    x_tip = x[tip]
    out[tip] = np.where(x_tip <= xo, 0.0, np.where(x_tip >= 0.0, s, s * (x_tip - xo) / (0.0 - xo)))
    return out.reshape(np.shape(z))

re_lo = (1.0 + pad) * float(_XB03.min()) - 0.02
im_hi = (1.0 + pad) * sqrt3 + 0.15
re = np.linspace(re_lo, 0.0, 500)
im = np.linspace(-im_hi, im_hi, 700)
X, Y = np.meshgrid(re, im)
Z = X + 1j * Y
yb = np.linspace(1e-4, sqrt3 - 1e-4, 500)
xb = np.interp(yb, _Y03, _XB03)
plot_triptych(
    X, Y,
    [(np.abs(R03(Z)), r"raw $R_{0,3}$"),
     (np.abs(R12(Z)), r"raw $R_{1,2}$"),
     (np.abs(mag_arg_blend(R03(Z), R12(Z), theta_03(Z))), r"stitched $R_D$")],
    [(xb, yb, "-"), (xb, -yb, "-"),
     ((1 + pad) * xb, yb, "--"), ((1 + pad) * xb, -yb, "--")],
    (re_lo, 0.0), (-im_hi, im_hi),
    os.path.join(sty.OUTDIR, "Fig1.png"),
)

# ---- Fig2: [0/4] repaired by [1/3] ----
sqrt8 = np.sqrt(8.0)

def R04(z):
    return 1.0 / (1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0)

def R13(z):
    return (1.0 + 0.25 * z) / (1.0 - 0.75 * z + 0.25 * z ** 2 - z ** 3 / 24.0)

def F04(x, y):
    z = np.asarray(x, float) + 1j * np.asarray(y, float)
    return np.abs(1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0) ** 2 - 1.0

def x_boundary_04(y):
    y, ay = float(y), abs(float(y))
    if ay <= 0.0 or ay >= sqrt8:
        return 0.0
    xs = np.linspace(-0.5, 0.0, 4001)
    v = F04(xs, y)
    hits = np.where(np.diff(np.signbit(v)))[0]
    a, b = float(xs[hits[-1]]), float(xs[hits[-1] + 1])
    for _ in range(60):
        m = 0.5 * (a + b)
        if float(F04(a, y)) * float(F04(m, y)) <= 0.0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)

_Y04 = np.linspace(0.0, sqrt8, 2001)
_XB04 = np.array([x_boundary_04(y) for y in _Y04])

def theta_04(z):
    x, y = np.real(z).ravel(), np.imag(z).ravel()
    ay = np.abs(y)
    y_max = (1.0 + pad) * sqrt8
    out = np.zeros(x.size, float)
    m = (ay < sqrt8) & (x <= 0.0)
    xb = np.interp(ay[m], _Y04, _XB04, left=0.0, right=0.0)
    x_outer = (1.0 + pad) * xb
    x_m = x[m]
    t = np.zeros(x_m.size, float)
    wide = xb < -1e-14
    xo, xb_w, xw = x_outer[wide], xb[wide], x_m[wide]
    tw = (xw - xo) / (xb_w - xo)
    tw = np.where(xw >= xb_w, 1.0, tw)
    tw = np.where(xw <= xo, 0.0, tw)
    t[wide] = np.clip(tw, 0.0, 1.0)
    t[(~wide) & (x_m >= -1e-14) & (ay[m] > 0.0)] = 1.0
    out[m] = t
    tip = (ay >= sqrt8) & (ay < y_max) & (x <= 0.0)
    s = (y_max - ay[tip]) / (pad * sqrt8)
    xo = -pad * 0.25 * s
    x_tip = x[tip]
    out[tip] = np.where(x_tip <= xo, 0.0, np.where(x_tip >= 0.0, s, s * (x_tip - xo) / (0.0 - xo)))
    return out.reshape(np.shape(z))

re_lo = (1.0 + pad) * float(_XB04.min()) - 0.03
im_hi = (1.0 + pad) * sqrt8 + 0.2
re = np.linspace(re_lo, 0.0, 500)
im = np.linspace(-im_hi, im_hi, 700)
X, Y = np.meshgrid(re, im)
Z = X + 1j * Y
yb = np.linspace(1e-4, sqrt8 - 1e-4, 500)
xb = np.interp(yb, _Y04, _XB04)
plot_triptych(
    X, Y,
    [(np.abs(R04(Z)), r"raw $R_{0,4}$"),
     (np.abs(R13(Z)), r"raw $R_{1,3}$"),
     (np.abs(mag_arg_blend(R04(Z), R13(Z), theta_04(Z))), r"stitched $R_D$")],
    [(xb, yb, "-"), (xb, -yb, "-"),
     ((1 + pad) * xb, yb, "--"), ((1 + pad) * xb, -yb, "--")],
    (re_lo, 0.0), (-im_hi, im_hi),
    os.path.join(sty.OUTDIR, "Fig2.png"),
)

# ---- Fig3: [0/5] repaired by [2/3] ----
def R05(z):
    return 1.0 / (1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0 - z ** 5 / 120.0)

def R23(z):
    return (1.0 + 0.4 * z + 0.05 * z ** 2) / (1.0 - 0.6 * z + 0.15 * z ** 2 - z ** 3 / 60.0)

def F05(z):
    return np.abs(1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0 - z ** 5 / 120.0) ** 2 - 1.0

Y_LO = np.sqrt((15 - np.sqrt(65)) / 2)
Y_HI = np.sqrt((15 + np.sqrt(65)) / 2)
C05 = 1j * (0.5 * (Y_LO + Y_HI))

def ray_boundary_05(alpha):
    d = np.exp(1j * alpha)
    rs = np.linspace(0.0, 1.5, 2001)
    Fs = F05(C05 + rs * d)
    i = np.where((Fs[:-1] < 0) & (Fs[1:] >= 0))[0]
    a, b = float(rs[i[0]]), float(rs[i[0] + 1])
    for _ in range(50):
        m = 0.5 * (a + b)
        if F05(C05 + m * d) < 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)

_A05 = np.linspace(0.5 * np.pi, 1.5 * np.pi, 801)
_R05 = np.array([ray_boundary_05(a) for a in _A05])

def theta_05(z):
    x = np.minimum(np.real(z), 0.0).ravel()
    ay = np.abs(np.imag(z)).ravel()
    d = (x + 1j * ay) - C05
    r = np.abs(d)
    out = np.zeros(x.size, float)
    tiny = r < 1e-14
    out[tiny] = 1.0
    ok = ~tiny
    alpha = np.angle(d[ok])
    alpha = np.where(alpha < 0, alpha + 2 * np.pi, alpha)
    alpha = np.clip(alpha, 0.5 * np.pi, 1.5 * np.pi)
    rb = np.interp(alpha, _A05, _R05)
    s = np.where(rb > 0, r[ok] / rb, np.inf)
    out[ok] = np.where(s <= 1.0, 1.0, np.where(s >= 1.0 + pad, 0.0, (1.0 + pad - s) / pad))
    return out.reshape(np.shape(z))

pts_b = C05 + _R05 * np.exp(1j * _A05)
pts_b = np.r_[pts_b, np.conj(pts_b[::-1])]
pts_o = C05 + (1.0 + pad) * _R05 * np.exp(1j * _A05)
pts_o = np.r_[pts_o, np.conj(pts_o[::-1])]
xb, yb = np.real(pts_b), np.imag(pts_b)
xo, yo = np.real(pts_o), np.imag(pts_o)
xb, yb = xb[yb >= 0], yb[yb >= 0]
xo, yo = xo[yo >= 0], yo[yo >= 0]
pad_box = 0.20
re_lo = min(float(np.min(xo)), float(np.min(xb))) - pad_box
im_lo = max(0.0, float(np.min(np.r_[yb, yo])) - pad_box)
im_hi = float(np.max(np.r_[yb, yo])) + pad_box
re = np.linspace(re_lo, 0.0, 500)
im = np.linspace(im_lo, im_hi, 500)
X, Y = np.meshgrid(re, im)
Z = X + 1j * Y
plot_triptych(
    X, Y,
    [(np.abs(R05(Z)), r"raw $R_{0,5}$"),
     (np.abs(R23(Z)), r"raw $R_{2,3}$"),
     (np.abs(mag_arg_blend(R05(Z), R23(Z), theta_05(Z))), r"stitched $R_D$")],
    [(xb, yb, "-"), (xo, yo, "--")],
    (re_lo, 0.0), (im_lo, im_hi),
    os.path.join(sty.OUTDIR, "Fig3.png"),
)

# ---- Fig4: [0/6] repaired by [2/4] ----
def Q06(z):
    return 1.0 - z + 0.5 * z ** 2 - z ** 3 / 6.0 + z ** 4 / 24.0 - z ** 5 / 120.0 + z ** 6 / 720.0

def R06(z):
    return 1.0 / Q06(z)

def R24(z):
    return (1.0 + z / 3.0 + z ** 2 / 30.0) / (
        1.0 - 2.0 * z / 3.0 + z ** 2 / 5.0 - z ** 3 / 30.0 + z ** 4 / 360.0
    )

def F06(z):
    return np.abs(Q06(z)) ** 2 - 1.0

P06 = -0.8 + 3.7j
for _ in range(50):
    Q5 = 1.0 - P06 + 0.5 * P06 ** 2 - P06 ** 3 / 6.0 + P06 ** 4 / 24.0 - P06 ** 5 / 120.0
    dz = Q06(P06) / (-Q5)
    P06 = P06 - dz
    if abs(dz) < 1e-15:
        break
P06 = complex(P06)

def ray_boundary_06(alpha):
    d = np.exp(1j * alpha)
    rs = np.linspace(0.0, 1.0, 3001)
    Fs = F06(P06 + rs * d)
    i = np.where((Fs[:-1] < 0) & (Fs[1:] >= 0))[0]
    a, b = float(rs[i[0]]), float(rs[i[0] + 1])
    for _ in range(50):
        m = 0.5 * (a + b)
        if F06(P06 + m * d) < 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)

_A06 = np.linspace(0.0, 2 * np.pi, 801)
_R06 = np.array([ray_boundary_06(a) for a in _A06])

def theta_06(z):
    d = (np.real(z) + 1j * np.abs(np.imag(z))) - P06
    d = d.ravel()
    r = np.abs(d)
    out = np.zeros(d.size, float)
    tiny = r < 1e-14
    out[tiny] = 1.0
    ok = ~tiny
    alpha = np.mod(np.angle(d[ok]), 2 * np.pi)
    rb = np.interp(alpha, _A06, _R06, period=2 * np.pi)
    s = np.where(rb > 0, r[ok] / rb, np.inf)
    out[ok] = np.where(s <= 1.0, 1.0, np.where(s >= 1.0 + pad, 0.0, (1.0 + pad - s) / pad))
    return out.reshape(np.shape(z))

xu = np.real(P06 + _R06 * np.exp(1j * _A06))
yu = np.imag(P06 + _R06 * np.exp(1j * _A06))
xo = np.real(P06 + (1.0 + pad) * _R06 * np.exp(1j * _A06))
yo = np.imag(P06 + (1.0 + pad) * _R06 * np.exp(1j * _A06))
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
plot_triptych(
    X, Y,
    [(np.abs(R06(Z)), r"raw $R_{0,6}$"),
     (np.abs(R24(Z)), r"raw $R_{2,4}$"),
     (np.abs(mag_arg_blend(R06(Z), R24(Z), theta_06(Z))), r"stitched $R_D$")],
    [(xu, yu, "-"), (xo, yo, "--")],
    (re_lo, re_hi), (im_lo, im_hi),
    os.path.join(sty.OUTDIR, "Fig4.png"),
)

for name in ("Fig1.png", "Fig2.png", "Fig3.png", "Fig4.png"):
    open(os.path.join(paper_fig, name), "wb").write(
        open(os.path.join(sty.OUTDIR, name), "rb").read()
    )
