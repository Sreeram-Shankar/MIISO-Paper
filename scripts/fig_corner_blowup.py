# generates figures 16 and 17 for the paper, which is 
# miiso4 vs exprb43 blowing up, geting its rsults from the test. DISCLAIMER:
# this file is mostly AI generated, because it 
# is almost all mechanical plotting, and whatever
# could be messed up was verified.
import os, struct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import fig_style as sty

sty.apply_style()
os.makedirs(sty.OUTDIR, exist_ok=True)

root = os.path.dirname(sty.HERE)
exp = os.path.join(root, "test", "membrance", "outlier_visual", "out", "export")
local_out = os.path.join(root, "test", "membrance", "outlier_visual", "out")
paper_fig = os.path.join(root, "paper", "figures")
t_target = 1.4

def read_f64(path):
    with open(path, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack("<%dq" % nd, f.read(8 * nd))
        return np.frombuffer(f.read(), dtype="<f8").reshape(shape, order="F")

def nice_lim(m):
    m = float(max(m, 1e-16))
    base = 10.0 ** int(np.floor(np.log10(m)))
    for f in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if f * base >= m:
            return f * base
    return 10.0 * base

def fmt_tick(x, _pos=None):
    ax = abs(x)
    if ax == 0.0:
        return r"$0$"
    if ax >= 10:
        return r"$%.0f$" % x
    if ax >= 1:
        return r"$%.1f$" % x
    if ax >= 0.01:
        return r"$%.2f$" % x
    return (r"$%.0e$" % x).replace("e-0", "e-").replace("e+0", "e+")

def draw_u(ax, X, Y, Uk, color_max, z_max):
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.78, 0.78, 0.78, 1.0))
        axis.pane.set_linewidth(0.55)
        axis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.45
    ax.plot_surface(
        X, Y, Uk, cmap="coolwarm", vmin=-color_max, vmax=color_max,
        rstride=1, cstride=1, linewidth=0.0, antialiased=True,
        shade=True, edgecolor="none",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(-z_max, z_max)
    ax.set_xlabel(r"$x$", labelpad=12, fontsize=20)
    ax.set_ylabel(r"$y$", labelpad=12, fontsize=20)
    ax.set_zlabel(r"$u$", labelpad=10, fontsize=20)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks(np.linspace(-z_max, z_max, 3))
    ax.zaxis.set_major_formatter(FuncFormatter(fmt_tick))
    ax.tick_params(labelsize=16, pad=3)
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect((1.0, 1.0, 0.62))

X = read_f64(os.path.join(exp, "X.bin"))
Y = read_f64(os.path.join(exp, "Y.bin"))
Um = read_f64(os.path.join(exp, "U_miiso.bin"))
Ue = read_f64(os.path.join(exp, "U_exprb.bin"))
Uom = read_f64(os.path.join(exp, "Uout_miiso.bin"))
Uoe = read_f64(os.path.join(exp, "Uout_exprb.bin"))
tm = read_f64(os.path.join(exp, "times_miiso.bin"))
te = read_f64(os.path.join(exp, "times_exprb.bin"))

im = int(np.argmin(np.abs(tm - t_target)))
ie = int(np.argmin(np.abs(te - t_target)))

step = 2 if X.shape[0] > 64 else 1
Xs = X[::step, ::step]
Ys = Y[::step, ::step]
UmT = Um[im, ::step, ::step]
UeT = Ue[ie, ::step, ::step]
UomT = Uom[im]
UoeT = Uoe[ie]

um_m = nice_lim(np.max(np.abs(UmT)))
ue_cap = 10.0
ue_zmax = nice_lim(np.max(np.abs(UeT)))
uo_vmax = nice_lim(np.max(np.abs(UoeT)))

fig = plt.figure(figsize=(14.5, 6.2))
ax0 = fig.add_axes([0.03, 0.18, 0.44, 0.78], projection="3d")
ax1 = fig.add_axes([0.52, 0.18, 0.44, 0.78], projection="3d")
draw_u(ax0, Xs, Ys, UmT, um_m, um_m)
draw_u(ax1, Xs, Ys, UeT, ue_cap, ue_zmax)
ax0.set_title(r"$(a)$", fontsize=22, pad=8)
ax1.set_title(r"$(b)$", fontsize=22, pad=8)
for i, vmax in enumerate((um_m, ue_cap)):
    cax = fig.add_axes([0.08 + i * 0.49, 0.08, 0.34, 0.035])
    mappable = cm.ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=-vmax, vmax=vmax))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(r"$u$", fontsize=18, labelpad=6)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(fmt_tick))
for d in (sty.OUTDIR, paper_fig, local_out):
    fig.savefig(os.path.join(d, "Fig16.png"))
fig.savefig(os.path.join(sty.OUTDIR, "Fig16.pdf"))
plt.close(fig)

norm = SymLogNorm(linthresh=1e-3, vmin=-uo_vmax, vmax=uo_vmax)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.9))
for ax, U, letter in ((axes[0], UomT, "a"), (axes[1], UoeT, "b")):
    im_show = ax.imshow(
        U, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu_r", norm=norm, aspect="equal", interpolation="bilinear",
    )
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$y$", fontsize=20)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["", r"$0.5$", r"$1.0$"])
    ax.tick_params(labelsize=16)
    ax.set_title(r"$(%s)$" % letter, fontsize=22, pad=10)
fig.subplots_adjust(bottom=0.24)
cax = fig.add_axes([0.16, 0.1, 0.68, 0.04])
cbar = fig.colorbar(im_show, cax=cax, orientation="horizontal")
cbar.set_label(r"$u_{\mathrm{out}}$ (log)", fontsize=18, labelpad=6)
cbar.ax.tick_params(labelsize=15)
ticks = [-10.0, -1.0, -0.1, -0.01, -1e-3, 1e-3, 0.01, 0.1, 1.0, 10.0]
cbar.set_ticks([t for t in ticks if -uo_vmax < t < uo_vmax])
cbar.ax.xaxis.set_major_formatter(FuncFormatter(fmt_tick))
for d in (sty.OUTDIR, paper_fig, local_out):
    fig.savefig(os.path.join(d, "Fig17.png"))
fig.savefig(os.path.join(sty.OUTDIR, "Fig17.pdf"))
plt.close(fig)

for src, dst in (("Fig16", "Figure8"), ("Fig17", "Figure9")):
    for ext in (".png", ".pdf"):
        open(os.path.join(paper_fig, dst + ext), "wb").write(
            open(os.path.join(sty.OUTDIR, src + ext), "rb").read()
        )
