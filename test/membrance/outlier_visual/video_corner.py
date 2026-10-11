#!/usr/bin/env python3
"""Two-panel 3D video: MIISO4 vs exprb43 from fine IGA-sampled fields.

  python3 test/membrance/outlier_visual/video_corner.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = Path(__file__).resolve().parent / "out"
EXP = OUT / "export"


def read_f64(path: Path):
    with open(path, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{nd}q", f.read(8 * nd))
        return np.frombuffer(f.read(), dtype="<f8").reshape(shape, order="F")


def _crash_info(data, prefixes):
    info = []
    for p in prefixes:
        crashed = bool(data.get(f"crash_{p}", False))
        tc = data.get(f"tcrash_{p}", None)
        if tc is None:
            tcrash = float("nan")
        else:
            tcrash = float(tc)
        info.append((crashed, tcrash))
    return info


def _render_video(U_keys, eout_keys, titles, times_keys, crash_prefixes, out_name, zlabel, vmax):
    data = json.loads((OUT / "frames.json").read_text())
    X = read_f64(EXP / "X.bin")
    Y = read_f64(EXP / "Y.bin")
    fields = []
    for key_name in U_keys:
        fields.append(read_f64(EXP / key_name))
    times = []
    for key_name in times_keys:
        times.append(read_f64(EXP / key_name))
    eouts = []
    for key_name in eout_keys:
        eouts.append(np.asarray(data[key_name], float))
    crashes = _crash_info(data, crash_prefixes)

    nF = 0
    for t_array in times:
        if t_array.size > nF:
            nF = t_array.size
    if nF == 0:
        raise RuntimeError("no frames to render")
    master_t = times[0]
    for t_array in times:
        if t_array.size > master_t.size:
            master_t = t_array

    vmax = max(float(vmax), 1e-12)
    zlim = (-1.15 * vmax, 1.15 * vmax)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = cm.coolwarm

    fig = plt.figure(figsize=(9.2, 4.6), dpi=140)
    axes = []
    for i in range(2):
        axes.append(fig.add_axes([0.04 + i * 0.42, 0.06, 0.38, 0.82], projection="3d"))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    fig.colorbar(mappable, cax=cax).set_label(zlabel)
    sup = fig.suptitle("", fontsize=12)

    path = OUT / out_name
    writer = imageio.get_writer(
        path,
        fps=8,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    print(f"rendering {nF} frames → {path}")

    for k in range(nF):
        t = float(master_t[min(k, master_t.size - 1)])
        e_line = []
        for ax, U, title, E, t_arr, (crashed, tcrash) in zip(
            axes, fields, titles, eouts, times, crashes
        ):
            ax.cla()
            n_local = U.shape[0]
            if n_local == 0:
                ax.set_axis_off()
                e_line.append("CRASH")
                continue
            kk = min(k, n_local - 1)
            Uk = U[kk]
            ax.plot_surface(
                X,
                Y,
                Uk,
                facecolors=cmap(norm(Uk)),
                linewidth=0,
                edgecolor="none",
                shade=False,
                antialiased=True,
                alpha=0.95,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(*zlim)
            try:
                ax.set_box_aspect((1, 1, 0.55))
            except Exception:
                pass
            ax.view_init(elev=28, azim=-60)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel(zlabel)
            dead = False
            if crashed:
                if k >= n_local - 1:
                    dead = True
                elif np.isfinite(tcrash) and t >= tcrash - 1e-15:
                    dead = True
            if dead:
                ax.set_title(
                    f"{title}  CRASHED\n"
                    + rf"$t_{{\mathrm{{crash}}}}={tcrash:.3g}$, $E_{{\mathrm{{out}}}}={E[kk]:.2e}$",
                    fontsize=10,
                    color="crimson",
                )
                e_line.append("CRASH")
            else:
                ax.set_title(f"{title}\n" + rf"$E_{{\mathrm{{out}}}}={E[kk]:.2e}$", fontsize=10)
                e_line.append(rf"${E[kk]:.2e}$")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        sup.set_text(
            rf"{zlabel}  corner-bump  $\alpha={data['alpha']}$  "
            rf"$t={t:.3g}$"
            rf"   $E_{{\mathrm{{out}}}}$: MIISO4 {e_line[0]}, exprb43 {e_line[1]}"
        )
        fig.canvas.draw()
        ww, hh = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(buf[: hh - (hh % 2), : ww - (ww % 2), :3].copy())

    writer.close()
    plt.close(fig)
    print(f"wrote {path}")


def main():
    data = json.loads((OUT / "frames.json").read_text())
    titles = ("MIISO4", "exprb43")
    times = ("times_miiso.bin", "times_exprb.bin")
    eouts = ("E_out_miiso", "E_out_exprb")
    prefixes = ("miiso", "exprb")

    _render_video(
        ("U_miiso.bin", "U_exprb.bin"),
        eouts,
        titles,
        times,
        prefixes,
        "solutions_u.mp4",
        r"$u$",
        data["umax"],
    )
    _render_video(
        ("Uout_miiso.bin", "Uout_exprb.bin"),
        eouts,
        titles,
        times,
        prefixes,
        "solutions_uout.mp4",
        r"$u_{\mathrm{out}}$",
        data["uomax"],
    )


if __name__ == "__main__":
    main()
