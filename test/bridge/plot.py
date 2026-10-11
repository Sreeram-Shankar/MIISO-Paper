#!/usr/bin/env python3
"""Bridge visuals from exported bins (displacement MP4, energy MP4/PNG, HTML).

Full window: export from frames_full (t=0 → 11.5), then:

  python3 test/bridge/plot.py              # all
  python3 test/bridge/plot.py displacement
  python3 test/bridge/plot.py energy
  python3 test/bridge/plot.py energy-still
  python3 test/bridge/plot.py html

Playback fps follows the save interval so video length ≈ sim window (11.5 s).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource, LogNorm, Normalize
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent / "out_bridge_nel3266"
EXP = ROOT / "export"
ENERGY = ROOT / "energy"
T_WINDOW = 11.5


def read_f64(path: Path):
    with open(path, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{nd}q", f.read(8 * nd))
        return np.frombuffer(f.read(), dtype="<f8").reshape(shape, order="F")


def mat(A):
    return np.asarray(A, dtype=np.float32).round(decimals=4).tolist()


def playback_fps(times, fallback=20.0):
    """Map sim time → wall video: fps = 1/median(Δt) so duration ≈ T_WINDOW."""
    t = np.asarray(times, dtype=float).ravel()
    if t.size >= 2:
        dt = float(np.median(np.diff(t)))
        if dt > 0:
            return 1.0 / dt
    if t.size >= 1 and T_WINDOW > 0:
        return max(t.size / T_WINDOW, 1.0)
    return fallback


def _limits(geom, faces, extra=None):
    xs = [geom[f][0].ravel() for f in faces]
    ws = [geom[f][1].ravel() for f in faces]
    hs = [geom[f][2].ravel() for f in faces]
    if extra is not None:
        xs.append(extra[0].ravel()); ws.append(extra[1].ravel()); hs.append(extra[2].ravel())
    xs, ws, hs = map(np.concatenate, (xs, ws, hs))
    pad = 0.08
    xlim = (xs.min() - pad * np.ptp(xs), xs.max() + pad * np.ptp(xs))
    wlim = (ws.min() - pad * np.ptp(ws), ws.max() + pad * np.ptp(ws))
    hlim = (hs.min() - pad * np.ptp(hs), hs.max() + pad * np.ptp(hs))
    aspect = (np.ptp(xs), 2.0 * np.ptp(ws), np.ptp(hs))
    return xlim, wlim, hlim, aspect


def plot_displacement():
    info = dict(line.strip().split("=", 1) for line in open(EXP / "info.txt"))
    nF, cmax = int(info["nF"]), float(info["cmax"])
    faces = [s.strip() for s in info["faces"].split(",") if s.strip()]
    times = read_f64(EXP / "times.bin")
    wires = read_f64(EXP / "wires.bin")
    nh = sum(1 for n in faces if n.startswith("h"))
    nw = sum(1 for n in faces if n.startswith("w"))
    outer = {f"h1", f"h{nh}", f"w1", f"w{nw}", "e0", "e1"}
    geom = {}
    for f in faces:
        X, Y, Z = read_f64(EXP / f"X_{f}.bin"), read_f64(EXP / f"Y_{f}.bin"), read_f64(EXP / f"Z_{f}.bin")
        geom[f] = (X, Z, Y)
    C = {f: read_f64(EXP / f"C_{f}.bin") for f in faces}
    up = 3
    for f in faces:
        geom[f] = tuple(zoom(g, up, order=3) for g in geom[f])
    Wspan, Wwidth, Wheight = wires[:, 0, :], wires[:, 2, :], wires[:, 1, :]
    xlim, wlim, hlim, aspect = _limits(geom, faces, (Wspan, Wwidth, Wheight))
    norm = Normalize(vmin=0.0, vmax=max(cmax, 1e-16))
    cmap = cm.turbo
    ls = LightSource(azdeg=315, altdeg=45)
    out = ROOT / "bridge_displacement.mp4"
    fps = playback_fps(times)
    t0, t1 = float(times[0]), float(times[-1])
    fig = plt.figure(figsize=(14.0, 8.75), dpi=180)
    ax = fig.add_axes([0.02, 0.02, 0.86, 0.94], projection="3d")
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.915, 0.18, 0.015, 0.64])
    fig.colorbar(mappable, cax=cax).set_label("|u| [m]")
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=None,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"rendering {nF} displacement frames  t={t0:.2f}→{t1:.2f}s  fps={fps:.2f} → {out}")
    for k in range(nF):
        ax.cla()
        for f in sorted(faces, key=lambda n: 0 if n not in outer else 1):
            Xp, Wp, Hp = geom[f]
            Ck = zoom(C[f][k], up, order=3)
            rgb = ls.shade_rgb(cmap(norm(Ck)), Hp, blend_mode="soft")
            ax.plot_surface(Xp, Wp, Hp, facecolors=rgb,
                            linewidth=0, edgecolor="none", shade=False, antialiased=True,
                            alpha=0.95 if f in outer else 0.15)
        for i in range(wires.shape[0]):
            ax.plot(Wspan[i], Wwidth[i], Wheight[i],
                    color="white", lw=0.7, alpha=0.9, solid_capstyle="round")
        ax.set_xlim(*xlim); ax.set_ylim(*wlim); ax.set_zlim(*hlim)
        try:
            ax.set_box_aspect(aspect)
        except Exception:
            pass
        az = -90 + 22 * np.sin(2 * np.pi * k / max(nF - 1, 1))
        el = 36 + 4 * np.cos(2 * np.pi * k / max(nF - 1, 1))
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel("span x [m]"); ax.set_ylabel("width z [m]"); ax.set_zlabel("height y [m]")
        ax.set_title(f"continuum arch  |u|  t = {times[k]:.2f} / {T_WINDOW:.1f} s", fontsize=11)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.tick_params(labelsize=8)
        fig.canvas.draw()
        ww, hh = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(buf[: hh - (hh % 2), : ww - (ww % 2), :3].copy())
        if (k + 1) % 40 == 0 or k == 0:
            print(f"  {k+1}/{nF}")
    writer.close(); plt.close(fig)
    print(f"wrote {out}")


def plot_energy_mp4():
    info = dict(line.strip().split("=", 1) for line in open(ENERGY / "info.txt"))
    nF = int(info["nF"]); ψmax = float(info["psi_max"])
    faces = [s.strip() for s in info["faces"].split(",") if s.strip()]
    times = read_f64(ENERGY / "times.bin")
    geom, Psi = {}, {}
    for f in faces:
        X, Y, Z = read_f64(ENERGY / f"X_{f}.bin"), read_f64(ENERGY / f"Y_{f}.bin"), read_f64(ENERGY / f"Z_{f}.bin")
        geom[f] = (X, Z, Y)
        Psi[f] = np.maximum(read_f64(ENERGY / f"P_{f}.bin"), 1e-30)
    xlim, wlim, hlim, aspect = _limits(geom, faces)
    vmin = max(ψmax * 1e-4, min(Psi[f].min() for f in faces))
    norm = LogNorm(vmin=vmin, vmax=max(ψmax, vmin * 10))
    cmap = cm.inferno
    wires = read_f64(ENERGY / "wires.bin") if (ENERGY / "wires.bin").is_file() else None
    out = ROOT / "bridge_strain_energy.mp4"
    fig = plt.figure(figsize=(11.5, 7.0), dpi=130)
    ax = fig.add_axes([0.02, 0.02, 0.86, 0.92], projection="3d")
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    fig.colorbar(mappable, cax=cax).set_label("ψ [J/m³]  (log)")
    fps = playback_fps(times)
    t0, t1 = float(times[0]), float(times[-1])
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=None,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"rendering {nF} energy frames  t={t0:.2f}→{t1:.2f}s  fps={fps:.2f} → {out}")
    for k in range(nF):
        ax.cla()
        for f in faces:
            Xp, Wp, Hp = geom[f]
            ax.plot_surface(Xp, Wp, Hp, facecolors=cmap(norm(Psi[f][k])),
                            linewidth=0, edgecolor="none", shade=False, antialiased=True, alpha=0.95)
        if wires is not None:
            for i in range(wires.shape[0]):
                ax.plot(wires[i, 0], wires[i, 2], wires[i, 1], color="white", lw=0.6, alpha=0.65)
        ax.set_xlim(*xlim); ax.set_ylim(*wlim); ax.set_zlim(*hlim)
        try:
            ax.set_box_aspect(aspect)
        except Exception:
            pass
        az = -90 + 22 * np.sin(2 * np.pi * k / max(nF - 1, 1))
        el = 36 + 4 * np.cos(2 * np.pi * k / max(nF - 1, 1))
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel("span x [m]"); ax.set_ylabel("width z [m]"); ax.set_zlabel("height y [m]")
        ax.set_title(f"SVK strain energy ψ  t = {times[k]:.2f} / {T_WINDOW:.1f} s", fontsize=12)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        fig.canvas.draw()
        ww, hh = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(buf[: hh - (hh % 2), : ww - (ww % 2), :3].copy())
        if (k + 1) % 40 == 0 or k == 0:
            print(f"  {k+1}/{nF}")
    writer.close(); plt.close(fig)
    print(f"wrote {out}")


def plot_energy_still():
    info = dict(line.strip().split("=", 1) for line in open(ENERGY / "info.txt"))
    t = float(info.get("t", info.get("T", 0.0)))
    # prefer last-frame time from times.bin if present
    tpath = ENERGY / "times.bin"
    if tpath.is_file():
        times = read_f64(tpath)
        t = float(times[-1])
        k = -1
    else:
        k = None
    ψmax = float(info["psi_max"])
    faces = [s.strip() for s in info["faces"].split(",") if s.strip()]
    geom, Psi = {}, {}
    for f in faces:
        X, Y, Z = read_f64(ENERGY / f"X_{f}.bin"), read_f64(ENERGY / f"Y_{f}.bin"), read_f64(ENERGY / f"Z_{f}.bin")
        geom[f] = (X, Z, Y)
        P = np.maximum(read_f64(ENERGY / f"P_{f}.bin"), 1e-30)
        Psi[f] = P[k] if P.ndim == 3 else P
    xlim, wlim, hlim, aspect = _limits(geom, faces)
    vmin = max(ψmax * 1e-4, min(Psi[f].min() for f in faces))
    norm = LogNorm(vmin=vmin, vmax=ψmax)
    cmap = cm.inferno
    out = ROOT / "bridge_strain_energy.png"
    fig = plt.figure(figsize=(11.5, 7.0), dpi=150)
    ax = fig.add_axes([0.02, 0.02, 0.86, 0.92], projection="3d")
    for f in faces:
        Xp, Wp, Hp = geom[f]
        ax.plot_surface(Xp, Wp, Hp, facecolors=cmap(norm(Psi[f])),
                        linewidth=0, edgecolor="none", shade=False, antialiased=True, alpha=0.95)
    wpath = ENERGY / "wires.bin"
    if wpath.is_file():
        W = read_f64(wpath)
        for i in range(W.shape[0]):
            ax.plot(W[i, 0], W[i, 2], W[i, 1], color="white", lw=0.7, alpha=0.7)
    ax.set_xlim(*xlim); ax.set_ylim(*wlim); ax.set_zlim(*hlim)
    try:
        ax.set_box_aspect(aspect)
    except Exception:
        pass
    ax.view_init(elev=36, azim=-90)
    ax.set_xlabel("span x [m]"); ax.set_ylabel("width z [m]"); ax.set_zlabel("height y [m]")
    ax.set_title(f"SVK strain-energy density ψ  t = {t:.2f} s   max = {ψmax:.3e} J/m³", fontsize=11)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label("ψ [J/m³]  (log)")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_html():
    info = dict(line.strip().split("=", 1) for line in open(EXP / "info.txt"))
    nF, cmax = int(info["nF"]), float(info["cmax"])
    T = float(info["T"])
    faces = [s.strip() for s in info["faces"].split(",") if s.strip()]
    times = read_f64(EXP / "times.bin")
    nh = sum(1 for f in faces if f.startswith("h"))
    nw = sum(1 for f in faces if f.startswith("w"))
    geom = {}
    for f in faces:
        X, Y, Z = read_f64(EXP / f"X_{f}.bin"), read_f64(EXP / f"Y_{f}.bin"), read_f64(EXP / f"Z_{f}.bin")
        geom[f] = (X, Z, Y)
    C = {f: read_f64(EXP / f"C_{f}.bin") for f in faces}
    colorscale = [
        [0.0, "#000004"], [0.13, "#1b0c41"], [0.25, "#4a0c6b"],
        [0.38, "#781c6d"], [0.5, "#a52c60"], [0.63, "#cf4446"],
        [0.75, "#ed6925"], [0.88, "#fb9b06"], [1.0, "#fcffa4"],
    ]

    def opacity(name):
        if name in (f"h1", f"h{nh}", f"w1", f"w{nw}") or name.startswith("e"):
            return 0.95
        return 0.32

    frames = [{"t": float(times[k]), "C": {f: mat(C[f][k]) for f in faces}} for k in range(nF)]
    payload = {
        "T": T, "cmax": cmax, "nel": info["nel"], "nF": nF,
        "times": [float(t) for t in times], "faces": faces,
        "opacity": {f: opacity(f) for f in faces},
        "geom": {f: {"X": mat(geom[f][0]), "Y": mat(geom[f][1]), "Z": mat(geom[f][2])} for f in faces},
        "frames": frames, "colorscale": colorscale,
    }
    data_path = ROOT / "bridge_video_data.json"
    with open(data_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {data_path} ({data_path.stat().st_size/1e6:.1f} MB)")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bridge continuum |u|</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #0e0e12; color: #eee; }}
  header {{ padding: 10px 16px; border-bottom: 1px solid #2a2a35; display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline; }}
  header h1 {{ font-size: 1.05rem; font-weight: 600; margin: 0; }}
  header .meta {{ color: #9aa; font-size: 0.85rem; }}
  #plot3d {{ width: 100%; height: calc(100vh - 120px); }}
  .controls {{ position: sticky; bottom: 0; background: #16161d; border-top: 1px solid #2a2a35; padding: 12px 16px 16px; display: grid; gap: 8px; }}
  .row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  button {{ background: #2b2b3a; color: #fff; border: 1px solid #3d3d52; border-radius: 8px; padding: 8px 14px; cursor: pointer; }}
  input[type=range] {{ flex: 1; min-width: 200px; accent-color: #fb9b06; }}
  #timeLabel {{ font-variant-numeric: tabular-nums; font-size: 1.05rem; min-width: 9rem; color: #fcffa4; }}
  #playRate {{ width: 5rem; }}
</style>
</head>
<body>
<header>
  <h1>IGA continuum arch — |u| (η / ζ layers)</h1>
  <div class="meta">nel={info['nel']} · T={T}s · frames={nF} · faces={len(faces)}</div>
</header>
<div id="plot3d"></div>
<div class="controls">
  <div class="row">
    <button id="btnPlay" type="button">Play</button>
    <label>speed <input id="playRate" type="number" value="1" min="0.1" max="10" step="0.1"/> ×</label>
    <span id="timeLabel">t = 0.00 s</span>
  </div>
  <div class="row">
    <span>0</span>
    <input id="slider" type="range" min="0" max="{nF-1}" value="0" step="1"/>
    <span>{T}s</span>
  </div>
</div>
<script>
fetch('bridge_video_data.json')
  .then(r => r.json())
  .then(D => {{
    const faces = D.faces;
    let i = 0, playing = false, timer = null;
    function traces(frame) {{
      return faces.map((f, idx) => ({{
        type: 'surface', x: D.geom[f].X, y: D.geom[f].Y, z: D.geom[f].Z,
        surfacecolor: frame.C[f], cmin: 0, cmax: D.cmax, colorscale: D.colorscale,
        showscale: idx === 0, colorbar: {{ title: '|u| [m]', len: 0.55 }},
        opacity: D.opacity[f], hoverinfo: 'skip', name: f
      }}));
    }}
    function layout(t) {{
      return {{
        paper_bgcolor: '#0e0e12', font: {{ color: '#ddd' }},
        margin: {{ l: 0, r: 0, t: 28, b: 0 }},
        title: {{ text: 'continuum · |u| · t=' + t.toFixed(2) + 's', font: {{ size: 14 }} }},
        scene: {{
          xaxis: {{ title: 'span x [m]', gridcolor: '#333', backgroundcolor: '#0e0e12', showbackground: true }},
          yaxis: {{ title: 'width z [m]', gridcolor: '#333', backgroundcolor: '#0e0e12', showbackground: true }},
          zaxis: {{ title: 'height y [m]', gridcolor: '#333', backgroundcolor: '#0e0e12', showbackground: true }},
          aspectmode: 'manual', aspectratio: {{ x: 5.0, y: 1.35, z: 1.15 }},
          camera: {{ eye: {{ x: 0.05, y: -1.25, z: 0.72 }}, up: {{ x: 0, y: 0, z: 1 }} }}
        }}
      }};
    }}
    Plotly.newPlot('plot3d', traces(D.frames[0]), layout(D.frames[0].t), {{ responsive: true }});
    const slider = document.getElementById('slider');
    const timeLabel = document.getElementById('timeLabel');
    const btn = document.getElementById('btnPlay');
    const rate = document.getElementById('playRate');
    function show(k) {{
      i = k; const f = D.frames[k];
      slider.value = k; timeLabel.textContent = 't = ' + f.t.toFixed(2) + ' s';
      Plotly.restyle('plot3d', {{ surfacecolor: faces.map(name => f.C[name]) }});
      Plotly.relayout('plot3d', {{ 'title.text': 'continuum · |u| · t=' + f.t.toFixed(2) + 's' }});
    }}
    slider.addEventListener('input', () => show(+slider.value));
    function tick() {{
      if (!playing) return;
      show((i + 1) % D.frames.length);
      timer = setTimeout(tick, Math.max(20, 50 / Math.max(0.1, +rate.value)));
    }}
    btn.addEventListener('click', () => {{
      playing = !playing; btn.textContent = playing ? 'Pause' : 'Play';
      if (playing) tick(); else if (timer) clearTimeout(timer);
    }});
    document.addEventListener('keydown', (e) => {{
      if (e.key === ' ') {{ e.preventDefault(); btn.click(); }}
      if (e.key === 'ArrowRight') show(Math.min(i + 1, D.frames.length - 1));
      if (e.key === 'ArrowLeft') show(Math.max(i - 1, 0));
    }});
  }})
  .catch(err => {{
    document.body.innerHTML = '<pre style="padding:2rem;color:#f66">Failed to load data\\n' + err + '</pre>';
  }});
</script>
</body>
</html>
"""
    (ROOT / "bridge_video.html").write_text(html)
    print(f"wrote {ROOT / 'bridge_video.html'}")


def main(argv=None):
    which = (argv or sys.argv[1:]) or ["displacement", "energy", "energy-still", "html"]
    for w in which:
        {
            "displacement": plot_displacement,
            "energy": plot_energy_mp4,
            "energy-still": plot_energy_still,
            "html": plot_html,
        }[w]()


if __name__ == "__main__":
    main()
