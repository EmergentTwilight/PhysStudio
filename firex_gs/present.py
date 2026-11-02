"""Build a local video gallery, paired comparison, and scientific diagnostics."""
import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/firex_matplotlib")
)
try:
    import matplotlib
except ModuleNotFoundError:
    # Plotting can use a separate existing Python without changing the GPU environment.
    if __name__ == "__main__":
        checked = {Path(sys.executable).resolve()}
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            candidate = Path(directory) / "python"
            if not candidate.is_file() or candidate.resolve() in checked:
                continue
            checked.add(candidate.resolve())
            check = subprocess.run(
                [str(candidate), "-c", "import matplotlib,numpy"], capture_output=True
            )
            if check.returncode == 0:
                os.execv(
                    str(candidate),
                    [str(candidate), "-m", "firex_gs.present", *sys.argv[1:]],
                )
    raise
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    a = p.parse_args()
    root = a.input.resolve()
    audit = json.loads((root / "audit.json").read_text())
    cfg = json.loads((root / "simulation/config.json").read_text())
    wet = json.loads((root / "simulation/diagnostics.json").read_text())
    dry = (
        json.loads((root / "dry_simulation/diagnostics.json").read_text())
        if "dry" in audit
        else None
    )
    fps = cfg["fps"]
    t = np.array([r["frame"] / fps for r in wet])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    styles = [(wet, "Water suppression", "#147f9a")]
    if dry:
        styles.append((dry, "No water", "#d45b43"))
    for data, label, color in styles:
        ts = [r["frame"] / fps for r in data]
        axes[0, 0].plot(
            ts, [r["heat_release_kw"] for r in data], label=label, color=color, lw=2
        )
        axes[0, 1].plot(
            ts,
            [r["wood_temperature_mean"] - 273.15 for r in data],
            label=label,
            color=color,
            lw=2,
        )
    for key, label, color in [
        ("water_emitted_kg", "Emitted", "#172b4d"),
        ("water_remaining_kg", "Moving liquid", "#147f9a"),
        ("water_retained_kg", "Retained water", "#668550"),
        ("water_evaporated_kg", "Evaporated", "#bf885b"),
        ("water_escaped_kg", "Escaped", "#888888"),
    ]:
        axes[1, 0].plot(t, [r[key] for r in wet], label=label, color=color, lw=1.8)
    axes[1, 1].plot(
        t,
        [r["water_mass_balance_error_kg"] * 1e6 for r in wet],
        color="#52688f",
        lw=1.8,
    )
    titles = [
        "Combustion heat release",
        "Mean wood temperature",
        "Water mass accounting",
        "Water mass residual",
    ]
    units = ["kW", "°C", "kg", "mg"]
    for ax, title, unit in zip(axes.flat, titles, units):
        ax.set(title=title, xlabel="Video time (s)", ylabel=unit)
        ax.axvspan(
            cfg["water_start"],
            min(cfg["water_stop"], t[-1]),
            color="#147f9a",
            alpha=0.07,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.16)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(cfg["title"], fontsize=14)
    fig.savefig(root / "diagnostics.png", dpi=180)
    fig.savefig(root / "diagnostics.pdf")
    plt.close(fig)
    if "dry_video" in audit:
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        draw = (
            lambda title: f"drawbox=x=0:y=0:w=iw:h=44:color=black@0.65:t=fill,drawtext=fontfile={font}:text='{title}':x=18:y=12:fontsize=20:fontcolor=white"
        )
        filters = f"[0:v]scale=960:-2,{draw('WATER SUPPRESSION')}[a];[1:v]scale=960:-2,{draw('NO WATER - MATCHED CONTROL')}[b];[a][b]hstack=inputs=2[v]"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(root / "render/firex_demo.mp4"),
                "-i",
                str(root / "dry_render/firex_demo.mp4"),
                "-filter_complex",
                filters,
                "-map",
                "[v]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(root / "comparison.mp4"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(root / "comparison.mp4"),
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
        )
    title = html.escape(cfg["title"])
    water = audit["water"]
    comparison = ""
    if dry:
        reduction = audit.get("post_water_reduction_vs_dry")
        percent = f"{100*reduction:.1f}%" if reduction is not None else "N/A"
        control_video = (
            '<video src="comparison.mp4" controls loop playsinline preload="metadata"></video>'
            if "dry_video" in audit
            else ""
        )
        comparison = f'<section><h2>Matched control</h2>{control_video}<p>Mean heat-release reduction after the spray stops: <b>{percent}</b>. Final heat release: <b>{water["final_heat_release_kw"]:.2f} kW</b> with water, <b>{audit["dry"]["final_heat_release_kw"]:.2f} kW</b> without water.</p></section>'
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fire-X GS | {title}</title><style>
body{{margin:0;background:#111416;color:#e7e9e9;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1280px;margin:45px auto;padding:0 24px}}h1{{font-size:42px;letter-spacing:-1.5px;margin:0}}h2{{font-size:23px}}p{{max-width:1000px;color:#b6c1c4}}a{{color:#8dc8d7}}video,img{{width:100%;border-radius:8px;display:block}}section{{margin:36px 0 55px}}.tag{{color:#80c1b1;font-size:12px;letter-spacing:2px;text-transform:uppercase}}.stats{{display:flex;gap:40px;flex-wrap:wrap;margin:20px 0}}.stats b{{font-size:25px;display:block;color:#fff}}
</style><main><span class="tag">Scene-level thermochemical water suppression</span><h1>{title}</h1><p>A reconstructed Gaussian environment with Warp gas / spray coupling and FieryGS surface carbonization. A finite combustible heat reservoir, water heating, evaporation and vapor dilution determine the fire response.</p><section><video src="render/firex_demo.mp4" poster="render/00000.png" controls autoplay muted loop playsinline></video></section>{comparison}<section><h2>Physical diagnostics</h2><div class="stats"><span><b>{water['frames']}</b>saved frames</span><span><b>{water['final']['wood_temperature_mean']-273.15:.1f} °C</b>final mean fuel temperature</span><span><b>{water['water_mass_error_kg']*1e6:.2f} mg</b>maximum water accounting error</span></div><img src="diagnostics.png" alt="Heat release, fuel temperature and water accounting"><p>Shaded interval marks nozzle operation. Plotted values come from the simulation state, independent of the renderer. <a href="audit.json">Audit JSON</a> · <a href="diagnostics.pdf">Figure PDF</a></p></section><p><a href="https://helgewrede.github.io/firex/">Fire-X (SIGGRAPH Asia 2025)</a> · <a href="https://pku-vcl-geometry.github.io/FieryGS/">FieryGS (ICLR 2026)</a></p></main></html>"""
    (root / "index.html").write_text(page)
    print(root / "index.html")


if __name__ == "__main__":
    main()
