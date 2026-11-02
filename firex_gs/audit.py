"""Validate saved suppression states and compare the paired dry control."""
import argparse, json, subprocess
from pathlib import Path
import numpy as np


def trajectory(path):
    cfg = json.loads((path / "config.json").read_text())
    records = json.loads((path / "diagnostics.json").read_text())
    assert json.loads((path / "summary.json").read_text())["complete"]
    files = sorted((path / "states").glob("*.npz"))
    assert len(files) == cfg["frames"]
    assert [int(p.stem) for p in files] == list(range(cfg["frames"]))
    assert max(r["species_sum_error"] for r in records) < 1e-5
    assert min(r["species_min"] for r in records) > -1e-6
    assert max(r["velocity_max"] for r in records) < 40
    assert max(abs(r["water_mass_balance_error_kg"]) for r in records) < max(
        records[-1]["water_emitted_kg"] * 1e-4, 1e-5
    )
    for p in [files[0], files[len(files) // 2], files[-1]]:
        with np.load(p) as a:
            assert all(np.isfinite(a[k]).all() for k in a.files), p
            assert a["mass"].min(initial=0) >= -1e-10, p
    post = [
        r["heat_release_kw"]
        for r in records
        if r["frame"] / cfg["fps"] >= cfg["water_stop"] + 1
    ]
    start = [
        r["heat_release_kw"]
        for r in records
        if r["frame"] / cfg["fps"] < cfg["water_start"]
    ]
    return dict(
        passed=True,
        frames=len(files),
        initial_heat_release_kw=float(np.mean(start)) if start else None,
        final_heat_release_kw=records[-1]["heat_release_kw"],
        post_water_heat_release_kw=float(np.mean(post)) if post else None,
        max_air_speed=max(r["velocity_max"] for r in records),
        water_mass_error_kg=max(abs(r["water_mass_balance_error_kg"]) for r in records),
        final=records[-1],
    )


def video(path, frames):
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    assert int(data["nb_frames"]) == frames
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    a = p.parse_args()
    result = {"water": trajectory(a.input / "simulation")}
    result["video"] = video(
        a.input / "render/firex_demo.mp4", result["water"]["frames"]
    )
    if (a.input / "dry_simulation").exists():
        result["dry"] = trajectory(a.input / "dry_simulation")
        if (a.input / "dry_render/firex_demo.mp4").exists():
            result["dry_video"] = video(
                a.input / "dry_render/firex_demo.mp4", result["dry"]["frames"]
            )
        water_cfg = json.loads((a.input / "simulation/config.json").read_text())
        dry_cfg = json.loads((a.input / "dry_simulation/config.json").read_text())
        assert {
            k: v for k, v in water_cfg.items() if k not in ["control", "source_hashes"]
        } == {k: v for k, v in dry_cfg.items() if k not in ["control", "source_hashes"]}
        for name in [
            "simulate.py",
            "solver.py",
            "chemistry.py",
            "water.py",
            "wood_thermal.py",
            "scenes.py",
        ]:
            assert (
                water_cfg["source_hashes"][name] == dry_cfg["source_hashes"][name]
            ), name
        initial_error = {}
        with np.load(a.input / "simulation/states/00000.npz") as water, np.load(
            a.input / "dry_simulation/states/00000.npz"
        ) as dry:
            for key in ["T", "Y", "wm", "wt"]:
                initial_error[key] = float(np.max(np.abs(water[key] - dry[key])))
                assert np.allclose(water[key], dry[key], rtol=1e-5, atol=1e-5), key
        result["initial_state_max_error"] = initial_error
        wet = result["water"]["post_water_heat_release_kw"]
        dry = result["dry"]["post_water_heat_release_kw"]
        if wet is not None and dry is not None and dry > 0:
            result["post_water_reduction_vs_dry"] = 1.0 - wet / dry
        result["final_reduction_vs_dry"] = (
            1.0
            - result["water"]["final_heat_release_kw"]
            / result["dry"]["final_heat_release_kw"]
        )
    (a.input / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
