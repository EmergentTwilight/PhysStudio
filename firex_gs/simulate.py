"""Simulate Fire-X-style suppression and save independent replay states."""
import argparse, json, time, hashlib, shutil, zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import warp as wp
from .scenes import load, geometry
from .solver import Fire
from .water import Water


def snapshot(sim, water):
    return dict(
        T=sim.T.numpy(),
        Y=sim.Y.numpy(),
        mist=sim.mist.numpy(),
        burn=sim.burn.numpy(),
        hrr=sim.hrr.numpy(),
        wm=sim.wm.numpy(),
        wt=sim.wt.numpy(),
        char=sim.char.numpy(),
        wet=water.wet.numpy(),
        film=water.film.numpy(),
        x=water.x.numpy()[: water.count],
        v=water.v.numpy()[: water.count],
        mass=water.mass.numpy()[: water.count],
        time=sim.time,
    )


def save_state(path, data):
    temporary = path.with_suffix(".partial")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for name, array in data.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="firewoods")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dt", type=float, default=1 / 120)
    p.add_argument("--warmup", type=float)
    p.add_argument("--config", type=Path)
    p.add_argument("--control", choices=["water", "dry", "top"], default="water")
    p.add_argument("--save-every", type=int, default=1)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    overrides = json.loads(a.config.read_text()) if a.config else {}
    scene, cfg = load(a.scene, overrides)
    geo, meta = geometry(scene, cfg)
    del scene
    torch.cuda.empty_cache()
    sim = Fire(cfg, geo)
    water = Water(sim)
    sub = int(np.ceil(1 / (a.fps * a.dt)))
    dt = 1 / (a.fps * sub)
    warmup = cfg["warmup"] if a.warmup is None else a.warmup
    cfg.update(
        scene=a.scene,
        frames=a.frames,
        fps=a.fps,
        effective_dt=dt,
        substeps=sub,
        warmup=warmup,
        control=a.control,
        geometry=meta,
        source_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    (a.output / "config.json").write_text(json.dumps(cfg, indent=2))
    np.savez_compressed(a.output / "geometry.npz", **geo)
    source = a.output / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    states = a.output / "states"
    states.mkdir()
    records = []
    start = time.monotonic()
    writers = ThreadPoolExecutor(max_workers=6)
    pending = deque()
    for step in range(round(warmup / dt)):
        sim.step(dt)
        if step % 120 == 0:
            d = sim.diagnostics()
            if d["velocity_max"] > 40:
                raise RuntimeError("Excessive velocity in warmup")
            d.update(stage="warmup", step=step, elapsed=time.monotonic() - start)
            print(json.dumps(d), flush=True)
    for frame in range(a.frames):
        for _ in range(sub):
            sim.step(dt)
            water.step(
                dt,
                frame / a.fps,
                enabled=a.control != "dry",
                aim="top" if a.control == "top" else "base",
            )
        if frame % a.save_every == 0 or frame == a.frames - 1:
            if len(pending) >= 12:
                pending.popleft().result()
            pending.append(
                writers.submit(
                    save_state, states / f"{frame:05d}.npz", snapshot(sim, water)
                )
            )
        if frame % 15 == 0 or frame == a.frames - 1:
            d = sim.diagnostics()
            d.update(water.diagnostics())
            d.update(frame=frame, elapsed=time.monotonic() - start)
            records.append(d)
            print(json.dumps(d), flush=True)
            if d["velocity_max"] > 40:
                raise RuntimeError("Excessive velocity in rollout")
            if not np.isfinite(d["temperature_max"]) or d["temperature_max"] > 10000:
                raise RuntimeError("Invalid gas temperature")
            if d["species_min"] < -0.00001 or d["species_sum_error"] > 0.001:
                raise RuntimeError("Invalid gas composition")
            (a.output / "status.json").write_text(json.dumps(d, indent=2))
            (a.output / "diagnostics.json").write_text(json.dumps(records, indent=2))
    for future in pending:
        future.result()
    writers.shutdown(wait=True)
    (a.output / "summary.json").write_text(
        json.dumps(
            dict(complete=True, seconds=time.monotonic() - start, final=records[-1]),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
