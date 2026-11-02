"""Render frozen suppression states with an orbiting 3DGS camera."""
import argparse, json, time, hashlib, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import warp as wp
from rainy_gs.scene import save_image
from rainy_gs.demo import Video
from .scenes import load, camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--frames", type=int, nargs="+")
    p.add_argument("--config", type=Path)
    p.add_argument("--clip-start", type=int)
    p.add_argument("--clip-end", type=int)
    p.add_argument("--renderer", choices=["spectral", "fierygs"], default="fierygs")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    wp.init()
    cfg = json.loads((a.input / "config.json").read_text())
    cfg.update(json.loads(a.config.read_text()) if a.config else {})
    if a.renderer == "fierygs":
        from .render_fierygs import Renderer
    else:
        from .render_spectral import Renderer
    scene, _ = load(cfg["scene"], cfg)
    geo = dict(np.load(a.input / "geometry.npz"))
    renderer = Renderer(scene, cfg, geo)
    files = sorted((a.input / "states").glob("*.npz"))
    if a.frames is not None:
        files = [p for p in files if int(p.stem) in a.frames]
    if a.clip_start is not None:
        files = [p for p in files if int(p.stem) >= a.clip_start]
    if a.clip_end is not None:
        files = [p for p in files if int(p.stem) < a.clip_end]
    if not files:
        raise ValueError("No states selected")
    contiguous = len(files) > 1 and all(
        int(b.stem) == int(a.stem) + 1 for a, b in zip(files, files[1:])
    )
    paths = []
    material = []
    video = None
    start = time.monotonic()
    complete = len(files) == cfg["frames"]
    source = a.output / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    hashes = {
        name: hashlib.sha256((a.input / name).read_bytes()).hexdigest()
        for name in ["config.json", "summary.json", "diagnostics.json", "geometry.npz"]
    }
    for path in [files[0], files[len(files) // 2], files[-1]]:
        hashes[str(path.relative_to(a.input))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (a.output / "input_hashes.json").write_text(json.dumps(hashes, indent=2))

    def read(path):
        with np.load(path) as archive:
            return dict(archive)

    pool = ThreadPoolExecutor(max_workers=2)
    pending = {i: pool.submit(read, files[i]) for i in range(min(3, len(files)))}
    for position, path in enumerate(files):
        state = pending.pop(position).result()
        if position + 3 < len(files):
            pending[position + 3] = pool.submit(read, files[position + 3])
        n = int(path.stem)
        renderer.update(state)
        cam = camera(cfg, n / max(cfg["frames"] - 1, 1), a.width)
        image, original = renderer.render(cam)
        save_image(a.output / f"{n:05d}.png", image)
        if n in [0, cfg["frames"] // 2, cfg["frames"] - 1] or a.frames is not None:
            save_image(a.output / f"original_{n:05d}.png", original)
        if a.renderer == "fierygs":
            material.append(dict(frame=n, **renderer.material_stats))
            if n in [0, 90, 210, cfg["frames"] - 1] or a.frames is not None:
                cold, _ = renderer.render_scene(cam, lighting=False)
                pristine, _ = scene.render(cam)
                save_image(a.output / f"carbon_{n:05d}.png", cold[..., :3])
                save_image(a.output / f"pristine_{n:05d}.png", pristine[..., :3])
        if contiguous:
            if video is None:
                video = Video(
                    a.output / "firex_demo.mp4", cam.width, cam.height, cfg["fps"]
                )
            video.write((image.cpu().numpy() * 255).round().astype("uint8"))
        paths.append(dict(frame=n, view=cam.view.tolist(), K=cam.K.tolist()))
        if n % 30 == 0 or not complete:
            print(
                json.dumps(
                    dict(
                        frame=n,
                        elapsed=time.monotonic() - start,
                        temperature_max=float(state["T"].max()),
                    )
                ),
                flush=True,
            )
    pool.shutdown(wait=True)
    if video:
        video.close()
    (a.output / "camera_path.json").write_text(json.dumps(paths))
    if material:
        (a.output / "material_history.json").write_text(json.dumps(material, indent=2))
        (a.output / "material_registration.json").write_text(
            json.dumps(renderer.registration, indent=2)
        )
    renderer_path = Path(__file__).with_name("render_" + a.renderer + ".py")
    (a.output / "config.json").write_text(
        json.dumps(
            dict(
                simulation=str(a.input.resolve()),
                settings=cfg,
                width=a.width,
                frames=len(files),
                complete=complete,
                renderer=a.renderer,
                selected_frames=[int(p.stem) for p in files],
                renderer_sha256=hashlib.sha256(renderer_path.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
