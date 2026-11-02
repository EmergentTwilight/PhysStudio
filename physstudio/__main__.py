"""Run the curated PhysStudio demos from portable presets."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .paths import ROOT, data_path


SCENES = {
    "rain": ("playground", "firewoods", "train"),
    "splashing": ("liquify", "wave", "plunge"),
    "enlivening": ("bread", "wolf", "vasedeck"),
    "magnetic": ("ferrofluid", "magnetic_bread", "magnetic_wolf"),
    "snow": ("playground", "firewoods", "train"),
    "lava": ("zohar", "lands_end", "calico"),
    "water_sand": ("castle", "sediment", "breach"),
    "wilting": ("potted_plant", "ficus", "alocasia"),
    "storm": ("playground", "lands_end", "calico"),
    "thunder": ("firewoods", "zohar", "train"),
    "capillary": ("orchard", "capillary_cluster", "paperclips"),
    "ice": ("cubes", "arch", "bunny"),
    "firex": ("firewoods", "playground", "bonsai"),
}


def command(module, **options):
    result = [sys.executable, "-m", module]
    for key, value in options.items():
        if value is None or value is False:
            continue
        result.append("--" + key.replace("_", "-"))
        if value is not True:
            result.extend(
                map(str, value if isinstance(value, (list, tuple)) else [value])
            )
    return result


def plan(args):
    out = args.output
    if args.method == "firex":
        return [command("firex_gs.harness", scene=args.scene, output=out,
                        gpu=args.gpu, frames=args.frames, width=args.width)]
    if args.method == "ice":
        cmd = command("ice_melting.harness", scene=args.scene, output=out,
                      gpu=args.gpu, frames=args.frames, width=args.width)
        cmd.insert(3, "run")
        return [cmd]
    if args.method == "capillary":
        cmd = command("surface_tension.harness", scene=args.scene, output=out,
                      gpu=args.gpu, frames=args.frames, width=args.width,
                      stirrer=args.scene != "orchard")
        cmd.insert(3, "run")
        return [cmd]
    if args.method == "thunder":
        return [
            command("thunder_gs.simulate", scene=args.scene, output=out / "simulation"),
            command("thunder_gs.demo", simulation=out / "simulation", output=out / "render",
                    frames=args.frames or 240, width=args.width or 1280),
        ]
    if args.method == "storm":
        return [command("storm_gs.demo", scene=args.scene, output=out,
                        frames=args.frames or 240, width=args.width or 1280,
                        warmup=8 if args.scene == "calico" else 5, dt=.012)]
    if args.method == "wilting":
        return [command("plant_wilting.demo", scene=args.scene, output=out,
                        frames=args.frames, width=args.width)]
    if args.method == "rain":
        opts = json.loads(
            (ROOT / "rainy_gs/configs" / (args.scene + "_demo.json")).read_text()
        )
        source = {
            "playground": "fierygs/Playground",
            "firewoods": "fierygs/firewoods_sand_dark",
            "train": "voxel51/train",
        }[args.scene]
        opts.update(scene=data_path("rainy_gs", source), output=out)
        if args.scene != "playground":
            opts["scene_config"] = ROOT / "rainy_gs/configs" / (args.scene + ".json")
        if args.frames is not None:
            opts["frames"] = args.frames
        if args.width is not None:
            opts["width"] = args.width
        return [command("rainy_gs.demo", **opts)]
    if args.method == "splashing":
        width = args.width or 1280
        return [
            command(
                "gaussian_splashing.demo",
                scene=args.scene,
                output=out,
                frames=args.frames or 180,
                width=width,
                height=round(width * 5 / 16) * 2,
            )
        ]
    if args.method == "enlivening":
        cmds = []
        if not data_path("enlivening_gs", args.scene, "gaussians.npz").exists():
            cmds.append(command("enlivening_gs.assets", scene=args.scene))
        vase = args.scene == "vasedeck"
        cmds.append(
            command(
                "enlivening_gs.locomotion",
                scene=args.scene,
                output=out,
                motion="turn" if args.scene == "wolf" else "hop",
                seconds=2.1 if args.frames is None else args.frames / 60,
                hz=120,
                resolution=7,
                young=30000 if vase else 18000,
                density=100 if vase else 40,
                mass_profile="base-heavy" if vase else "uniform",
            )
        )
        width = args.width or 960
        cmds.append(
            command(
                "enlivening_gs.render",
                input=out,
                output=out / "render",
                width=width,
                height=round(width * 3 / 8) * 2,
                split=True,
                orbit=24,
            )
        )
        return cmds
    if args.method == "magnetic":
        from magnetic_mpm.harness import PRESETS

        opts = PRESETS[args.scene].copy()
        if args.frames is not None:
            opts["frames"] = args.frames
        width = args.width or 960
        return [
            command("magnetic_mpm.demo", output=out, **opts),
            command(
                "magnetic_mpm.render",
                input=out,
                output=out / "render",
                width=width,
                height=round(width * 3 / 8) * 2,
                orbit=22,
            ),
            [
                sys.executable,
                "-m",
                "magnetic_mpm.harness",
                "audit",
                "--input",
                str(out),
            ],
        ]
    if args.method == "water_sand":
        return [
            command(
                "water_sand.demo",
                scenes=[args.scene],
                output=out,
                frames=args.frames,
                width=args.width,
            )
        ]
    if args.method == "lava":
        return [
            command(
                "lava_gs.demo",
                scene=args.scene,
                output=out,
                frames=args.frames,
                width=args.width,
                cache_states=True,
                save_frames=True,
            ),
            [sys.executable, "-m", "lava_gs.audit", str(out)],
        ]
    opts = dict(
        scene=args.scene,
        output=out,
        frames=args.frames or 240,
        width=args.width or (1512 if args.scene == "playground" else 1280),
        warmup=2,
        haze=0.002 if args.scene == "firewoods" else 0.003,
    )
    return [
        command("snowy_gs.demo", **opts),
        [sys.executable, "-m", "snowy_gs.harness", "audit", "--run", str(out)],
    ]


def environment(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PHYSSTUDIO_DATA_ROOT"] = str(data_path())
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    for key, folder in [
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
        ("WARP_CACHE_PATH", "warp"),
        ("TORCH_HOME", "torch"),
    ]:
        env.setdefault(key, str(ROOT / ".cache" / folder))
    env.setdefault("MAX_JOBS", "4")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Simulate and render a named demo")
    run.add_argument("method", choices=SCENES)
    run.add_argument("scene")
    run.add_argument("--output", type=Path)
    run.add_argument("--gpu", default="0")
    run.add_argument("--frames", type=int)
    run.add_argument("--width", type=int)
    run.add_argument("--dry-run", action="store_true")
    fetch = sub.add_parser("fetch", help="Download selected pretrained assets")
    fetch.add_argument(
        "asset",
        choices=(
            "playground",
            "firewoods",
            "train",
            "bread",
            "wolf",
            "vasedeck",
            "zohar",
            "lands_end",
            "calico",
            "potted_plant",
            "ficus",
            "alocasia",
            "orchard",
            "bunny",
            "bonsai",
            "all",
        ),
    )
    args = parser.parse_args()
    if args.action == "fetch":
        assets = (
            (
                "playground",
                "firewoods",
                "train",
                "bread",
                "wolf",
                "vasedeck",
                "zohar",
                "lands_end",
                "calico",
                "potted_plant",
                "ficus",
                "alocasia",
                "orchard",
                "bunny",
                "bonsai",
            )
            if args.asset == "all"
            else (args.asset,)
        )
        for asset in assets:
            if asset in ("playground", "firewoods"):
                cmd = command("rainy_gs.fetch", scene=asset)
            elif asset == "train":
                cmd = command(
                    "rainy_gs.fetch_graphdeco", scene="train", provider="voxel51"
                )
            elif asset in ("zohar", "lands_end", "calico"):
                cmd = command("lava_gs.fetch", scene=asset)
            elif asset == "bonsai":
                cmd = command("firex_gs.fetch", scene="bonsai")
            elif asset == "bunny":
                cmd = command("ice_melting.fetch")
            elif asset == "orchard":
                cmd = command("surface_tension.fetch")
            elif asset in ("potted_plant", "ficus", "alocasia"):
                cmd = command("plant_wilting.fetch", scene=asset)
            else:
                cmd = command("gaussian_splashing.fetch", model=asset)
            subprocess.run(cmd, cwd=ROOT, check=True)
        return
    if args.scene not in SCENES[args.method]:
        parser.error("Choose a scene from: " + ", ".join(SCENES[args.method]))
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    if args.method in ("ice", "firex") and args.frames is not None and args.frames < 2:
        parser.error("This method requires at least two frames")
    if args.width is not None and (args.width < 32 or args.width % 2):
        parser.error("--width must be even and at least 32")
    args.output = (
        (args.output or ROOT / "outputs" / args.method / args.scene)
        .expanduser()
        .resolve()
    )
    if args.output.exists() and not args.dry_run:
        parser.error("Choose a fresh --output directory: " + str(args.output))
    commands = plan(args)
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, env=environment(args.gpu), check=True)
    if not args.dry_run:
        (args.output / "run_commands.json").write_text(json.dumps(commands, indent=2))


if __name__ == "__main__":
    main()
