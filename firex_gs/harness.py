"""Prepare, simulate, render, and validate the three fire-suppression scenes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from physstudio.__main__ import environment
from physstudio.paths import ROOT, data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=["firewoods", "playground", "bonsai"], default="firewoods"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--width", type=int, default=1280)
    control = parser.add_mutually_exclusive_group()
    control.add_argument(
        "--dry-control",
        action="store_true",
        help="Simulate and render a matched no-water control",
    )
    control.add_argument(
        "--dry-diagnostics",
        action="store_true",
        help="Simulate the control and include its diagnostics",
    )
    args = parser.parse_args()
    frames = (
        args.frames
        if args.frames is not None
        else (420 if args.scene == "firewoods" else 450)
    )
    if frames < 2 or args.width < 32 or args.width % 2:
        parser.error("Use at least two frames and an even width of at least 32")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Choose a fresh output directory")
    output.mkdir(parents=True)
    env = environment(args.gpu)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "3"
    config = Path(__file__).parent / "configs" / (args.scene + ".json")
    commands = []

    def run(module, *options):
        cmd = [sys.executable, "-m", "firex_gs." + module, *map(str, options)]
        commands.append(cmd)
        (output / "commands.json").write_text(json.dumps(commands, indent=2))
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    if (
        args.scene != "firewoods"
        and not data_path("firex_gs/prepared", args.scene, "material.npz").exists()
    ):
        run("prepare", "--scene", args.scene)
    run("validate", "--output", output / "numerics.json")
    run("validate_wood", "--output", output / "wood_numerics.json")
    run(
        "simulate",
        "--scene",
        args.scene,
        "--output",
        output / "simulation",
        "--frames",
        frames,
        "--dt",
        1 / 180,
        "--config",
        config,
    )
    run(
        "refine",
        "--input",
        output / "simulation",
        "--output",
        output / "render",
        "--width",
        args.width,
        "--renderer",
        "fierygs",
    )
    run(
        "validate_appearance",
        "--input",
        output / "simulation",
        "--config",
        config,
        "--output",
        output / "appearance_audit.json",
    )
    if args.dry_control or args.dry_diagnostics:
        run(
            "simulate",
            "--scene",
            args.scene,
            "--output",
            output / "dry_simulation",
            "--frames",
            frames,
            "--dt",
            1 / 180,
            "--config",
            config,
            "--control",
            "dry",
        )
        if args.dry_control:
            run(
                "refine",
                "--input",
                output / "dry_simulation",
                "--output",
                output / "dry_render",
                "--width",
                args.width,
                "--renderer",
                "fierygs",
            )
    run("audit", "--input", output)
    run("present", "--input", output)


if __name__ == "__main__":
    main()
