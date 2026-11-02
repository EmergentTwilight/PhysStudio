"""Download the Gaussian scenes and material registration needed by Fire-X GS."""
import argparse
import subprocess
import sys

from physstudio.paths import data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=["firewoods", "playground", "bonsai", "all"], required=True
    )
    args = parser.parse_args()
    scenes = (
        ["firewoods", "playground", "bonsai"] if args.scene == "all" else [args.scene]
    )
    for scene in scenes:
        if scene == "bonsai":
            subprocess.run([sys.executable, "-m", "firex_gs.fetch_bonsai"], check=True)
            continue
        subprocess.run(
            [sys.executable, "-m", "rainy_gs.fetch", "--scene", scene], check=True
        )
        folder = "firewoods_sand_dark" if scene == "firewoods" else "Playground"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rainy_gs.fetch",
                "--scene",
                scene,
                "--output",
                str(data_path("firex_gs/fierygs")),
                "--member",
                folder + "/mask_pts_in_grids.pth",
                "--member",
                folder + "/indices_pts_in_grids.pth",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
