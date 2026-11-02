"""Register Playground wood labels or the Bonsai ornament and collision region."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lava_gs.scene import Scene
from physstudio.paths import data_path
from rainy_gs.scene import save_image
from .scenes import PRESETS, camera, asset_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["playground", "bonsai"], required=True)
    parser.add_argument(
        "--inspect", action="store_true", help="Render the selected combustible region"
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    output = data_path("firex_gs/prepared", args.scene)
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(PRESETS[args.scene])
    cfg.update(
        json.loads(
            (Path(__file__).parent / "configs" / (args.scene + ".json")).read_text()
        )
    )
    config = (
        {}
        if args.scene == "playground"
        else json.loads(
            (Path(__file__).parent / "configs/bonsai_scene.json").read_text()
        )
    )
    scene = Scene(asset_path(cfg["asset"]), config=config)
    xyz = scene.means.cpu().numpy()
    scale = scene.scales.cpu().numpy().max(-1)
    opacity = scene.opacity.cpu().numpy()
    if args.scene == "playground":
        mapping = data_path("firex_gs/fierygs/Playground")
        mask = (
            torch.load(
                mapping / "mask_pts_in_grids.pth", map_location="cpu", weights_only=True
            )
            .numpy()
            .astype(bool)
        )
        ids = torch.load(
            mapping / "indices_pts_in_grids.pth", map_location="cpu", weights_only=True
        ).numpy()
        lower = np.array([-1.190500020980835, -0.9205000400543213, -1.2904999256134033])
        upper = np.array([1.2404999732971191, 1.5104999542236328, 1.1405000686645508])
        fractional = (xyz[mask] - lower) / ((upper - lower) / 256)
        error = float(np.maximum(ids - fractional, fractional - ids - 1).max())
        if error > 0.001:
            raise ValueError(
                "Released Playground bounds do not match the Gaussian mapping"
            )
        material = np.load(scene.path / "material_grid.npy")
        burn_ids = np.flatnonzero(mask)[material[tuple(ids.T)] == 1]
        wood = np.argwhere(material == 1)
        fuel = lower + (wood + 0.5) * (upper - lower) / 256
        collision = (
            ((xyz > [-1.3, -1.1, -0.55]) & (xyz < [1.4, 1.65, 0.5])).all(-1)
            & (opacity > 0.6)
            & (scale < 0.08)
        )
        ground = scene.ground.cpu().numpy()
        annotation = {
            "method": "Released FieryGS material label 1 (wood)",
            "source": "FieryGS Playground",
        }
    else:
        lower = np.array([-0.35, -0.35, -2.02])
        upper = np.array([1.3, 1.35, -1.15])
        mask = (
            ((xyz > lower) & (xyz < upper)).all(-1) & (scale < 0.16) & (opacity > 0.06)
        )
        burn_ids = np.flatnonzero(mask)
        fuel = xyz[mask]
        collision = (
            ((xyz > [-1.5, -1.5, -4.2]) & (xyz < [2.5, 2.5, -1.1])).all(-1)
            & (scale < 0.1)
            & (opacity > 0.55)
        )
        ground = np.array(config["ground_plane"])
        error = 0.0
        annotation = {
            "method": "Authored combustible ornament ROI",
            "bounds": [lower.tolist(), upper.tolist()],
            "source": "Graphdeco Bonsai",
            "min_opacity": 0.06,
            "max_scale": 0.16,
        }
        (output / "scene_config.json").write_text(json.dumps(config, indent=2))
    if not len(burn_ids) or not len(fuel):
        raise ValueError("The asset has no selected combustible geometry")
    np.savez_compressed(
        output / "material.npz",
        gaussian_ids=burn_ids,
        fuel_points=fuel,
        collision_points=xyz[collision],
        ground=ground,
        registration_error=error,
    )
    annotation.update(
        gaussians=len(burn_ids),
        registration_error=error,
        asset=cfg["asset"],
        material_sha256=hashlib.sha256(
            (output / "material.npz").read_bytes()
        ).hexdigest(),
    )
    (output / "material_annotation.json").write_text(json.dumps(annotation, indent=2))
    if args.inspect:
        colors = (scene.sh[:, 0] * 0.28209479177387814 + 0.5).clamp(0, 1)
        colors[burn_ids] = colors.new_tensor([1.0, 0.1, 0.03])
        image, _ = scene.render(camera(cfg, 0.5, 960), colors=colors)
        save_image(output / "material_mask.png", image[..., :3])
    print(json.dumps(annotation, indent=2))


if __name__ == "__main__":
    main()
