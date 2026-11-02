"""Check FieryGS appearance adaptation against its formulas and scene identity."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from .scenes import load, camera
from .reference_sh import eval_sh
from .render_fierygs import Renderer, carbon_offset, illuminate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    wp.init()
    cfg = json.loads((args.input / "config.json").read_text())
    cfg.update(json.loads(args.config.read_text()))
    scene, _ = load(cfg["scene"], cfg)
    geo = dict(np.load(args.input / "geometry.npz"))
    renderer = Renderer(scene, cfg, geo)
    cam = camera(cfg, 0.0, 480)
    state = dict(np.load(args.input / "states/00000.npz"))
    pristine_state = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in state.items()
    }
    pristine_state["char"].fill(0)
    pristine_state["wet"].fill(0)
    renderer.update(pristine_state)
    original, _ = scene.render(cam)
    adapted, _ = renderer.render_scene(cam, lighting=False)
    zero_image_error = float((original - adapted).abs().max())
    assert zero_image_error < 2e-5, zero_image_error
    reference = eval_sh
    eye = (
        torch.linalg.inv(scene.world_transform)
        @ torch.cat([cam.center, cam.center.new_ones(1)])
    )[:3]
    direction = torch.nn.functional.normalize(scene.render_means - eye, dim=-1)
    expected = (reference(3, scene.sh.transpose(1, 2), direction) + 0.5).clamp_min(0)
    actual = renderer.surface_colors(cam, lighting=False)
    sh_error = float((expected - actual).abs().max())
    assert sh_error < 2e-5, sh_error
    occupancy = torch.tensor([1.0, 0.8, 0.7, 0.6, 0.3, 0.0], device=scene.device)
    color = torch.tensor([[0.8, 0.4, 0.2]], device=scene.device).expand(
        len(occupancy), -1
    )
    ported = (color - carbon_offset(occupancy)[:, None]).clamp_min(0)
    upstream = color.clone()
    selected = occupancy < 0.7
    upstream[selected] = (
        upstream[selected] - 0.5 * (0.7 - occupancy[selected, None]) / 0.7
    ).clamp_min(0)
    formula_error = float((ported - upstream).abs().max())
    assert formula_error < 1e-7
    outside = torch.ones(len(scene.means), device=scene.device, dtype=torch.bool)
    outside[renderer.roi] = False
    previous = torch.zeros(len(renderer.roi), device=scene.device)
    records = []
    previous_conversion = np.zeros(geo["wood"].shape, dtype=np.float32)
    selected = sorted(
        set(
            [
                0,
                min(30, cfg["frames"] - 1),
                min(90, cfg["frames"] - 1),
                cfg["frames"] // 2,
                cfg["frames"] - 1,
            ]
        )
    )
    for frame in selected:
        state = dict(np.load(args.input / "states" / f"{frame:05d}.npz"))
        assert np.all(state["char"] >= previous_conversion - 1e-9)
        previous_conversion = state["char"]
        renderer.update(state)
        assert torch.all(renderer.offset >= previous - 1e-6)
        previous = renderer.offset.clone()
        changed = renderer.surface_colors(cam, lighting=False)
        no_char = renderer.surface_colors(cam, lighting=False, carbon=False)
        assert torch.equal(changed[outside], no_char[outside])
        assert torch.all(changed[renderer.roi] <= no_char[renderer.roi] + 1e-6)
        records.append(dict(frame=frame, **renderer.material_stats))
    state0 = dict(np.load(args.input / "states/00000.npz"))
    # Replaying earlier frames must not inherit later carbonization.
    renderer.update(state0)
    assert (
        abs(
            renderer.material_stats["mean_rgb_subtraction"]
            - records[0]["mean_rgb_subtraction"]
        )
        < 1e-7
    )
    result = dict(
        passed=True,
        zero_state_max_pixel_error=zero_image_error,
        upstream_sh_max_error=sh_error,
        upstream_carbon_formula_error=formula_error,
        carbon_outside_material_mask_changes=0,
        carbon_is_monotone=True,
        replay_order_independent=True,
        registration=renderer.registration,
        frames=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
