"""Conservation, locality and cooling checks for the wood propagation adapter."""
import argparse, json
from pathlib import Path
import numpy as np
import warp as wp
from .solver import Fire
from .scenes import PRESETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = dict(
        PRESETS["firewoods"],
        shape=[12, 6, 6],
        origin=[0.0, 0.0, 0.0],
        dx=0.035,
        wood_conductivity=0.15,
        ignition_regions=[dict(center=[0.07, 0.105, 0.105], radius=0.065)],
    )
    shape = tuple(cfg["shape"])
    wood = np.zeros(shape, "i")
    wood[1:5, 1:5, 1:5] = 1
    wood[8:11, 1:5, 1:5] = 1
    geo = dict(
        wood=wood,
        solid=wood.copy(),
        sdf=np.ones(shape, "f"),
        exits=np.zeros(shape + (3,), "i"),
    )
    sim = Fire(cfg, geo)
    initial = sim.wt.numpy()
    mass = sim.wm.numpy().astype("d")
    assert initial[8:11, 1:5, 1:5].max() == 310.0
    assert (initial > 450).sum() > 0 and (initial[wood > 0] == 310).sum() > 0
    energy0 = float((initial * mass).sum())
    hot = initial > 450
    cold = (wood > 0) & ~hot
    for _ in range(240):
        sim.wood_thermal.step(0.05)
    final = sim.wt.numpy()
    energy1 = float((final * mass).sum())
    relative = abs(energy1 - energy0) / energy0
    assert relative < 2e-6, relative
    assert final[hot].mean() < initial[hot].mean()
    assert final[cold].max() > initial[cold].max() + 1
    assert np.all(final[8:11, 1:5, 1:5] == 310.0)
    assert final.min() >= initial.min() - 1e-4 and final.max() <= initial.max() + 1e-4
    # A large requested step must subcycle without violating the maximum principle.
    before = sim.wt.numpy()
    sim.wood_thermal.step(200.0)
    after = sim.wt.numpy()
    assert (
        after.max() <= before.max() + 1e-4
        and after[wood > 0].min() >= before[wood > 0].min() - 1e-4
    )
    assert abs(float((after * mass).sum()) - energy1) / energy1 < 3e-6
    # An extinguished region is free to cool; ignition is not reapplied each step.
    sim.wt.fill_(310.0)
    sim.wood_thermal.step(1.0)
    assert float(sim.wt.numpy().max()) == 310.0
    sim.wt.assign(initial)
    for _ in range(6):
        sim.step(1 / 180)
    diagnostic = sim.diagnostics()
    assert np.isfinite(diagnostic["temperature_max"])
    assert diagnostic["species_sum_error"] < 1e-5 and diagnostic["species_min"] >= -1e-7
    result = dict(
        passed=True,
        relative_solid_heat_error=relative,
        localized_ignition=True,
        heat_spreads_within_connected_wood=True,
        disconnected_wood_stays_cold=True,
        stable_large_step=True,
        cooling_is_not_overridden=True,
        coupled_solver_step_passed=True,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
