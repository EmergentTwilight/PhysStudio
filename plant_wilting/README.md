# <img src="../media/project-icons/plant-wilting.svg" width="40" height="40" align="absmiddle" alt=""> Plant Wilting

[A Physically-inspired Approach to the Simulation of Plant Wilting](https://dl.acm.org/doi/10.1145/3610548.3618218) couples water transport and material softening to animate drying plants, rendered here with 3D Gaussians.

## Run

After [installation](../INSTALL.md), install the pinned Newton solver into the
project-local cache and download the plant assets:

```bash
python -m plant_wilting.runtime
python -m physstudio fetch potted_plant
python -m physstudio fetch ficus
python -m physstudio fetch alocasia

python -m physstudio run wilting potted_plant --gpu 0
python -m physstudio run wilting ficus --gpu 0
python -m physstudio run wilting alocasia --gpu 0
```

The solver uses Newton VBD rods and Warp 1.17.0 in `.cache/plant_wilting_runtime/`.
Gaussian preparation and rendering use the shared environment. Set
`PHYSSTUDIO_PLANT_RUNTIME` to choose another solver dependency directory.

Each preset produces an eight-second, 1920 × 1080 video with a 70-degree camera
orbit. Appearance and local water content play side by side; the plot shows
total water and the local 10th–90th percentile range. Results are written to
`outputs/wilting/<scene>/render/water_demo.mp4`.

A short end-to-end run:

```bash
python -m physstudio run wilting potted_plant --frames 2 --width 640 \
  --output outputs/wilting/check_plant --gpu 0
```

Use `--dry-run` to inspect commands and `--output` for a fresh destination.

| Scene | Simulation frames | Final water retained | Video frames |
| :--- | ---: | ---: | ---: |
| Potted Plant | 180 | 32% | 240 |
| Ficus | 180 | 32% | 240 |
| Alocasia | 120 | 16% | 240 |

## Simulation and rendering

- Extract a rooted stem and venation graph from the Gaussian geometry. Fix the
  root and pot; keep hanging lower leaves deformable.
- Solve conservative graph transport with leaf transpiration. Separate the
  structural and effective hydraulic radii to retain local water gradients.
- Map relative water content to rod stiffness and update the water mass. Use
  1 ms mechanical steps, 12 substeps, 160 VBD iterations, and 120 warmup steps.
- Bind Gaussians through compact spatial weights and branch-graph distances.
  Rotate covariance axes and spherical-harmonic directions with the blended
  rod frames, preserving Gaussian principal sizes.

The effective hydraulic radius defaults to 0.08 times the structural radius,
calibrated for these extracted cages. Hydraulic time and mechanical relaxation
time are recorded separately. The heatmap uses a fixed 0–100% scale.

To include a fully hydrated control or adjust transport:

```bash
CUDA_VISIBLE_DEVICES=0 python -m plant_wilting.demo --scene potted_plant \
  --output outputs/wilting/plant_control --controls --hydraulic-radius-ratio .08
```

The simulation freezes its Gaussians, rig, parameters, and source files.
Render a saved trajectory with a different camera orbit:

```bash
CUDA_VISIBLE_DEVICES=0 python -m plant_wilting.render_water \
  --input outputs/wilting/potted_plant/simulation \
  --output outputs/wilting/plant_fixed --orbit 0
```

Add `--preview` for five review frames. `plant_wilting.render` produces an
appearance-only video. The default binding is `material` and covariance mode
is `rod`; `--binding spatial --covariance jacobian` exposes the alternate map.

## Assets and checks

The three models and their segmentation settings come from
[DiffWind](https://github.com/zju3dv/DiffWind/tree/a47530ec2fafd6f647c87a8f8e086b1905dbfe52/assets).
The downloader verifies the pinned files against [SHA256 records](references/assets.json).
Downloaded assets stay under `data/plant_wilting/source/` or `PHYSSTUDIO_DATA_ROOT`.

Every demo checks mass conservation, spectral integration, the retained-water
target, fixed roots, rod connectivity, Gaussian identity binding, finite-difference
Jacobians, and covariance size preservation. Results are in `validation.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m plant_wilting.validate --gpu-binding \
  --trajectories outputs/wilting/potted_plant/simulation \
  --output outputs/wilting/plant_validation.json
```

The water model follows the [authors' implementation](https://github.com/filthynobleman/plant-water-diffusion);
the GPU rod backend uses [Newton](https://github.com/newton-physics/newton).
See the [reference license](LICENSE.reference) and [pinned sources](references/sources.json).
