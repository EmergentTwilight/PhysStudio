# <img src="../media/project-icons/stormgs.svg" width="40" height="40" align="absmiddle" alt=""> StormGS · storms in Gaussian scenes

[Cyclogenesis: Simulating Hurricanes and Tornadoes](https://computationalsciences.org/publications/amador-herrera-2024-cyclogenesis.html)
provides the moist-flow formulation for rotating clouds, rain, sea spray, and
terrain-driven dust in these 3D Gaussian scenes.

## Run

After [installation](../INSTALL.md), fetch the three pretrained environments.
Land's End and Calico use the same Node.js 22+ converter as LavaGS:

```bash
npm install --prefix .cache/lava_tools --cache .cache/npm \
  --ignore-scripts --no-audit --no-fund @playcanvas/splat-transform@3.1.7
python -m physstudio fetch playground
python -m physstudio fetch lands_end
python -m physstudio fetch calico

python -m physstudio run storm playground --gpu 0
python -m physstudio run storm lands_end --gpu 0
python -m physstudio run storm calico --gpu 0
```

Set `PHYSSTUDIO_DATA_ROOT` to reuse downloaded models from another location.
[Asset links and credits](../ASSETS.md).

| Scene | Effect | Grid | Warmup |
| :--- | :--- | :--- | ---: |
| Playground | Rain-wrapped tornado and airborne fragments | 80 × 80 × 88 | 5 s |
| Land's End | Coastal funnel and sea spray | 80 × 80 × 104 | 5 s |
| Calico | Dust devil over rocks and shrubs | 72 × 72 × 128 | 8 s |

Each preset renders 240 frames at 30 fps with a moving camera. Outputs include
`storm_demo.mp4`, `comparison.mp4`, PNG frames, three frozen physical keyframes,
camera matrices, source snapshots, and diagnostics under `outputs/storm/<scene>/`.
Use a fresh `--output` directory; `--frames 2 --width 320` runs a short check.

To generate all three clips and a local gallery in one run:

```bash
python -m storm_gs.showcase --gpu 0 --output outputs/storm/showcase
```

## Customize

Presets are in [scenes.py](scenes.py). The direct entry point supports a
three-keyframe preview and JSON overrides:

```bash
CUDA_VISIBLE_DEVICES=0 python -m storm_gs.demo \
  --scene lands_end --output outputs/storm/coast_preview \
  --frames 90 --width 960 --warmup 5 --dt .012 --preview
```

`--config settings.json` replaces selected top-level preset values. Supply the
full `simulation` or `camera` dictionary when overriding either one. `--dt`
sets the maximum step; the actual step divides a video frame evenly and is
saved as `effective_dt`. The released videos use 1/90-second steps.

Re-render the saved physical keyframes with different extinction or exposure:

```bash
CUDA_VISIBLE_DEVICES=0 python -m storm_gs.replay \
  --input outputs/storm/calico --output outputs/storm/calico_rerender \
  --config render_settings.json
```

## Implementation

- Warp advances staggered air velocity, temperature, vapor, cloud water, rain
  water, dust, and turbulent energy. Moist buoyancy, latent heat, and eddy mixing
  couple the fields.
- Warm-started pressure projection enforces zero normal flow at solid faces.
  A controlled rotating target supplies the updraft and return flow; localized
  vapor supply sustains the condensation funnel.
- Limited MacCormack transport preserves dust detail. Inertial Gaussian
  particles follow the projected flow with drag, gravity, and terrain contact.
- Self-shadowed volume rendering is bounded by Gaussian expected depth.
  Velocity-aligned particle covariances supply motion blur. Sky replacement
  and an analytic ocean background complete the environments.

The atmospheric model uses warm-cloud phases and curl-based vorticity
confinement. Existing scene Gaussians remain static collision geometry. The
scene-to-simulation scale, forcing, and cloud sources are preset controls for
these local vortex demonstrations.

## Checks

```bash
CUDA_VISIBLE_DEVICES=0 python -m storm_gs.validate \
  --output outputs/storm_numerics.json
python -m storm_gs.showcase --output outputs/storm/showcase --audit-only
```

Checks cover phase-change water and latent-heat balance, pressure projection,
solid-wall velocities, bounded dust transport, foreground occlusion, moving
cameras, saved states, frame counts, and video decoding.

Paper and asset sources are recorded in
[references/sources.json](references/sources.json). The authors' [CycloCore
notebook](https://github.com/AgrosAmad/CycloCore) was consulted for the atmospheric
equations. Shared scene loading and camera utilities come from RainyGS and LavaGS.
