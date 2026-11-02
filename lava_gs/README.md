# <img src="../media/project-icons/lavags.svg" width="40" height="40" align="absmiddle" alt=""> LavaGS · lava cascades in Gaussian scenes

LavaGS combines [Augmented MPM for Phase-Change and Varied Materials](https://www.cs.ucr.edu/~craigs/papers/2014-melting/paper.pdf) with Gaussian scenes for lava flow, crust formation, rock contact, and thermal emission.

## Run

After [installation](../INSTALL.md), install the scene converter with Node.js 22+:

```bash
npm install --prefix .cache/lava_tools --cache .cache/npm \
  --ignore-scripts --no-audit --no-fund @playcanvas/splat-transform@3.1.7
python -m physstudio fetch zohar
python -m physstudio fetch lands_end
python -m physstudio fetch calico

python -m physstudio run lava zohar --gpu 0
python -m physstudio run lava lands_end --gpu 0
python -m physstudio run lava calico --gpu 0
```

The downloader selects the released LOD, converts it to PLY, and installs calibrated cameras and attribution. Calico's degree-zero Gaussians receive zero-valued unused SH coefficients; the native PLY is retained. Set `PHYSSTUDIO_DATA_ROOT` to store models elsewhere. [Asset sources and licenses](../ASSETS.md).

| Scene | Inlet width | Shape | Flow | Camera / duration |
| :--- | ---: | :--- | ---: | :--- |
| Zohar | 2.2 m | Rounded | 2.5248 m³/s | Canyon cascade, 8 s |
| Land's End | 4.2 m | Uneven slot following the crest | 2.3936 m³/s | Split rock falls, 8 s |
| Calico | 1.0 m | Rounded, following the slope | 0.3510 m³/s | 80° orbit with a descending approach, 5 s |

Each run writes `lava_demo.mp4`, `comparison.mp4`, PNG frames, cached states, collision geometry, camera matrices, source snapshots, and `audit.json` under `outputs/lava/<scene>/`. Use a fresh `--output` directory for each run. A short check uses `--frames 2 --width 320`; `--dry-run` prints the launcher's commands.

## Customize

The native entry point exposes inlet, material, and camera controls:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lava_gs.demo \
  --scene lands_end --output outputs/lava/custom \
  --inlet-width 3.2 --inlet-thickness .7 --inlet-speed 1.0 \
  --cache-states --save-frames
```

`--inlet-profile` selects `ellipse` or `slot`; `--inlet-conform` fits the inlet bottom to the terrain. Width, thickness, and speed use simulation metres and seconds. The adopted scene scales define these units; source X/Y and camera coordinates use scene units. Particle counts determine the supplied flow.

Camera-only changes reuse saved physics:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lava_gs.demo \
  --scene calico --replay outputs/lava/calico --output outputs/lava/calico_front \
  --frames 151 --crust 1.25 --surface-iso .12 \
  --camera-config lava_gs/configs/calico_front_camera.json \
  --cache-states --save-frames
```

`calico_orbit_camera.json` specifies the 80° orbit; `calico_front_camera.json` provides a closer frontal shot. `python -m lava_gs.preview --help` lists single-frame material and camera previews.

## Implementation

- Warp quadratic MLS/APIC MPM evolves particles with logarithmic volumetric stress, Maxwell relaxation, temperature-dependent yield, and adaptive substeps.
- A Gaussian-derived terrain SDF supplies frictional rock contact. Continuous inlet layers account for injected volume and birth time.
- An implicit heat step handles conduction and boundary exchange. Advected material coordinates and surface age control rough crust, local molten patches, restrained bloom, and contact shading.
- A density isosurface is ray marched against Gaussian depth. Calico uses a directional sky background where its exported asset is transparent.

The presets use 0.16 m grids for Zohar and Land's End, and 0.12 m for Calico. The calibrated scene settings and rendering parameters are in `configs/`.

## Checks

```bash
CUDA_VISIBLE_DEVICES=0 python -m lava_gs.validate --output outputs/lava_physics.json
CUDA_VISIBLE_DEVICES=0 python -m lava_gs.audit outputs/lava/calico
```

Checks cover free fall, mass transfer, heat balance, cooling, rheology, volume conservation, impact, inflow accounting, contact separation, replay consistency, depth occlusion, and complete video decoding.
