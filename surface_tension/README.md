# Capillary Contact

[Solid-Fluid Interaction with Surface-Tension-Dominant Contact](https://arxiv.org/abs/2105.08471)
(SIGGRAPH 2021) couples surface tension and solid contact to support floating
objects and produce capillary clustering.

| Demo | Scene | Motion | Video |
| :--- | :--- | :--- | :--- |
| Miniature orchard | `orchard` | Scanned apples land, rock, and gather | [6 seconds](../media/surface_tension/orchard.mp4) |
| Capillary cluster | `capillary_cluster` | A stirring rod separates and regroups floating discs | [8 seconds](../media/surface_tension/capillary_cluster.mp4) |
| Paperclips | `paperclips` | A stirring rod drives wire contacts, currents, and wakes | [8 seconds](../media/surface_tension/paperclips.mp4) |

## Install and run

Run from the repository root with Python 3.10+. Simulation uses NumPy/SciPy;
rendering uses [Blender 4.2.9](https://download.blender.org/release/Blender4.2/)
with Cycles. Install FFmpeg for video encoding.

```bash
python -m pip install -r surface_tension/requirements.txt
export BLENDER=/path/to/blender
python -m physstudio fetch orchard

python -m physstudio run capillary orchard --gpu 0
python -m physstudio run capillary capillary_cluster --gpu 0
python -m physstudio run capillary paperclips --gpu 0
```

`BLENDER` defaults to `blender` on PATH. `FFMPEG` selects the encoder executable.
Set `PHYSSTUDIO_DATA_ROOT` to reuse downloaded assets. The cluster and paperclips
are procedural and can run immediately. GPU rendering uses NVIDIA OptiX;
`--cpu` is available through the dedicated harness below.

Each run saves simulation states, contact and volume checks, PNG frames, and
`render/demo.mp4` under `outputs/capillary/<scene>/`. Choose a fresh `--output`
for each simulation. `--frames 2 --width 320` makes a short pipeline check.

## Customize and replay

The unified launcher selects the updated stirring and current-coupling presets
for the cluster and paperclips. The dedicated harness exposes their parameters:

```bash
python -m surface_tension.harness run --scene paperclips --stirrer \
  --output outputs/capillary/custom --gpu 0 --samples 32

# Render an existing trajectory again.
python -m surface_tension.harness run --scene paperclips --stirrer \
  --output outputs/capillary/custom --stage render --gpu 0
```

Defaults use a 96 x 77 surface mesh and save states every 1/120 second.
The orchard uses 180 frames with a 1/360-second step. Stirred scenes settle for
0.18 seconds, then save 240 frames with a 1/480-second step and ten nonlinear
iterations. Contact or high flow speeds trigger local timestep subdivision.
Videos play at 30 fps, showing the simulated motion at quarter speed.

Use `--stage simulate` to cache the trajectory, and `--dt`, `--newton`,
`--settle`, `--viscosity`, or `--resolution` to change the simulation settings.
`--width` changes image size; `--samples` changes Cycles samples per pixel.
Scene geometry and stirring paths are in [scenes.py](scenes.py) and
[stirrer.py](stirrer.py).

Create independent MP4s, animated GIFs, and posters from completed render folders:

```bash
python -m surface_tension.harness package \
  --output outputs/capillary/media \
  --renders outputs/capillary/orchard/render \
            outputs/capillary/capillary_cluster/render \
            outputs/capillary/paperclips/render
```

## Method and assets

An implicit solve couples nonlinear triangle-area energy, finite-depth fluid
inertia, and six-degree-of-freedom rigid contact. Stirred scenes add transported
horizontal currents, two-way drag, continuous wire collisions, and refined
triangle contact. Sparse passive tracers show the simulated water velocity.
Blender renders the saved surface heights and rigid poses with transmission
and reflection. The fluid uses a height-field surface with a finite-depth
wave operator; [METHOD.md](METHOD.md) details its equations and relation to the paper.

The orchard downloads these [CC0](https://polyhaven.com/license) Poly Haven scans:
[Food Apple 01](https://polyhaven.com/a/food_apple_01),
[Dry Quiver Leaf](https://polyhaven.com/a/dry_quiver_leaf), and
[Wooden Bowl 02](https://polyhaven.com/a/wooden_bowl_02).
The downloader verifies MD5 checksums and records source URLs and SHA256 hashes
in `data/surface_tension/sources.json`. Other geometry is procedural.

## Checks

```bash
python -m surface_tension.validate --output outputs/capillary/numerics.json
python -m surface_tension.flow_checks --output outputs/capillary/flow_checks.json
```

Add `--simulations <simulation-folder>` to audit saved trajectories. Checks cover
energy and contact derivatives, action/reaction, compatible flow operators,
volume, continuous solid gaps, water penetration, and per-substep CFL bounds.
The run harness performs these checks before rendering and verifies the encoded
video by decoding it.
