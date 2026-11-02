# <img src="../media/project-icons/thundergs.svg" width="40" height="40" align="absmiddle" alt=""> ThunderGS · electrical storms in Gaussian scenes

[Thunderstruck: Visually Simulating Electrical Storms](https://jonathank.de/research/paper/Herrera2025_thunderstruck.pdf)
provides the electric-field-driven discharge model for branched lightning,
ground attachment, and transient illumination in these Gaussian scenes.

## Run

After [installation](../INSTALL.md), download the pretrained environments.
Zohar uses the Node.js 22+ converter shared with LavaGS:

```bash
npm install --prefix .cache/lava_tools --cache .cache/npm \
  --ignore-scripts --no-audit --no-fund @playcanvas/splat-transform@3.1.7
python -m physstudio fetch firewoods
python -m physstudio fetch zohar
python -m physstudio fetch train

python -m physstudio run thunder firewoods --gpu 0
python -m physstudio run thunder zohar --gpu 0
python -m physstudio run thunder train --gpu 0
```

Set `PHYSSTUDIO_DATA_ROOT` to reuse downloaded models.
[Asset sources and credits](../ASSETS.md).

| Scene | Effect | Electrical grid |
| :--- | :--- | :--- |
| Firewoods | Close strike and illuminated ground | 72 × 72 × 96 |
| Zohar | Canyon discharge and rock illumination | 72 × 88 × 112 |
| Train | Roof attachment and flash lighting | 72 × 80 × 96 |

Each preset generates three discharges and renders an 8-second moving-camera
clip at 30 fps. Under `outputs/thunder/<scene>/`, `simulation/` contains the
electrical fields and branching trees; `render/` contains `thunder_demo.mp4`,
`comparison.mp4`, PNG frames, camera matrices, and diagnostics.

Use a fresh `--output` for each run. `--frames 2 --width 320` makes a short
rendering check while retaining the full discharge simulation.

Generate all three clips and a local gallery:

```bash
python -m thunder_gs.showcase --gpu 0 --output outputs/thunder/showcase
```

## Customize and replay

Scene transforms are in [configs/](configs/), and electrical parameters,
cloud charge reservoirs, exposure, and camera paths are in [scenes.py](scenes.py).
The simulation and renderer can run separately:

```bash
CUDA_VISIBLE_DEVICES=0 python -m thunder_gs.simulate \
  --scene firewoods --events 3 --output outputs/thunder/custom_sim
CUDA_VISIBLE_DEVICES=0 python -m thunder_gs.demo \
  --simulation outputs/thunder/custom_sim \
  --output outputs/thunder/custom_render --width 1280 --frames 240
```

Add `--preview` to render the opening, three strike keyframes, and final frame.
Camera and exposure edits take effect when rendering an existing simulation.
The saved render configuration and camera matrices support independent replay:

```bash
CUDA_VISIBLE_DEVICES=0 python -m thunder_gs.replay \
  --root outputs/thunder --scenes firewoods --frames 33 110 186
```

## Implementation

- Warp solves the Poisson equation with red-black SOR and captured CUDA graphs.
  Terrain and structures extracted from Gaussian geometry form the grounded
  boundary; the top and sides use homogeneous Neumann boundaries.
- Bipolar growth follows the local electric field, critical breakdown strength,
  and internal channel resistance. Each accepted edge updates the potential.
  A floating-conductor correction maintains channel neutrality before contact.
- Prescribed cloud charge reservoirs drive the discharge model. Each strike
  solves a separate reservoir configuration; restrike timing controls exposure.
- World-space emissive segments use Gaussian depth for occlusion. Accumulated
  induced charge controls branch brightness, with a stronger grounded return path.
- Distributed channel lights, terrain shadows, depth-derived surface normals,
  clouds, rain, and bloom produce the scene lighting and atmosphere.

## Checks

```bash
CUDA_VISIBLE_DEVICES=0 python -m thunder_gs.validate \
  --output outputs/thunder/numerics.json \
  --simulations outputs/thunder/firewoods/simulation
python -m thunder_gs.showcase --output outputs/thunder/showcase --audit-only
```

Checks cover Poisson accuracy, gauge invariance, conductor field enhancement,
depth occlusion, tree connectivity, ground contact, charge balance, video
decoding, and camera motion. Replay compares saved PNGs against independently
rendered electrical states.

[Paper and asset references](references/sources.json). Scene loading, calibrated
cameras, and video encoding share the RainyGS and LavaGS utilities.
