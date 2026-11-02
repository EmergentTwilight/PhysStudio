# <img src="../media/project-icons/snowygs.svg" width="40" height="40" align="absmiddle" alt=""> SnowyGS · a winter pass for Gaussian scenes

SnowyGS brings the elastoplastic model of [A Material Point Method for Snow
Simulation](https://alexey.stomakhin.com/research/snow.html) into Gaussian scenes,
with wind-driven flakes and a continuous accumulated snow surface.

## Run

After [installation](../INSTALL.md), download the scenes once:

```bash
python -m physstudio fetch playground
python -m physstudio fetch firewoods
python -m physstudio fetch train
python -m physstudio run snow playground --gpu 0
python -m physstudio run snow firewoods --gpu 0
python -m physstudio run snow train --gpu 0
```

Each run writes `snow_demo.mp4`, `comparison.mp4`, camera poses, final particle
state, collision geometry, and an audit under `outputs/snow/<scene>/`.
The showcase presets render 240 moving-camera frames at 30 fps.

## Implementation

- Warp MLS-MPM evolves the foreground snow blanket, using SVD plastic return,
  hardening, frictional scene contact, and adaptive substeps.
- Wind and drag move airborne flakes in world space. Representative impacts
  deposit particle mass onto the evolving snow layer.
- A ray-marched mass isosurface gives the settled snow a continuous outline.
  Snow normals, soft lighting, and per-pixel scene depth determine its shading
  and occlusion. Airborne flakes use exposure streaks and optical depth.
- A surface-exposure appearance pass adds distant snow and foliage frost.

The presets start with an existing snow blanket and warm it up for two seconds.
Material settings are Young's modulus 20 kPa, density 300 kg/m³, Poisson ratio
0.2, hardening 10, compression/stretch thresholds 0.025/0.0075, and friction
0.45. Plastic volume is regularized to `[0.6, 2.0]`.

`geometry.py` contains each scene's metric scale, collision volume, and camera
endpoints. Weather controls are available through
`python -m snowy_gs.demo --help`: `--flakes`, `--wind`, `--haze`, and `--strength`.

## Checks

```bash
CUDA_VISIBLE_DEVICES=0 python -m snowy_gs.validate --output outputs/check_snow_physics
CUDA_VISIBLE_DEVICES=0 python -m snowy_gs.validate_render --output outputs/check_snow_render
python -m snowy_gs.harness audit --run outputs/snow/playground
```

Checks cover free fall, plastic impact, friction, deposited mass, scene
penetration, depth occlusion, isosurface intersections, and video decoding.
Assets and download links: [ASSETS.md](../ASSETS.md).
