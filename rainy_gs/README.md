# <img src="../media/project-icons/rainygs.svg" width="40" height="40" align="absmiddle" alt=""> RainyGS · rain, ripples, and wet reflections

Inspired by [RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian
Splatting](https://pku-vcl-geometry.github.io/RainyGS/), which brings physically
simulated rainfall and water effects into reconstructed Gaussian scenes.

## Run

Follow [installation](../INSTALL.md), then download the scenes once:

```bash
python -m physstudio fetch playground
python -m physstudio fetch firewoods
python -m physstudio fetch train
python -m physstudio run rain playground --gpu 0
python -m physstudio run rain firewoods --gpu 0
python -m physstudio run rain train --gpu 0
```

Each run writes `rain_demo.mp4`, `comparison.mp4`, camera paths, configuration,
water diagnostics, and a saved simulation state under `outputs/rain/<scene>/`.
Add `--frames 12 --width 640 --output outputs/rain_preview` for a short preview.

## Implementation

- World-space rain particles follow gravity, wind, and terminal drag; impacts
  generate ballistic splash particles and a damped wave field.
- A Gaussian-derived terrain proxy supports finite-volume shallow water with
  hydrostatic reconstruction, CFL substeps, and a rainfall mass source.
- Exposure-aware Gaussian streaks use segmented depth occlusion. Water combines
  mirror-camera reflection, ripple normals, Fresnel blending, and refraction.
- Wet vegetation and overcast illumination share the original scene geometry.

`configs/*_demo.json` contains the showcase settings; `firewoods.json` and
`train.json` calibrate coordinates, ground fitting, and simulation bounds.
Initial puddles are pre-existing water. Rain intensity is measured in mm/hour;
visible particle count is a separate rendering control. Scene scales are
estimated as 2.5, 0.7, and 2.5 metres per unit for Playground, Firewoods, and Train.

For individual controls, use `python -m rainy_gs.demo --help`. This includes
`--rain-mm-hour`, `--wind-x`, `--wind-y`, `--exposure`, `--surface-water-mm`,
`--camera`, and `--camera-end`.

## Checks

```bash
python -m rainy_gs.test_physics
CUDA_VISIBLE_DEVICES=0 python -m rainy_gs.test_render
```

Checks cover still-water preservation, wet/dry flow, mass balance, impact
particles, exposure moments, camera transforms, and per-pixel occlusion.
Assets and download links: [ASSETS.md](../ASSETS.md).
