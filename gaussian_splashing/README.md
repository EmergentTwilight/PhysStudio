# <img src="../media/project-icons/gaussian-splashing.svg" width="40" height="40" align="absmiddle" alt=""> Gaussian Splashing · turn splats into liquid

[Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and
Rendering](https://gaussiansplashing.github.io/) connects particle simulation
with Gaussian-based rendering for fluid and solid motion.

## Run

After [installation](../INSTALL.md), download the two object models:

```bash
python -m physstudio fetch wolf
python -m physstudio fetch bread
python -m physstudio run splashing liquify --gpu 0
python -m physstudio run splashing wave --gpu 0
python -m physstudio run splashing plunge --gpu 0
```

The demos show a liquefying wolf, a wave striking the wolf, and bread falling
into water. Each produces `demo.mp4`, cached particle states, camera paths,
and diagnostic images under `outputs/splashing/<scene>/`.
Showcase clips use 180 frames at 1280 × 800 and 30 fps, with half-speed playback.

## Implementation

The Warp solver uses position-based fluids: poly6 density, spiky constraint
gradients, tensile correction, XSPH viscosity, and five constraint iterations
per substep. Six substeps advance each 1/60-second simulation interval.
The object collision proxy is built from closed, filled Gaussian occupancy.
Plunge couples the fluid to a moving rigid body with equal-and-opposite contact
corrections; the wave obstacle is fixed.

Fluid rendering uses anisotropic Gaussian kernels, depth/thickness normals,
Fresnel reflection, Beer absorption, and screen-space refraction. The solid
objects retain their pretrained spherical-harmonic appearance.

## Simulate and render separately

```bash
CUDA_VISIBLE_DEVICES=0 python -m gaussian_splashing.demo \
  --scene plunge --output outputs/plunge_states --simulate-only
CUDA_VISIBLE_DEVICES=0 python -m gaussian_splashing.demo \
  --scene plunge --output outputs/plunge_states --render-only \
  --render-output outputs/plunge_render --width 1280 --height 800
```

Use `--spacing` to change particle resolution, or `--render-step 8` for a sparse
render preview. `--azimuth-offset` changes the orbit's starting angle.

## Checks

```bash
CUDA_VISIBLE_DEVICES=0 python -m gaussian_splashing.validate
```

This checks density relaxation, kernel normalization, and equal-and-opposite
fluid-solid contact corrections.
Assets and download links: [ASSETS.md](../ASSETS.md).
