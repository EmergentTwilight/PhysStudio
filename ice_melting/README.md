# <img src="../media/project-icons/ice-melting.svg" width="40" height="40" align="absmiddle" alt=""> Ice Melting

[Fast Particle-based Visual Simulation of Ice Melting](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2010.01810.x)
(Computer Graphics Forum, 2010) and
[Ice Melting Simulation with Water Flow Handling](https://people.cs.nycu.edu.tw/~cswingo/research/papers/Ice%20Melting%20Simulation_Lii_Wong.pdf)
(2014) explore particle heat transfer, melting ice, and flowing water.

| Demo | Scene | Motion | Video |
| :--- | :--- | :--- | :--- |
| Ice cubes | `cubes` | Ice cubes drift and melt in warm water | [8 seconds](../media/ice_melting/cubes.mp4) |
| Thawing arch | `arch` | Local heating weakens an ice arch into moving fragments | [8 seconds](../media/ice_melting/arch.mp4) |
| Bunny | `bunny` | An ice bunny melts into a pool on a heated copper plate | [8 seconds](../media/ice_melting/bunny.mp4) |

## Install and run

Run from the repository root with Python 3.10+, NVIDIA Warp 1.6.2, and
[Blender 4.2.9](https://download.blender.org/release/Blender4.2/). Install FFmpeg
and the DejaVu Sans fonts for video encoding and gallery labels.

```bash
python -m pip install -r ice_melting/requirements.txt
export BLENDER=/path/to/blender

python -m physstudio run ice cubes --gpu 0
python -m physstudio run ice arch --gpu 0
python -m physstudio fetch bunny
python -m physstudio run ice bunny --gpu 0
```

The cubes and arch are procedural. Bunny uses the downloaded Stanford mesh.
`PHYSSTUDIO_DATA_ROOT` selects an asset directory; `BLENDER` defaults to
`blender` on PATH, and `FFMPEG` selects the encoder executable.

Each run saves particle states, thermal and mechanical checks, ice/water meshes,
PNG renders, and `render/demo.mp4` under `outputs/ice/<scene>/`.
The full demo uses 240 frames at 30 fps, covering four seconds of mechanical
simulation. Heat evolves on an accelerated clock recorded in each scene preset.

A short pipeline check keeps the normal simulation interval between frames:

```bash
python -m physstudio run ice cubes --frames 2 --width 320 \
  --output outputs/ice/check_cubes --gpu 0
```

## Customize and replay

The dedicated harness exposes the simulation duration, particle spacing, and
render quality:

```bash
python -m ice_melting.harness run --scene bunny --gpu 0 \
  --output outputs/ice/custom_bunny --frames 240 --seconds 4 \
  --width 1280 --samples 48

# Rebuild surfaces and render a saved trajectory.
python -m ice_melting.harness run --scene bunny --gpu 0 \
  --output outputs/ice/custom_bunny --stage mesh
python -m ice_melting.harness run --scene bunny --gpu 0 \
  --output outputs/ice/custom_bunny --stage render
```

`--stage simulate` writes the particle trajectory. Use a fresh output directory
for a new simulation. Skip meshing when surfaces already exist. `--dx` sets particle spacing, `--thermal-scale` controls the heat clock,
and `--cpu` selects CPU rendering. Simulation uses Warp CUDA.

Bunny includes local plate-film drag and a thickness-aware water kernel: thin
water layers spread as a connected film, while deeper pools recover volumetric
reconstruction. `--settings overrides.json` accepts the boundary and surface
parameters listed in [simulate.py](simulate.py).

Package independent MP4s, animated GIFs, and a progression sheet:

```bash
python -m ice_melting.harness package --output outputs/ice/gallery \
  --renders outputs/ice/cubes/render outputs/ice/arch/render outputs/ice/bunny/render
```

## Method and assets

The Warp solver transports specific enthalpy, pays latent heat before particle
release, couples position-based liquid constraints to rigid ice fragments,
and resolves solid contact. Heat comes from warm water, the plate, ambient air,
and local sources. Cycles renders the reconstructed interfaces with refraction
and volume scattering. [METHOD.md](METHOD.md) gives the equations and checks.

The [Stanford Bunny](https://graphics.stanford.edu/data/3Dscanrep/) is fetched
from the [Open3D data release](https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/BunnyMesh.ply).
The downloader checks its pinned SHA-256 and writes `data/ice_melting/source.json`.
Its open scan boundaries are capped before solid voxel filling. The other
models, trays, and studio are procedural. [Sources](references/sources.json).

## Checks

```bash
python -m ice_melting.validate --check-bunny --output outputs/ice/numerics.json
python -m ice_melting.validate --skip-numerics \
  --simulations outputs/ice/bunny/simulation --meshes outputs/ice/bunny/meshes \
  --output outputs/ice/bunny_checks.json
```

Checks cover heat and mass conservation, the latent-heat plateau, rigid-body
stability, solid filling, phase connectivity, liquid density, local film drag,
and thin/deep water reconstruction. The harness also decodes the final video.

For temperature and phase-progress plots, install Matplotlib and run:

```bash
python -m pip install matplotlib
python -m ice_melting.diagnostics \
  --simulations outputs/ice/cubes/simulation outputs/ice/arch/simulation outputs/ice/bunny/simulation \
  --output outputs/ice/thermal_report
```
