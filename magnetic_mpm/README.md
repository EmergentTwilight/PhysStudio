# <img src="../media/project-icons/magnetic-mpm.svg" width="40" height="40" align="absmiddle" alt=""> Magnetic MPM · fields meet Gaussian matter

[A Material Point Method for Nonlinearly Magnetized Materials](https://binwangbfa.github.io/publication/siga21_mpm/)
couples nonlinear magnetization to MPM for magnetic fluids and deformable solids.

## Run

After [installation](../INSTALL.md), download the two Gaussian models:

```bash
python -m physstudio fetch bread
python -m physstudio fetch wolf
python -m physstudio run magnetic ferrofluid --gpu 0
python -m physstudio run magnetic magnetic_bread --gpu 0
python -m physstudio run magnetic magnetic_wolf --gpu 0
```

Each run writes material trajectories, magnetic-solver diagnostics, an audit,
and `render/demo.mp4` under `outputs/magnetic/<scene>/`. The liquid scene surrounds
a Gaussian bread object; the solid scenes deform bread and wolf splats.
The showcase camera orbits 22 degrees.

## Implementation

The Warp implementation solves a nonlinear Langevin magnetic potential with
Newton-PCG on a Cartesian grid, using adjoint-consistent gradient/divergence
operators and a homogeneous Neumann boundary for the induced potential.
Maxwell stress divergence supplies the magnetic force to MLS-MPM. A prescribed
moving permanent magnet drives viscous liquid or Neo-Hookean solid material.

The solid presets clamp the lower support region. Affine particle bindings
transport Gaussian centers and covariance `F Σ Fᵀ`; polar rotations transport
spherical-harmonic appearance. The liquid uses the anisotropic fluid renderer
shared with Gaussian Splashing.

`harness.py` stores the grid, time step, magnet, material, and clamp presets.
Use `python -m magnetic_mpm.demo --help` to adjust them, or
`python -m magnetic_mpm.render --help` to change orbit and playback.

## Checks and control

```bash
CUDA_VISIBLE_DEVICES=0 python -m magnetic_mpm.validate
CUDA_VISIBLE_DEVICES=0 python -m magnetic_mpm.validate_binding --input outputs/magnetic/magnetic_bread
python -m magnetic_mpm.harness audit --input outputs/magnetic/magnetic_bread
python -m magnetic_mpm.harness run --case ferrofluid_control \
  --output outputs/magnetic/ferrofluid_control --gpu 0
python -m magnetic_mpm.harness compare \
  --magnetic outputs/magnetic/ferrofluid \
  --control outputs/magnetic/ferrofluid_control
```

Checks cover nonlinear residuals, constitutive response, discrete adjointness,
contact, positive deformation determinants, and affine Gaussian binding.
The zero-magnet control preserves the material, time step, and moving geometry.
Assets and download links: [ASSETS.md](../ASSETS.md).
