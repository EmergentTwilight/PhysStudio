# <img src="../media/project-icons/enliveninggs.svg" width="40" height="40" align="absmiddle" alt=""> EnliveningGS · active Gaussian locomotion

[EnliveningGS: Active Locomotion of 3DGS](https://gapszju.github.io/EnliveningGS/)
animates Gaussian objects through muscle actuation, deformable simulation,
and frictional contact.

## Run

After [installation](../INSTALL.md), download and animate the objects:

```bash
python -m physstudio fetch bread
python -m physstudio fetch wolf
python -m physstudio fetch vasedeck
python -m physstudio run enlivening bread --gpu 0
python -m physstudio run enlivening wolf --gpu 0
python -m physstudio run enlivening vasedeck --gpu 0
```

The first run prepares the Gaussian object and its coordinate transform.
Each simulation builds a tetrahedral cage, records `trajectory.npz`, and renders
`render/demo.mp4` under `outputs/enlivening/<scene>/`. Bread and vase hop; the
wolf turns. The camera orbits 24 degrees during the clip.

## Implementation

Warp evaluates Neo-Hookean FEM forces and Hessians; an implicit solve advances
the cage. A reduced OSQP problem selects muscle actuation, with a penalty
prediction and Coulomb LCP correction for ground contact. The reduced QP and
contact active set run on CPU; FEM assembly and PCG run on CUDA.

Gaussians use a center-plus-axis embedding. Deformation transports covariance
as `F Σ Fᵀ`, and the polar rotation transports spherical-harmonic directions.
Adaptive splitting preserves the Gaussian moments when the embedding bends.
The vase adapter fits the supporting deck plane in depth space before cutting
and regularizing intersecting splats.

| Preset | Motion | Young's modulus | Density | Mass profile |
| :--- | :--- | ---: | ---: | :--- |
| Bread | Hop | 18,000 | 40 | Uniform |
| Wolf | Turn | 18,000 | 40 | Uniform |
| Vase | Hop | 30,000 | 100 | Base-heavy |

All showcase presets use 120 Hz, cage resolution 7, friction 0.65, and 2.1 seconds
of simulated motion, rendered at half speed. Units follow the normalized object
scale. Use `python -m enlivening_gs.locomotion --help` for the material, contact,
and motion controls; `python -m enlivening_gs.render --help` controls playback.

## Checks

```bash
CUDA_VISIBLE_DEVICES=0 python -m enlivening_gs.validate_warp --scenes bread
CUDA_VISIBLE_DEVICES=0 python -m enlivening_gs.validate_contact_binding
```

Prepare Bread first with `python -m enlivening_gs.assets --scene bread`.
Validation compares Warp assembly to the NumPy FEM reference and checks contact,
affine Gaussian binding, and moment-preserving splitting.
Assets and download links: [ASSETS.md](../ASSETS.md).
