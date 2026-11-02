# Thermochemistry, water, and Gaussian appearance

The gas state contains fuel, oxygen, carbon dioxide, water vapor, nitrogen,
and carbonaceous residual. Oxygen-limited reactions conserve elemental mass
and release heat into a mixture with composition-dependent heat capacity.
Staggered velocities, buoyancy, viscosity, pressure projection, and solid-masked
advection transport the reacting gas through an open domain.

Water parcels follow gravity and two-way gas drag. SPH pressure handles dense
liquid regions; swept contact uses the scene SDF. Impacted water enters a finite
surface reservoir or continues as runoff. Particle/grid exchange transfers
sensible and latent heat, evaporated mass, and vapor dilution. The diagnostics
track emitted, moving, retained, escaped, and evaporated water independently.

The solid has finite heat capacity, pyrolysis, gas feedback, and optional
internal conduction. `wood_conductivity` controls diffusion, and
`ignition_regions` can initialize local heated regions. No hot boundary resets
the surface temperature after water cools it. Only nozzle operation is scheduled.
Semi-Lagrangian gas transport uses numerical diffusion; the local reaction and
water exchange are mass/energy balanced.

Firewoods and Playground reuse released FieryGS material registration. Bonsai
uses a fixed authored ornament region; its calibration and selection are saved
with the prepared material. One scene unit is treated as one meter for these
visual presets. All configurations are in [configs/](configs/).

## Appearance

The renderer integrates Planck/CIE flame emission, soot, and the evaporated-water
plume to the Gaussian expected depth. Optical detail resolves flame tongues
inside the simulated hot region. Emitting blocks illuminate nearby Gaussians;
spray parcels are rendered with velocity-aligned Gaussian covariances.

The saved `char` field is cumulative pyrolyzed mass. It defines a consumed
fraction of a thin exposed optical layer:

```text
capacity = density * exposed_faces * dx^2 * skin_depth
occupancy = 1 - clamp(pyrolyzed_mass / capacity, 0, 1)
rgb_offset = 0.5 * clamp((0.7 - occupancy) / 0.7, 0, 1)
```

Four nearby exposed cells bind to each combustible Gaussian. Its SH color is
reduced by `rgb_offset` before rasterization, retaining texture and view
dependence. The skin depth is an appearance parameter; presets use 1–1.3 mm.
Wet darkening is evaluated separately. Geometry remains fixed, and carbonization
persists after the hot gas and vapor leave.

The suppression plume represents evaporated water optically; the implementation
does not add a separate condensation-microphysics solve. The thermochemistry,
spray, and appearance are implemented in Warp / gsplat. The source references
and vendored CIE / SH provenance are in [references/sources.json](references/sources.json).
