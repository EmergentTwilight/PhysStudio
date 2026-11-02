# Thermomechanical model

The implementation combines particle enthalpy transport with a unified liquid/rigid-particle solver. Ice remains rigid until it has absorbed its latent heat. Melted particles retain their mass, position, and velocity and become liquid; loss of an ice connection creates independently moving fragments.

## Thermal state

Specific enthalpy `e` is the stored state, measured relative to ice at 0 °C:

```text
e < 0:       temperature = e / c_ice,       liquid fraction = 0
0 <= e <= L: temperature = 0,               liquid fraction = e / L
e > L:       temperature = (e-L) / c_water, liquid fraction = 1
```

Constants are `L = 333500 J/kg`, `c_ice = 2100 J/(kg K)`, `c_water = 4180 J/(kg K)`, and densities 917 and 1000 kg/m³. Conductivity interpolates between 2.2 and 0.6 W/(m K). All particles have mass `917 dx³`; phase fraction determines their thermal volume.

Heat is exchanged through a symmetric SPH conduction coefficient:

```text
G_ij = [4 k_i k_j / (k_i+k_j)] V_i V_j
       [-r_ij · grad W_ij / (|r_ij|² + 0.01 h²)]
m de_i/dt = sum_j G_ij (T_j-T_i) + Q_external,i
```

Pairwise heat cancels in the global energy balance. External heat includes exposed-surface convection, contact heating from the tray, and a spatially localized heat source. Surface exposure and orientation are estimated from kernel occupancy and its gradient. The local source uses Gaussian spatial falloff; it does not trace heat-source visibility. Numerical integration subcycles the thermal equation and records its diagonal stability number.

The heat clock runs faster than mechanical time to show melting within a short video. Each saved configuration records `thermal_scale`; the playback clock is also separate (240 frames at 30 fps for four mechanical seconds). These are accelerated thermal demonstrations, not laboratory melting-time predictions.

## Mechanics and phase interfaces

The support radius is `2.3 dx`. Liquid incompressibility uses unilateral position-based density constraints and seven solver iterations per step. Symmetric corrections transfer pressure and contact impulses to rigid ice particles. Akinci-style cohesion, curvature, and adhesion terms give liquid cohesion and ice/water wetting. Equal particle masses and symmetric pair terms preserve pairwise momentum before boundary response and correction limiting.

Ice particles share a rest-shape graph. After each saved time interval, particles with `e >= L` leave this graph; connected components become separate rigid bodies. Each body uses a least-squares rigid pose fitted to its corrected particles. Centroid and covariance reductions, SVD, and rigid projection arithmetic use double precision. Particle positions remain single precision. This distinction prevents accumulated artificial motion when many particles repeatedly contribute to a large rigid body.

The basin uses rigid contact bounds. After shape projection, a whole-body translation resolves any remaining body/basin overlap. Liquid particles receive contact projection, and contacting particles receive tangential floor friction. Meltwater is resolved with ordinary liquid particles.

The bunny additionally uses a local viscous closure for the thin film against the stationary copper plate. After velocity recovery, water within `2 dx` of the contact plane receives the backward-Euler update `v <- v / (1 + 80 dt w)`, where `w = max(1 - distance/(2 dx), 0)^2`. Ice and water beyond this layer are unchanged. This dissipative boundary treatment reduces the coarse-particle squeeze-out impulse when a bottom layer melts. Every particle retains its mass and enthalpy. The coefficient is a visual closure parameter; this is distinct from Lii and Wong's attached virtual-water reservoirs. Arch and floating-cube presets retain their original boundary treatment.

## Surface and image construction

Before particle sampling, the Stanford scan's five open boundary loops are capped. The resulting watertight mesh is voxelized and its interior is filled. The sampler verifies the filled volume against the mesh volume, and a regression test compares mass at two voxel resolutions. This prevents accidental hollow-shell ice.

Separate phase density fields use isotropic Gaussian width `0.65 dx` away from the plate. For the bunny's water film, tangential width increases smoothly to `1.0 dx` at the plate, while normal width decreases so the kernel determinant and integrated particle volume stay constant. The support returns to isotropic over `2 dx`. Weighted vertical variance of water neighbors within `3 dx` also controls this blend: a single layer receives a flattened kernel, and a deeper pool recovers an isotropic kernel. A smooth transition between variances `0.02 dx²` and `0.12 dx²` prevents separately reconstructed layers as the pool deepens. This boundary-aligned reconstruction represents thin pools without deleting isolated particles. It uses grid spacing `0.4 dx`; the isotropic mode retains `0.6 dx`. The normalized field integral is conserved, while the extracted isosurface volume remains resolution- and threshold-dependent.

Marching cubes extracts the 0.48 isosurface. Winding is oriented outward for correct refraction; validation checks positive signed volume. A small render-time smoothing pass reduces grid imprint. Ice and water use indices of refraction 1.309 and 1.333 respectively. Ice includes spatially varying volume scattering, while liquid remains clear. These are triangle-surface Cycles renders.

## Verification

`validate.py` checks:

- Equal/opposite two-particle heat flux and closed-system heat conservation.
- The full latent-heat plateau under a known heat input.
- Rigid shape recovery and a resting 18,679-particle body over repeated solver steps.
- Solid bunny filling and volume consistency across two particle spacings.
- Separation into two fragments when a single thermal bridge melts.
- Density recovery of a compressed water patch and surface-force momentum balance.
- Local film-drag dissipation, unchanged ice and distant-water velocities, and Gaussian volume normalization in both reconstruction modes.
- Complete trajectories, constant particle count and mass, finite states, rigid edge strain, basin floor clearance, and external-heat balance.
- Valid mesh indices and outward surface winding.

```bash
CUDA_VISIBLE_DEVICES=0 python -m ice_melting.validate \
  --simulations outputs/ice_melting/bunny_run/simulation \
  --meshes outputs/ice_melting/bunny_run/meshes \
  --output outputs/ice_melting/validation.json
```

Every simulation saves source hashes, initial states, per-frame states, and an energy ledger. Every gallery records video hashes and the source render directories.

## Literature choices

The 2010 ice-melting work supplies the central ideas of heat-driven particle release, liquid surface forces, and disconnected rigid ice. The water-flow paper by Lii and Wong is from **2014**, despite sometimes being abbreviated as LW10. Blom's 2014 SPH solidification thesis and subsequent heat-conduction studies motivate explicit temperature and latent-heat treatment. The direct enthalpy equation here is our implementation, not a reproduction of the 2017 paper's smoothed effective-heat-capacity formulation.

PBF and unified particle physics provide a practical GPU route to two-way liquid/rigid coupling. This representation was chosen to keep cold ice rigid while allowing water to flow, drip, and carry detached fragments. The [2024 SPH survey](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.15059) reviews the broader visual snow and ice literature. References and asset links are listed in [README.md](README.md).
