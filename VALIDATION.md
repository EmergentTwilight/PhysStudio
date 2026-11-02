# Release validation

Validated from the PhysStudio checkout on Linux / RTX 3090 with the environment
listed in [INSTALL.md](INSTALL.md).

| Area | Completed checks |
| :--- | :--- |
| Rain | Shallow-water and particle checks; GPU occlusion; simulation and rendering |
| Splashing | Density relaxation and contact balance; simulation and rendering |
| Locomotion | FEM assembly against the NumPy reference; contact, affine binding, Gaussian splitting; simulation and rendering |
| Magnetism | Nonlinear field solve and discrete operators; Gaussian binding; simulation, rendering, trajectory audit |
| Snow | Free fall, plasticity, friction, deposition; depth rendering; simulation, rendering, mass/contact audit |
| Lava | Free fall, mass and heat balance, rheology, impact; all three scene simulations, rendering, and replay audits |
| Water + Sand | All three pipelines; conservation, porous equilibrium, resolution-scaled drag, grain attachment, and same-state rendering |
| Plant Wilting | Public-asset hashes, isolated Newton installation, all three pipelines, transport and binding checks |
| Storms | Moisture and latent heat, pressure projection, wall contact, dust transport, occlusion; all three pipelines and frozen-state rendering |
| Lightning | Poisson accuracy, gauge invariance, conductor enhancement, charge balance, occlusion; all three pipelines and frozen-state rendering |
| Capillary Contact | Energy/contact derivatives, surface-current operators, solid gaps, volume; all three simulation/render pipelines |
| Ice Melting | Heat and latent energy, rigid ice, liquid density, wetting kernels; all three simulation/render pipelines |
| Launcher | All 36 presets; existing 33 command plans unchanged |
| Downloads | ZIP extraction, CRC-checked cache, destination confinement; SuperSplat manifest, conversion, and camera setup |
| Media | All 37 MP4s decoded; preview files and hashes verified; MeGAS full-frame looping preview |
| Package | Wheel includes all method modules, reference attribution, and Plant Wilting and Ice Melting runtime/download metadata |
| Source | Python syntax, English text, portable paths, local documentation links |

Each end-to-end pipeline check used two simulation/video steps. The Playground
rain showcase was also regenerated for its full 240 frames from the release
preset. The other showcase videos are the completed runs collected from the
original experiments. Sixteen core physics and rendering modules retain the
same Python syntax trees after import-path normalization and formatting. Five
LavaGS core modules also match their experiment versions after import-path
normalization. The packaged Calico demo uses the final 80-degree orbit.

Run the packaging check from the repository root:

```bash
python tools/check_release.py
```

The method READMEs list their numerical and GPU validation commands.

Thirteen Water + Sand solver, rendering, and validation modules match the
development version after import-path normalization and formatting. Shared
camera, video, and sampling helpers also match. Each of its three presets
completed a two-frame simulation, Gaussian render, and video audit at the full
128 × 84 × 75 simulation resolution. Same-state rendering was compared at
1280 × 800. The showcase contains the completed 180-frame trajectories.

Seven Plant Wilting core modules match the development version after import-path
normalization. The release workflow checks all three two-frame pipelines at
640 × 360, using the pinned public models and Newton installation. The Ficus
renderer reproduces five full-resolution showcase frames pixel for pixel.
The three plant showcase videos contain the complete reviewed trajectories.

Four StormGS core modules match the development version after import-path
normalization and formatting. Each of its three presets completed a two-frame
simulation, Gaussian render, and video audit. Nine frozen showcase keyframes
reproduce the development renders pixel for pixel at native resolution. The
three showcase videos contain 240 frames each. The wheel includes the StormGS
modules, scene calibrations, run guide, and reference metadata.

ThunderGS preserves the development solver and renderer after formatting.
All three presets regenerate their three discharges and complete a two-frame
Gaussian render, video audit, and gallery build. Nine full-resolution frozen
keyframes reproduce the development images pixel for pixel. The three released
videos contain 240 frames at 30 fps; the wheel includes all ThunderGS modules,
scene calibrations, run guide, and paper/asset references.

Capillary Contact keeps ten simulation and numerical-check modules byte-identical
to the accepted development version. Each exported preset completed a two-frame
simulation at the full 96 x 77 resolution, its numerical and trajectory checks,
a Cycles render, and video decoding. The complete showcase contains the accepted
180-frame orchard and 240-frame stirred cluster and paperclip trajectories.
All nine published MP4/GIF/JPG files match the approved originals; every GIF
frame and all three videos decode. Fresh Poly Haven downloads match the orchard
source assets. The wheel includes the solver, renderer, checks, run guide, and
paper/asset references.

Ice Melting keeps five core modules byte-identical to the development version;
its renderer matches after root-path normalization. All three presets completed
two-frame pipelines at the full particle resolution, including numerical checks,
mesh reconstruction, Cycles rendering, and video encoding. Cubes and arch run
without downloaded models. A fresh Bunny download matches its pinned SHA-256.
Two frozen Bunny frames were compared at 1280 x 800 with a maximum channel
difference of one 8-bit intensity level. The released videos match the three
approved 240-frame originals, including Bunny's updated wetting film; every GIF
frame decodes. The wheel includes the solver, renderer, checks, run guide, and
source metadata.
