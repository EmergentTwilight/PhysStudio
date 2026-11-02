# Credits

PhysStudio contains physics-based implementations and demo workflows inspired
by the papers linked in each method's README.

Runtime dependencies are installed separately: [PyTorch](https://pytorch.org/),
[NVIDIA Warp](https://github.com/NVIDIA/warp),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[OSQP](https://github.com/osqp/osqp-python), NumPy, SciPy, Pillow, and plyfile.
Video encoding uses [FFmpeg](https://ffmpeg.org/).

Pretrained scene and object models come from FieryGS, Graphdeco, Voxel51,
PhysGaussian, and the SuperSplat authors credited in the LavaGS presets; [ASSETS.md](ASSETS.md) links each download and source.
Models, camera archives, paper PDFs, and third-party binary packages are not
included in this repository. The implementation demos are rendered by PhysStudio using those publicly released assets.

The MeGAS showcase uses the official Bidirectional Phase-Change Editing video from the [MeGAS project page](https://zju3dv.github.io/MeGAS/). The README plays a full-frame animated loop, and the video gallery autoplays the original MP4. The source URL and hash are recorded in `media/manifest.json`. [MeGAS code](https://github.com/zju3dv/MeGAS).

Water + Sand adapts constitutive and coupling routines from the
[released 2D LBM–MPM implementation](https://github.com/DacianShaw/LBM-MPM-Air-Water-Sand)
by Xiaoyu Xiao; its [MIT notice](water_sand/LICENSE.reference) is included.
[OpenHOMELBM](https://github.com/kuiwuchn/OpenHOMELBM) provides the 3D Warp LBM
organization reference. Pinned revisions are in
[water_sand/references/sources.json](water_sand/references/sources.json).

Plant Wilting uses the water-transport formulation from
[Filippo Maggioli and collaborators](https://github.com/filthynobleman/plant-water-diffusion),
with its [MIT notice](plant_wilting/LICENSE.reference), and
[Newton VBD rods](https://github.com/newton-physics/newton) as the GPU mechanical backend.
Plant Gaussians are downloaded from [DiffWind](https://github.com/zju3dv/DiffWind).
[Pinned sources](plant_wilting/references/sources.json) and
[asset hashes](plant_wilting/references/assets.json) accompany the workflow.

StormGS implements the moist-flow model from
[Cyclogenesis: Simulating Hurricanes and Tornadoes](https://doi.org/10.1145/3658149).
The authors' [CycloCore notebook](https://github.com/AgrosAmad/CycloCore)
(MPL 2.0) supplies reference equations; the Warp solver is implemented here.
The scene assets are shared with RainyGS and LavaGS.
[Pinned sources](storm_gs/references/sources.json).

ThunderGS implements the electrical-discharge model from
[Thunderstruck: Visually Simulating Electrical Storms](https://jonathank.de/research/paper/Herrera2025_thunderstruck.pdf),
with a neutral floating-channel correction and Gaussian-scene lighting.
Firewoods and Train reuse the RainyGS assets; Zohar is the CC BY 4.0 scene
by barak9177 used in LavaGS. [Sources](thunder_gs/references/sources.json).

Capillary Contact follows the surface-energy and rigid-contact formulation of
[Solid-Fluid Interaction with Surface-Tension-Dominant Contact](https://arxiv.org/abs/2105.08471)
(SIGGRAPH 2021), with a height-field fluid adapter and transported surface currents.
[Blender](https://www.blender.org/) supplies Cycles rendering; the orchard models
come from [Poly Haven](https://polyhaven.com/) under CC0.
[Sources](surface_tension/references/sources.json).
