# <img src="../media/project-icons/fire-x-gs.svg" width="40" height="40" align="absmiddle" alt=""> Fire-X GS · fire suppression in Gaussian scenes

[Fire-X: Extinguishing Fire with Stoichiometric Heat Release](https://helgewrede.github.io/firex/)
(SIGGRAPH Asia 2025) supplies the coupled combustion and water-energy model.
[FieryGS](https://pku-vcl-geometry.github.io/FieryGS/) supplies Gaussian fire
rendering and surface carbonization. A Warp solver advances the gas, heated
material, liquid spray, retained water, and evaporated mass.

## Run

After [installation](../INSTALL.md), download the Gaussian scenes and material
registration files, then run from the repository root:

```bash
python -m firex_gs.fetch --scene all
python -m physstudio run firex firewoods --gpu 0
python -m physstudio run firex playground --gpu 0
python -m physstudio run firex bonsai --gpu 0
```

Use `--scene firewoods`, `--scene playground`, or `--scene bonsai` with the
downloader to fetch one scene. Set `PHYSSTUDIO_DATA_ROOT` to reuse assets from
another directory. Playground and Bonsai material maps are prepared on the
first run. [Asset sources](../ASSETS.md).

| Scene | Effect | Frames | Grid |
| --- | --- | ---: | --- |
| Firewoods | Overhead spray on burning logs | 420 | 104 × 104 × 136 |
| Playground | Sprinkler over wooden towers and metal slides | 450 | 88 × 88 × 104 |
| Bonsai | Diagonal spray on an indoor ornament | 450 | 96 × 96 × 128 |

All presets use 30 fps, moving cameras, and a 1/180-second simulation step.
The three showcase videos are in [the gallery](../media/index.html#fire-x-gs).

A short run uses the same scene resolution and frame interval:

```bash
python -m physstudio run firex bonsai --gpu 0 \
  --frames 2 --width 320 --output outputs/firex/bonsai_check
```

Outputs under `outputs/firex/<scene>/` include `render/firex_demo.mp4`, PNGs,
lossless simulation states, camera matrices, material statistics, source
snapshots, diagnostics, and an `index.html` report. Choose a fresh output for
each run. `--dry-run` prints the launcher's command.

## Replay and customize

Presets are in [configs/](configs/). Re-render saved states with another width
or a JSON containing camera, exposure, or emission settings:

```bash
CUDA_VISIBLE_DEVICES=0 python -m firex_gs.refine \
  --input outputs/firex/bonsai/simulation \
  --output outputs/firex/bonsai_replay --width 1280
```

Use `--frames 0 90 225 449` to render selected states, or `--config look.json`
to supply appearance overrides. Surface carbonization is recovered separately
for each saved frame and stays attached to the registered Gaussians.

Run a matched no-water control and include both heat-release histories:

```bash
python -m firex_gs.harness --scene playground --gpu 0 \
  --output outputs/firex/playground_control --dry-diagnostics
```

Use `--dry-control` to render the control and create a side-by-side video too.
Both variants share the initial state, material, gas parameters, and camera.

## Scene materials and rendering

Firewoods uses the released occupied cells and Gaussian registration.
Playground selects the released wood label and uses nearby scene geometry for
collision; the metal slides stay outside the carbonization mask. Bonsai uses
an authored combustible ornament region in a calibrated frame. Its pot, stand,
and table remain noncombustible collision geometry.

Inspect the generated material mask with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m firex_gs.prepare --scene playground --inspect
CUDA_VISIBLE_DEVICES=0 python -m firex_gs.prepare --scene bonsai --inspect
```

Spectral flame color uses Planck emission and the CIE matching table. Emitting
volume blocks provide local diffuse and specular light. FieryGS's per-Gaussian
carbonization response converts consumed surface fuel into persistent color
change; water contributes cooling, evaporation, runoff, and wet darkening.
[Model details](METHOD.md) and [paper / asset references](references/sources.json).

## Checks

The harness runs reaction, phase-energy, drag, pressure, solid-boundary, and
wood-conduction checks before simulation. It then checks material registration,
SH evaluation, carbonization, species balance, water accounting, and complete
video decoding. The matched control checks the initial states and solver hashes.

```bash
CUDA_VISIBLE_DEVICES=0 python -m firex_gs.validate --output outputs/firex_numerics.json
CUDA_VISIBLE_DEVICES=0 python -m firex_gs.validate_wood --output outputs/firex_wood.json
python -m firex_gs.audit --input outputs/firex/playground_control
```
