"""FieryGS geometry and material registration for fire suppression."""
from pathlib import Path
import copy, json, hashlib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt
from lava_gs.scene import Scene
from lava_gs.cameras import cascade_camera

from physstudio.paths import data_path

PACKAGE = Path(__file__).resolve().parent


def asset_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else data_path(path)


PRESETS = {
    "firewoods": dict(
        asset="rainy_gs/fierygs/firewoods_sand_dark",
        origin=[-2.4, -2.4, -2.55],
        shape=[80, 80, 100],
        dx=0.06,
        wood_min=[-2.1, -2.1, -3.7],
        wood_max=[1.5, 1.5, -0.1],
        warmup=2.5,
        seconds=10.0,
        water_start=1.5,
        water_stop=7.3,
        nozzle=[1.25, -0.4, -1.10],
        target=[-0.24, -0.43, -1.83],
        sweep=0.30,
        flow=0.22,
        speed=8.0,
        cone=0.085,
        droplet_radius=0.0006,
        parcel_mass=0.00006,
        wind=[0.12, -0.04, 0.0],
        wood_density=16.0,
        wood_initial_temperature=820.0,
        pyrolysis_rate=0.008,
        wood_transfer=220.0,
        wood_feedback=0.16,
        camera=dict(
            target=[-0.25, -0.38, -1.25],
            fov=49,
            orbit=dict(
                azimuth_start=28,
                azimuth_end=85,
                radius_start=3.6,
                radius_end=3.1,
                height_start=1.20,
                height_end=1.35,
            ),
        ),
        exposure=0.72,
        flame_gain=3.5,
        smoke_extinction=4.0,
        steam_extinction=2.2,
        title="Firewoods | Water-jet suppression",
    )
}

PRESETS["playground"] = dict(
    PRESETS["firewoods"],
    asset="rainy_gs/fierygs/Playground",
    scene_config={},
    prepared_material="firex_gs/prepared/playground/material.npz",
    origin=[-1.7, -1.5, -0.65],
    shape=[88, 88, 104],
    dx=0.04,
    nozzle=[0.12, -0.4, 1.45],
    target=[0.05, 0.25, -0.10],
    cone=0.64,
    sweep=0.5,
    flow=2.0,
    parcel_mass=0.0004,
    speed=4.8,
    water_start=2.0,
    water_stop=10.0,
    wood_initial_temperature=790.0,
    pyrolysis_rate=0.006,
    wood_transfer=110.0,
    wood_conductivity=0.15,
    warmup=2.0,
    exposure=0.85,
    smoke_extinction=2.2,
    camera=dict(
        target=[0.05, 0.2, 0.15],
        fov=50,
        orbit=dict(
            azimuth_start=-142,
            azimuth_end=-105,
            radius_start=3.4,
            radius_end=3.1,
            height_start=1.05,
            height_end=1.1,
        ),
    ),
    title="Playground | Timber suppression",
)

PRESETS["bonsai"] = dict(
    PRESETS["firewoods"],
    asset="firex_gs/graphdeco/bonsai",
    scene_config="firex_gs/prepared/bonsai/scene_config.json",
    prepared_material="firex_gs/prepared/bonsai/material.npz",
    title="Bonsai | Canopy suppression",
)


def load(name, overrides=None):
    cfg = copy.deepcopy(PRESETS[name])
    cfg.update(overrides or {})
    config = cfg.get("scene_config", str(PACKAGE / "configs/firewoods_scene.json"))
    config = (
        copy.deepcopy(config)
        if isinstance(config, dict)
        else json.loads(asset_path(config).read_text())
    )
    config.update(background=[0.4, 0.45, 0.49], far_plane=120.0)
    scene = Scene(asset_path(cfg["asset"]), config=config)
    return scene, cfg


def camera(cfg, t, width=1280):
    return cascade_camera(t, width, settings=cfg["camera"])


def geometry(scene, cfg):
    shape = np.array(cfg["shape"])
    dx = cfg["dx"]
    origin = np.array(cfg["origin"])
    coords = np.stack(
        np.meshgrid(
            *[origin[d] + (np.arange(shape[d]) + 0.5) * dx for d in range(3)],
            indexing="ij",
        ),
        -1,
    )
    if cfg.get("prepared_material"):
        return prepared_geometry(scene, cfg, coords)
    path = asset_path(cfg["asset"]) / "voxel_grid.npy"
    official = np.load(path) > 0
    lower = np.array(cfg["wood_min"])
    upper = np.array(cfg["wood_max"])
    import torch

    mapping = data_path("firex_gs/fierygs/firewoods_sand_dark")
    mask = torch.load(
        mapping / "mask_pts_in_grids.pth", map_location="cpu", weights_only=True
    ).numpy()
    ids = torch.load(
        mapping / "indices_pts_in_grids.pth", map_location="cpu", weights_only=True
    ).numpy()
    xyz = scene.means.detach().cpu().numpy()[mask]
    dx_original = (upper - lower) / np.array(official.shape)
    fractional = (xyz - lower) / dx_original
    errors = np.maximum(ids - fractional, fractional - ids - 1.0)
    if errors.max() > 0.001:
        raise ValueError("Released voxel-to-Gaussian registration mismatch")
    # Voxel dimensions come from the released array, not the newer YAML's 256.
    centers = lower + (np.argwhere(official) + 0.5) * (upper - lower) / np.array(
        official.shape
    )
    cells = np.floor((centers - origin) / dx).astype(int)
    valid = ((cells >= 0) & (cells < shape)).all(-1)
    cells = cells[valid]
    wood = np.zeros(tuple(shape), dtype=bool)
    wood[tuple(cells.T)] = True
    wood = binary_fill_holes(binary_closing(wood, iterations=1))
    ground = scene.ground.detach().cpu().numpy()
    floor = coords[:, :, 0, 0] * ground[0] + coords[:, :, 0, 1] * ground[1] + ground[2]
    solid = wood | (coords[..., 2] < floor[..., None])
    outside = distance_transform_edt(~solid) * dx
    inside = distance_transform_edt(solid) * dx
    sdf = (outside - inside).astype("f")
    exits = (
        distance_transform_edt(solid, return_distances=False, return_indices=True)
        .transpose(1, 2, 3, 0)
        .astype("i")
    )
    metadata = dict(
        official_voxel_shape=list(official.shape),
        official_voxel_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        wood_bounds=[centers.min(0).tolist(), centers.max(0).tolist()],
        wood_cells=int(wood.sum()),
        ground=ground.tolist(),
        mapping_max_cell_error=float(errors.max()),
        mapped_gaussians=int(mask.sum()),
    )
    return (
        dict(
            solid=solid.astype("i"),
            wood=wood.astype("i"),
            sdf=sdf,
            exits=exits,
            floor=floor.astype("f"),
            coords=coords.astype("f"),
        ),
        metadata,
    )


def prepared_geometry(scene, cfg, coords):
    path = asset_path(cfg["prepared_material"])
    material = dict(np.load(path))
    shape = np.array(cfg["shape"])
    origin = np.array(cfg["origin"])
    dx = cfg["dx"]

    def voxelize(points):
        cells = np.floor((points - origin) / dx).astype(int)
        valid = ((cells >= 0) & (cells < shape)).all(-1)
        cells = cells[valid]
        grid = np.zeros(tuple(shape), dtype=bool)
        grid[tuple(cells.T)] = True
        return grid

    wood = voxelize(material["fuel_points"])
    wood = binary_fill_holes(
        binary_closing(wood, iterations=cfg.get("wood_closing", 1))
    )
    obstacles = voxelize(material["collision_points"])
    ground = material["ground"]
    floor = coords[:, :, 0, 0] * ground[0] + coords[:, :, 0, 1] * ground[1] + ground[2]
    # Preserve non-combustible geometry as collision boundaries.
    solid = wood | obstacles | (coords[..., 2] < floor[..., None])
    outside = distance_transform_edt(~solid) * dx
    inside = distance_transform_edt(solid) * dx
    sdf = (outside - inside).astype("f")
    exits = (
        distance_transform_edt(solid, return_distances=False, return_indices=True)
        .transpose(1, 2, 3, 0)
        .astype("i")
    )
    ids = material["gaussian_ids"].astype("i")
    meta = dict(
        material_source=str(path),
        material_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        wood_bounds=[
            material["fuel_points"].min(0).tolist(),
            material["fuel_points"].max(0).tolist(),
        ],
        wood_cells=int(wood.sum()),
        collision_cells=int(solid.sum()),
        mapped_gaussians=len(ids),
        ground=ground.tolist(),
    )
    geo = dict(
        solid=solid.astype("i"),
        wood=wood.astype("i"),
        sdf=sdf,
        exits=exits,
        floor=floor.astype("f"),
        coords=coords.astype("f"),
        gaussian_ids=ids,
        registration_error=material.get("registration_error", np.array(0.0)),
    )
    return geo, meta
