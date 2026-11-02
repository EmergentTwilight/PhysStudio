"""FieryGS surface carbonization and voxel illumination for Fire-X replay.

References: FieryGS rendering/scene_render_fire.py and
gaussian_renderer/renderer_with_fire.py. Upstream files and assets are unchanged.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from scipy.spatial import cKDTree
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from .render_spectral import Renderer as SpectralRenderer
from .scenes import asset_path
from physstudio.paths import data_path


def carbon_offset(occupancy, threshold=0.7):
    """FieryGS's subtractive RGB response, before Gaussian rasterization."""
    return 0.5 * ((threshold - occupancy) / threshold).clamp(0, 1)


def surface_registration(scene, cfg, geometry):
    """Bind released burnable splats to exposed thermal cells once."""
    if "gaussian_ids" in geometry:
        mask = np.zeros(len(scene.means), dtype=bool)
        mask[geometry["gaussian_ids"]] = True
        error = float(geometry["registration_error"])
    else:
        mapping = data_path("firex_gs/fierygs/firewoods_sand_dark")
        mask = (
            torch.load(
                mapping / "mask_pts_in_grids.pth", map_location="cpu", weights_only=True
            )
            .numpy()
            .astype(bool)
        )
        official_ids = torch.load(
            mapping / "indices_pts_in_grids.pth", map_location="cpu", weights_only=True
        ).numpy()
    xyz = scene.means.detach().cpu().numpy()[mask]
    if "gaussian_ids" not in geometry:
        lower = np.asarray(cfg["wood_min"])
        upper = np.asarray(cfg["wood_max"])
        official_shape = np.load(asset_path(cfg["asset"]) / "voxel_grid.npy").shape
        fractional = (xyz - lower) / ((upper - lower) / official_shape)
        error = float(
            np.maximum(official_ids - fractional, fractional - official_ids - 1).max()
        )
    if error > 0.001:
        raise ValueError("Released Gaussian material registration is inconsistent")
    solid = geometry["solid"] > 0
    wood = geometry["wood"] > 0
    padded = np.pad(solid, 1, constant_values=False)
    faces = np.zeros(wood.shape, dtype=np.float32)
    for axis in range(3):
        for offset in [-1, 1]:
            slices = [slice(1, n + 1) for n in wood.shape]
            slices[axis] = slice(1 + offset, wood.shape[axis] + 1 + offset)
            faces += ~padded[tuple(slices)]
    surface = wood & (faces > 0)
    cells = np.argwhere(surface)
    centers = np.asarray(cfg["origin"]) + (cells + 0.5) * cfg["dx"]
    distance, indices = cKDTree(centers).query(xyz, k=4)
    if distance[:, 0].max() > 4 * cfg["dx"]:
        raise ValueError("Burnable Gaussians are too far from the thermal surface")
    weights = np.exp(-np.square(distance / (cfg["dx"] * 0.7)))
    weights /= np.maximum(weights.sum(-1, keepdims=True), 1e-30)
    return (
        mask,
        cells,
        faces[surface],
        indices,
        weights.astype("f"),
        {
            "gaussians": int(mask.sum()),
            "thermal_surface_cells": len(cells),
            "mapping_error_cells": error,
            "nearest_surface_distance_max": float(distance[:, 0].max()),
            "nearest_surface_distance_median": float(np.median(distance[:, 0])),
        },
    )


@wp.kernel
def illuminate(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    base: wp.array(dtype=wp.vec3),
    lights: wp.array(dtype=wp.vec3),
    radiance: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.vec3),
    eye: wp.vec3,
    gain: float,
):
    i = wp.tid()
    p = points[i]
    view = wp.normalize(eye - p)
    n = normals[i]
    if wp.dot(n, view) < 0.0:
        n = -n
    diffuse = wp.vec3(0.0)
    specular = wp.vec3(0.0)
    for j in range(lights.shape[0]):
        delta = lights[j] - p
        d2 = wp.dot(delta, delta)
        direction = delta / wp.sqrt(wp.max(d2, 0.01))
        cosine = wp.max(wp.dot(n, direction), 0.0)
        reflection = 2.0 * wp.dot(n, direction) * n - direction
        glossy = wp.pow(wp.max(wp.dot(view, reflection), 0.0), 32.0)
        weight = gain / (d2 + 1.0)
        diffuse += radiance[j] * (0.8 * cosine * weight)
        specular += radiance[j] * (0.5 * glossy * weight)
    b = base[i]
    linear = wp.vec3(
        wp.pow(wp.max(b[0], 0.0), 2.2),
        wp.pow(wp.max(b[1], 0.0), 2.2),
        wp.pow(wp.max(b[2], 0.0), 2.2),
    )
    # Keep the original appearance at zero illumination; charcoal reflects less.
    linear += wp.cw_mul(linear, diffuse) + specular * 0.04
    result[i] = wp.vec3(
        wp.pow(wp.max(linear[0], 0.0), 1.0 / 2.2),
        wp.pow(wp.max(linear[1], 0.0), 1.0 / 2.2),
        wp.pow(wp.max(linear[2], 0.0), 1.0 / 2.2),
    )


class Renderer(SpectralRenderer):
    def __init__(self, scene, cfg, geometry):
        cfg = dict(cfg, legacy_surface=False)
        super().__init__(scene, cfg, geometry)
        mask, cells, faces, ids, weights, self.registration = surface_registration(
            scene, cfg, geometry
        )
        self.roi = torch.as_tensor(np.flatnonzero(mask), device=self.device)
        self.cells = tuple(cells.T)
        self.faces = faces
        self.bind_ids = ids
        self.bind_weights = weights
        self.base_sh = scene.sh
        center = scene.means[self.roi].mean(0)
        self.light_roi = torch.where(
            torch.linalg.vector_norm(scene.means - center, dim=-1)
            < cfg.get("illumination_radius", 4.0)
        )[0]
        self.base_opacity = scene.opacity
        self.material_stats = {}

    def update(self, state):
        super().update(state)
        cfg = self.cfg
        # `char` stores cumulative pyrolyzed mass, not the residual char mass.
        # Convert it into the consumed fraction of a thin exposed wood skin.
        mass = state["char"][self.cells]
        depth = cfg.get("carbon_skin_depth", 0.001)
        capacity = cfg["wood_density"] * self.faces * cfg["dx"] ** 2 * depth
        consumed = np.clip(mass / np.maximum(capacity, 1e-12), 0.0, 1.0)
        occupancy = (1.0 - consumed)[self.bind_ids]
        self.occupancy = torch.as_tensor(
            (occupancy * self.bind_weights).sum(-1), device=self.device
        )
        self.offset = carbon_offset(self.occupancy, cfg.get("carbon_threshold", 0.7))
        self.wood_temperature = torch.as_tensor(
            (state["wt"][self.cells][self.bind_ids] * self.bind_weights).sum(-1),
            device=self.device,
        )
        wet = state["wet"][self.cells] / np.maximum(
            self.faces * cfg["dx"] ** 2 * 1000 * 0.00015, 1e-12
        )
        self.surface_wet = torch.as_tensor(
            (np.clip(wet, 0, 1)[self.bind_ids] * self.bind_weights).sum(-1),
            device=self.device,
        )
        self.lights, self.radiance = self.voxel_lights(state)
        self.material_stats = dict(
            time=float(state["time"]),
            carbonized_gaussians=int((self.offset > 0).sum()),
            fully_carbonized_gaussians=int((self.occupancy < 0.01).sum()),
            mean_rgb_subtraction=float(self.offset.mean()),
            occupancy_mean=float(self.occupancy.mean()),
            hot_voxel_lights=len(self.lights),
            light_rgb_sum=self.radiance.sum(0).tolist(),
        )

    def voxel_lights(self, state):
        """Energy-weighted blocks of the same emitting volume as the fire pass."""
        t = wp.to_torch(self.thermal)[::2, ::2, ::2]
        field = wp.to_torch(self.field)
        index = ((t - 300) * (1023 / 2700)).clamp(0, 1022.99)
        lo = index.long()
        weight = index - lo
        table = wp.to_torch(self.table)
        rgb = table[lo] * (1 - weight[..., None]) + table[lo + 1] * weight[..., None]
        emission = (
            field[..., 1].clamp(0, 4) * 0.25 + field[..., 3].clamp_min(0).sqrt() * 0.8
        )
        rgb *= emission[..., None] * self.look.emission * self.cfg["dx"] ** 3
        stride = 4
        flux = (
            F.avg_pool3d(rgb.permute(3, 0, 1, 2)[None], stride, stride)[0].permute(
                1, 2, 3, 0
            )
            * stride**3
        )
        score = flux.sum(-1)
        indices = torch.nonzero(score > 1e-7)
        if len(indices) > self.cfg.get("light_count", 96):
            values = score[tuple(indices.T)]
            order = values.topk(self.cfg.get("light_count", 96)).indices
            indices = indices[order]
        colors = flux[tuple(indices.T)].contiguous()
        positions = (indices.to(torch.float32) + 0.5) * (
            stride * self.cfg["dx"]
        ) + torch.tensor(self.cfg["origin"], device=self.device)
        return positions.contiguous(), colors

    @torch.no_grad()
    def surface_colors(self, cam, lighting=True, carbon=True):
        scene = self.scene
        # Evaluate SH in the asset frame, as in the original renderer.
        eye = (
            torch.linalg.inv(scene.world_transform)
            @ torch.cat([cam.center, cam.center.new_ones(1)])
        )[:3]
        direction = F.normalize(scene.render_means - eye, dim=-1)
        colors = (spherical_harmonics(3, direction, self.base_sh) + 0.5).clamp_min(0)
        if carbon:
            colors[self.roi] = (colors[self.roi] - self.offset[:, None]).clamp_min(0)
        colors[self.roi] *= 1.0 - 0.12 * self.surface_wet[:, None]
        if lighting and len(self.lights):
            ids = self.light_roi
            points = scene.means[ids].contiguous()
            normals = scene.normals[ids].contiguous()
            base = colors[ids].contiguous()
            result = torch.empty_like(base)
            wp.launch(
                illuminate,
                len(ids),
                [
                    wp.from_torch(points, dtype=wp.vec3),
                    wp.from_torch(normals, dtype=wp.vec3),
                    wp.from_torch(base, dtype=wp.vec3),
                    wp.from_torch(self.lights, dtype=wp.vec3),
                    wp.from_torch(self.radiance, dtype=wp.vec3),
                    wp.from_torch(result, dtype=wp.vec3),
                    wp.vec3(*cam.center.tolist()),
                    self.cfg.get("voxel_light_gain", 1.0),
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
            colors[ids] = result
        return colors

    @torch.no_grad()
    def render_scene(self, cam, lighting=True, carbon=True):
        scene = self.scene
        colors = self.surface_colors(cam, lighting, carbon)
        image, alpha, _ = rasterization(
            means=scene.render_means,
            quats=scene.quats,
            scales=scene.scales,
            opacities=scene.opacity,
            colors=colors,
            viewmats=(cam.view @ scene.world_transform)[None],
            Ks=cam.K[None],
            width=cam.width,
            height=cam.height,
            packed=False,
            near_plane=0.02,
            far_plane=scene.config.get("far_plane", 100.0),
            backgrounds=colors.new_tensor(
                [scene.config.get("background", [0.4, 0.45, 0.49])]
            ),
            render_mode="RGB+ED",
        )
        return image[0], alpha[0, ..., 0]
