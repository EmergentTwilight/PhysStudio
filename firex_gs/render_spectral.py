"""Spectral flame rendering, using the FieryGS Planck/CIE color formulation.

Reference: FieryGS simulation/color_mapping.py; see references/sources.json.
FieryGS authors, Apache-2.0. Gas/liquid states remain frozen during rendering.
"""
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from gsplat import rasterization
from scipy.ndimage import gaussian_filter
from storm_gs.solver import sample4, sample
from .solver import Parameters


def blackbody_table(device):
    """Planck spectrum integrated against FieryGS's released CIE color matching data."""
    cie = np.loadtxt(Path(__file__).parent / "references/cie-cmf.txt")
    wavelength = cie[:, 0][None, :] * 1e-9
    temp = np.linspace(300.0, 3000.0, 1024)[:, None]
    spectra = (2 * 6.62607015e-34 * 299792458.0**2) / (
        wavelength**5
        * np.expm1(6.62607015e-34 * 299792458.0 / (wavelength * 1.380649e-23 * temp))
    )
    xyz = spectra @ cie[:, 1:4]
    matrix = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.057],
        ]
    )
    rgb = np.maximum(xyz @ matrix.T, 0.0)
    reference = np.interp(1750.0, temp[:, 0], rgb[:, 0])
    rgb /= reference
    return wp.array(rgb.astype("f"), dtype=wp.vec3, device=device)


@wp.func
def blackbody(table: wp.array(dtype=wp.vec3), temperature: float):
    index = wp.clamp((temperature - 300.0) * (1023.0 / 2700.0), 0.0, 1022.99)
    lo = int(index)
    return wp.lerp(table[lo], table[lo + 1], index - float(lo))


@wp.struct
class Look:
    emission: float
    thermal_detail: float
    smoke_scale: float
    absorption: float
    macro_scale: float


@wp.func
def noise(p: wp.vec3):
    return (
        wp.noise(wp.uint32(153), p) * 0.65
        + wp.noise(wp.uint32(735), p * 2.01) * 0.25
        + wp.noise(wp.uint32(121), p * 4.03) * 0.10
    )


@wp.kernel
def detail_temperature(
    field: wp.array3d(dtype=wp.vec4),
    thermal: wp.array3d(dtype=float),
    c: Parameters,
    look: Look,
    t: float,
):
    i, j, k = wp.tid()
    p = c.origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * (
        c.dx * 0.5
    )
    temperature = sample4(field, (p - c.origin) / c.dx - wp.vec3(0.5))[0]
    result = temperature
    if temperature > 700.0:
        phase = wp.cw_mul(
            p, wp.vec3(look.macro_scale, look.macro_scale, look.macro_scale * 0.52)
        ) - wp.vec3(t * 0.35, t * 0.12, t * 10.5)
        coarse = phase * 0.38
        bend = wp.vec3(
            wp.noise(wp.uint32(219), coarse),
            wp.noise(wp.uint32(241), coarse),
            wp.noise(wp.uint32(271), coarse),
        )
        phase += bend * 2.2
        broad = wp.noise(wp.uint32(827), phase)
        fine = wp.noise(wp.uint32(891), phase * 2.3 + wp.vec3(2.1, 7.2, -t * 0.8))
        n = broad * 0.76 + fine * 0.24
        result = wp.clamp(
            temperature
            + look.thermal_detail * n * wp.smoothstep(700.0, 1250.0, temperature),
            293.15,
            2600.0,
        )
    thermal[i, j, k] = result


@wp.kernel
def shade(
    field: wp.array3d(dtype=wp.vec4), lit: wp.array3d(dtype=float), c: Parameters
):
    i, j, k = wp.tid()
    q = wp.vec3(float(i), float(j), float(k))
    tau = float(0.0)
    sun = wp.normalize(wp.vec3(-0.45, 0.3, 1.0))
    for n in range(20):
        at = q + sun * (float(n) + 0.5) * 1.8
        if (
            at[0] >= 0.0
            and at[1] >= 0.0
            and at[2] >= 0.0
            and at[0] < float(c.nx - 1)
            and at[1] < float(c.ny - 1)
            and at[2] < float(c.nz - 1)
        ):
            v = sample4(field, at)
            tau += (v[1] + v[2]) * c.dx * 1.8
    lit[i, j, k] = wp.exp(-tau * 0.7)


@wp.kernel
def composite(
    rgb: wp.array2d(dtype=wp.vec3),
    depth: wp.array2d(dtype=float),
    alpha: wp.array2d(dtype=float),
    rays: wp.array2d(dtype=wp.vec3),
    field: wp.array3d(dtype=wp.vec4),
    lit: wp.array3d(dtype=float),
    char: wp.array3d(dtype=float),
    wet: wp.array3d(dtype=float),
    prgb: wp.array2d(dtype=wp.vec3),
    pa: wp.array2d(dtype=float),
    pd: wp.array2d(dtype=float),
    out: wp.array2d(dtype=wp.vec3),
    emitted: wp.array2d(dtype=wp.vec3),
    thermal_grid: wp.array3d(dtype=float),
    c: Parameters,
    eye: wp.vec3,
    t: float,
    exposure: float,
    flame_gain: float,
    power: float,
    table: wp.array(dtype=wp.vec3),
    look: Look,
    legacy_surface: int,
):
    y, x = wp.tid()
    ray = rays[y, x]
    surface = depth[y, x]
    limit = surface
    if alpha[y, x] < 0.05:
        limit = 1000.0
    bg = rgb[y, x]
    bg = (
        wp.vec3(
            wp.pow(wp.max(bg[0], 0.0), 2.2),
            wp.pow(wp.max(bg[1], 0.0), 2.2),
            wp.pow(wp.max(bg[2], 0.0), 2.2),
        )
        * exposure
    )
    point = eye + ray * surface
    q = (point - c.origin) / c.dx - wp.vec3(0.5)
    if (
        legacy_surface == 1
        and q[0] >= 0.0
        and q[1] >= 0.0
        and q[2] >= 0.0
        and q[0] < float(c.nx - 1)
        and q[1] < float(c.ny - 1)
        and q[2] < float(c.nz - 1)
    ):
        burn = sample(char, q)
        water = sample(wet, q)
        bg *= 1.0 - wp.clamp(burn * 0.6, 0.0, 0.65)
        bg *= 1.0 - wp.clamp(water * 0.22, 0.0, 0.22)
    distance_to_fire = wp.length(point - wp.vec3(-0.25, -0.4, -1.7))
    light = (
        power
        / (distance_to_fire * distance_to_fire + 0.4)
        * 0.25
        * float(legacy_surface)
    )
    bg += wp.cw_mul(bg + wp.vec3(0.03), wp.vec3(1.0, 0.24, 0.025)) * light
    upper = c.origin + wp.vec3(float(c.nx), float(c.ny), float(c.nz)) * c.dx
    near = float(0.0)
    far = wp.min(limit, 1000.0)
    for axis in range(3):
        direction = ray[axis]
        if wp.abs(direction) < 0.000001:
            direction = 0.000001
        a = (c.origin[axis] - eye[axis]) / direction
        b = (upper[axis] - eye[axis]) / direction
        near = wp.max(near, wp.min(a, b))
        far = wp.min(far, wp.max(a, b))
    trans = float(1.0)
    color = wp.vec3(0.0)
    emission = wp.vec3(0.0)
    particle_done = bool(False)
    step = c.dx * 0.48
    state = wp.rand_init(728, y * rgb.shape[1] + x)
    distance = near + step * wp.randf(state)
    iteration = int(0)
    while distance < far and trans > 0.008 and iteration < 400:
        if (
            not particle_done
            and pd[y, x] < distance
            and pd[y, x] < limit
            and pa[y, x] > 0.0
        ):
            color += trans * prgb[y, x]
            trans *= 1.0 - pa[y, x]
            particle_done = True
        p = eye + ray * distance
        at = (p - c.origin) / c.dx - wp.vec3(0.5)
        v = sample4(field, at)
        temperature = v[0]
        smoke = v[1]
        steam = v[2]
        burn = v[3]
        # Coherent temperature detail sharpens subgrid tongues before the
        # nonlinear spectral transform. Thermal support comes from saved gas.
        thermal = sample(thermal_grid, (p - c.origin) / (c.dx * 0.5) - wp.vec3(0.5))
        detail = wp.clamp(
            1.0 + (thermal - temperature) / wp.max(look.thermal_detail, 1.0) * 0.75,
            0.35,
            1.6,
        )
        hot = wp.smoothstep(850.0, 1450.0, thermal)
        soot = wp.max(smoke, 0.0)
        fuel_emission = wp.min(soot, 4.0) * 0.25 + wp.sqrt(wp.max(burn, 0.0)) * 0.8
        smoke *= look.smoke_scale * (1.0 - 0.85 * hot)
        sigma = (smoke + steam) * detail + look.absorption * hot * detail
        if sigma > 0.00001 or thermal > 850.0:
            lighting = sample(lit, at)
            albedo = (
                smoke * wp.vec3(0.13, 0.12, 0.11) + steam * wp.vec3(0.77, 0.81, 0.84)
            ) / wp.max(smoke + steam, 0.00001)
            scattering = albedo * (0.28 + 0.72 * lighting)
            fire = blackbody(table, thermal) * look.emission * fuel_emission * detail
            ds = wp.min(step, far - distance)
            a = 1.0 - wp.exp(-sigma * ds)
            integrated = ds
            if sigma > 0.00001:
                integrated = a / sigma
            emission += trans * fire * integrated
            color += trans * (a * scattering + fire * integrated)
            trans *= 1.0 - a
        distance += step
        iteration += 1
    if not particle_done and pd[y, x] < limit and pa[y, x] > 0.0:
        color += trans * prgb[y, x]
        trans *= 1.0 - pa[y, x]
    color += trans * bg
    out[y, x] = color
    emitted[y, x] = emission


def hardware(cfg, device, time=0.0):
    origin = np.array(cfg["nozzle"])
    target = np.array(cfg["target"], float)
    target[0] += cfg["sweep"] * np.sin(max(time - cfg["water_start"], 0.0) * 1.8)
    if cfg.get("control") == "top":
        target[2] += 1.55
    axis = target - origin
    axis /= np.linalg.norm(axis)
    right = np.cross(axis, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(axis, right)
    positions = []
    colors = []
    sizes = []
    for length in np.linspace(-0.45, 0.04, 75):
        nozzle = length > -0.1
        radius = 0.022 if nozzle else 0.027
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            p = (
                origin
                + axis * length
                + radius * (np.cos(angle) * right + np.sin(angle) * up)
            )
            positions.append(p)
            sizes.append(0.006)
            shade = 0.65 + 0.35 * max(
                np.dot(np.cos(angle) * right + np.sin(angle) * up, [-0.4, 0.2, 0.7]),
                0.0,
            )
            colors.append(
                np.array([0.36, 0.4, 0.43] if nozzle else [0.025, 0.033, 0.038]) * shade
            )
    anchor = origin - axis * 0.45
    for length in np.linspace(0.0, 3.2, 200):
        center = anchor - axis * length + np.array([0.0, 0.0, -0.22 * length * length])
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            normal = np.cos(angle) * right + np.sin(angle) * up
            positions.append(center + 0.024 * normal)
            sizes.append(0.009)
            shade = 0.6 + 0.4 * max(np.dot(normal, [-0.4, 0.2, 0.7]), 0.0)
            colors.append(np.array([0.026, 0.032, 0.038]) * shade)
    return [
        torch.tensor(np.array(v), device=device, dtype=torch.float32)
        for v in [positions, colors, sizes]
    ]


class Renderer:
    def __init__(self, scene, cfg, geometry):
        self.scene, self.cfg, self.geometry = scene, cfg, geometry
        self.device = scene.device
        self.c = Parameters()
        self.c.origin = wp.vec3(*cfg["origin"])
        self.c.dx = cfg["dx"]
        self.c.nx, self.c.ny, self.c.nz = cfg["shape"]
        self.shape = tuple(cfg["shape"])
        self.field = wp.zeros(self.shape, dtype=wp.vec4, device=self.device)
        self.lit = wp.zeros(self.shape, dtype=float, device=self.device)
        self.hardware = hardware(cfg, self.device)
        self.table = blackbody_table(self.device)
        self.look = Look()
        self.thermal = wp.zeros(
            tuple(n * 2 for n in self.shape), dtype=float, device=self.device
        )
        self.look.emission = cfg.get("spectral_emission", 32.0)
        self.look.thermal_detail = cfg.get("thermal_detail", 320.0)
        self.look.smoke_scale = cfg.get("spectral_smoke_scale", 0.35)
        self.look.absorption = cfg.get("flame_absorption", 0.2)
        self.look.macro_scale = cfg.get("flame_detail_scale", 10.0)

    def update(self, state):
        self.state = state
        self.hardware = hardware(
            self.cfg, self.device, float(state["time"]) - self.cfg["warmup"]
        )
        t = state["T"]
        y = state["Y"]
        burn = state["burn"]
        solid = self.geometry["solid"]
        # Carbonaceous residual scatters dark smoke. The white suppression
        # plume is a visible-condensate proxy from the transported vapor source.
        rho = 101325 / (
            8.3144626
            * np.maximum(t, 293.15)
            * np.maximum(
                (y / np.array([0.016, 0.032, 0.044, 0.018, 0.028, 0.012])).sum(-1), 1.0
            )
        )
        smoke = y[..., 5] * rho * 220 * self.cfg["smoke_extinction"]
        steam = state["mist"] * self.cfg["steam_extinction"] * 10
        field = np.stack([t, smoke, steam, burn], -1).astype("f")
        field[solid > 0] = 0
        self.field.assign(field)
        wp.launch(
            detail_temperature,
            self.thermal.shape,
            [self.field, self.thermal, self.c, self.look, float(state["time"])],
            device=self.device,
        )
        initial = self.cfg["wood_density"] * self.cfg["dx"] ** 3
        ch = gaussian_filter(state["char"] / max(initial, 1e-8), 0.65).astype("f")
        wet = gaussian_filter(state["wet"] * 100, 0.8).astype("f")
        self.char = wp.array(ch, device=self.device)
        self.wet = wp.array(wet, device=self.device)
        self.power = float(
            state.get("hrr", burn * 52125000).sum() * self.cfg["dx"] ** 3 / 1000 / 200
        )
        wp.launch(shade, self.shape, [self.field, self.lit, self.c], device=self.device)

    @torch.no_grad()
    def particles(self, cam, surface):
        s = self.state
        device = self.device
        active = s["mass"] > 1e-9
        x = torch.tensor(s["x"][active], device=device)
        v = torch.tensor(s["v"][active], device=device)
        mass = torch.tensor(s["mass"][active], device=device)
        r = (3 * mass / (4 * np.pi * 1000)).pow(1 / 3)
        identity = torch.eye(3, device=device)[None]
        cov = (
            r[:, None, None].square() * identity
            + v[:, :, None] * v[:, None, :] * 0.014**2 / 12
        )
        highlight = (
            0.76 + 0.22 * torch.sin(x[:, 0] * 61 + x[:, 1] * 47 + x[:, 2] * 19).square()
        )
        color = (
            highlight[:, None] * torch.tensor([0.75, 0.86, 0.96], device=device)[None]
        )
        opacity = (mass / self.cfg["parcel_mass"]).pow(0.33) * 0.26
        hx, hc, hr = self.hardware
        x = torch.cat([x, hx])
        cov = torch.cat([cov, hr[:, None, None].square() * identity])
        color = torch.cat([color, hc])
        opacity = torch.cat([opacity, torch.full_like(hr, 0.94)])
        image, alpha, _ = rasterization(
            means=x,
            quats=None,
            scales=None,
            covars=cov,
            opacities=opacity,
            colors=color,
            viewmats=cam.view[None],
            Ks=cam.K[None],
            width=cam.width,
            height=cam.height,
            near_plane=0.02,
            far_plane=100.0,
            render_mode="RGB+ED",
            packed=False,
            rasterize_mode="antialiased",
        )
        image = image[0]
        alpha = alpha[0, ..., 0]
        visible = (image[..., 3] < surface) & (image[..., 3] > 0.01)
        return (
            (image[..., :3] * visible[..., None]).contiguous(),
            (alpha * visible).contiguous(),
            image[..., 3].contiguous(),
        )

    @torch.no_grad()
    def render_scene(self, cam):
        return self.scene.render(cam)

    @torch.no_grad()
    def render(self, cam):
        rgbd, alpha = self.render_scene(cam)
        rgb = rgbd[..., :3].clamp(0, 1).contiguous()
        raw = cam.world_points(torch.ones_like(alpha)) - cam.center
        scale = torch.linalg.vector_norm(raw, dim=-1)
        rays = (raw / scale[..., None]).contiguous()
        depth = (rgbd[..., 3] * scale).contiguous()
        pr, pa, pd = self.particles(
            cam, torch.where(alpha > 0.05, rgbd[..., 3], torch.full_like(alpha, 10000))
        )
        pd = (pd * scale).contiguous()
        out = torch.empty_like(rgb)
        emitted = torch.empty_like(rgb)
        wp.synchronize_device(self.device)
        wp.launch(
            composite,
            (cam.height, cam.width),
            [
                wp.from_torch(rgb, dtype=wp.vec3),
                wp.from_torch(depth),
                wp.from_torch(alpha.contiguous()),
                wp.from_torch(rays, dtype=wp.vec3),
                self.field,
                self.lit,
                self.char,
                self.wet,
                wp.from_torch(pr, dtype=wp.vec3),
                wp.from_torch(pa),
                wp.from_torch(pd),
                wp.from_torch(out, dtype=wp.vec3),
                wp.from_torch(emitted, dtype=wp.vec3),
                self.thermal,
                self.c,
                wp.vec3(*cam.center.tolist()),
                float(self.state["time"]),
                self.cfg["exposure"],
                self.cfg["flame_gain"],
                self.power,
                self.table,
                self.look,
                int(self.cfg.get("legacy_surface", True)),
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        # Keep the accepted background/water transfer curve. The spectral
        # highlight response is applied only where fire contributes radiance.
        highlights = (emitted - 0.85).clamp_min(0).permute(2, 0, 1)[None]
        bloom = F.avg_pool2d(F.avg_pool2d(highlights, 5, 1, 2), 9, 1, 4)[0].permute(
            1, 2, 0
        )
        hdr = out + 0.035 * bloom
        luminance = (
            hdr * torch.tensor([0.2126, 0.7152, 0.0722], device=self.device)
        ).sum(-1, keepdim=True)
        neutral = (1 - torch.exp(-(luminance - 0.8).clamp_min(0) * 0.08)) * 0.35
        optical = hdr * (1 - neutral) + luminance * neutral
        filmic = (optical * (2.51 * optical + 0.03)) / (
            optical * (2.43 * optical + 0.59) + 0.14
        )
        base = out / (1 + out * 0.35)
        fire_luma = (
            emitted * torch.tensor([0.2126, 0.7152, 0.0722], device=self.device)
        ).sum(-1, keepdim=True)
        blend = ((fire_luma - 0.01) / 0.25).clamp(0, 1)
        blend = blend * blend * (3 - 2 * blend)
        mapped = base * (1 - blend) + filmic * blend
        return mapped.clamp(0, 1).pow(1 / 2.2), rgb
