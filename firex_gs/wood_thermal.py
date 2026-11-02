"""Localized ignition and conservative solid heat diffusion for reactive wood."""
import math
import warp as wp


@wp.kernel
def cold_wood(
    wood: wp.array3d(dtype=int),
    temperature: wp.array3d(dtype=float),
    gas: wp.array3d(dtype=float),
    cold: float,
):
    i, j, k = wp.tid()
    gas[i, j, k] = 293.15
    if wood[i, j, k] > 0:
        temperature[i, j, k] = cold


@wp.kernel
def ignite(
    wood: wp.array3d(dtype=int),
    temperature: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    center: wp.vec3,
    radius: float,
    hot: float,
):
    i, j, k = wp.tid()
    point = origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx
    if wood[i, j, k] > 0 and wp.length(point - center) <= radius:
        temperature[i, j, k] = wp.max(temperature[i, j, k], hot)


@wp.kernel
def seed_air(
    wood: wp.array3d(dtype=int),
    solid: wp.array3d(dtype=int),
    temperature: wp.array3d(dtype=float),
    gas: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    if solid[i, j, k] == 0:
        hottest = float(0.0)
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if (
                    q[0] >= 0
                    and q[1] >= 0
                    and q[2] >= 0
                    and q[0] < wood.shape[0]
                    and q[1] < wood.shape[1]
                    and q[2] < wood.shape[2]
                ):
                    if wood[q[0], q[1], q[2]] > 0:
                        hottest = wp.max(hottest, temperature[q[0], q[1], q[2]])
        if hottest > 450.0:
            gas[i, j, k] = wp.min(hottest, 750.0)


@wp.kernel
def diffuse(
    wood: wp.array3d(dtype=int),
    mass: wp.array3d(dtype=float),
    old: wp.array3d(dtype=float),
    new: wp.array3d(dtype=float),
    ratio: float,
):
    i, j, k = wp.tid()
    m = mass[i, j, k]
    t = old[i, j, k]
    increment = float(0.0)
    if wood[i, j, k] > 0 and m > 1e-10:
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if (
                    q[0] >= 0
                    and q[1] >= 0
                    and q[2] >= 0
                    and q[0] < wood.shape[0]
                    and q[1] < wood.shape[1]
                    and q[2] < wood.shape[2]
                ):
                    if wood[q[0], q[1], q[2]] > 0:
                        neighbor_mass = mass[q[0], q[1], q[2]]
                        increment += (
                            wp.min(m, neighbor_mass) / m * (old[q[0], q[1], q[2]] - t)
                        )
    new[i, j, k] = t + ratio * increment


class WoodThermal:
    def __init__(self, simulation):
        self.sim = simulation
        cfg = simulation.cfg
        self.alpha = cfg.get("wood_conductivity", 0.0) / (cfg["wood_density"] * 1600.0)
        if self.alpha < 0:
            raise ValueError("Wood conductivity must be nonnegative")
        self.buffer = wp.zeros_like(simulation.wt) if self.alpha > 0 else None
        regions = cfg.get("ignition_regions")
        if regions is not None:
            wp.launch(
                cold_wood,
                simulation.shape,
                [
                    simulation.wood,
                    simulation.wt,
                    simulation.T,
                    cfg.get("wood_cold_temperature", 310.0),
                ],
                device=simulation.device,
            )
            for region in regions:
                if region["radius"] <= 0:
                    raise ValueError("Ignition radius must be positive")
                wp.launch(
                    ignite,
                    simulation.shape,
                    [
                        simulation.wood,
                        simulation.wt,
                        simulation.c.origin,
                        simulation.c.dx,
                        wp.vec3(*region["center"]),
                        float(region["radius"]),
                        float(
                            region.get("temperature", cfg["wood_initial_temperature"])
                        ),
                    ],
                    device=simulation.device,
                )
            wp.launch(
                seed_air,
                simulation.shape,
                [simulation.wood, simulation.solid, simulation.wt, simulation.T],
                device=simulation.device,
            )

    def step(self, dt):
        if not self.alpha:
            return
        sim = self.sim
        steps = max(1, math.ceil(6 * self.alpha * dt / (sim.c.dx**2 * 0.8)))
        ratio = self.alpha * dt / (sim.c.dx**2 * steps)
        for _ in range(steps):
            wp.launch(
                diffuse,
                sim.shape,
                [sim.wood, sim.wm, sim.wt, self.buffer, ratio],
                device=sim.device,
            )
            sim.wt, self.buffer = self.buffer, sim.wt
