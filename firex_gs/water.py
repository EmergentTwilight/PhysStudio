"""SPH spray parcels, SDF contact, and conservative grid heat/mass exchange."""
import math
import numpy as np
import warp as wp
from storm_gs.solver import sample, velocity
from .solver import Parameters, inside
from .chemistry import Species, heat_capacity, gas_density, water_exchange


@wp.kernel
def emit(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    contact: wp.array(dtype=wp.vec3i),
    start: int,
    origin: wp.vec3,
    direction: wp.vec3,
    speed: float,
    cone: float,
    parcel: float,
    seed: int,
):
    n = wp.tid()
    i = n + start
    state = wp.rand_init(seed, i)
    axis = wp.cross(direction, wp.vec3(0.0, 0.0, 1.0))
    axis = wp.normalize(axis)
    second = wp.cross(direction, axis)
    angle = wp.randf(state) * 6.2831853
    r = wp.sqrt(wp.randf(state))
    disk = (axis * wp.cos(angle) + second * wp.sin(angle)) * r
    x[i] = origin + disk * 0.012
    v[i] = (
        wp.normalize(direction + disk * cone) * speed * (0.95 + 0.1 * wp.randf(state))
    )
    mass[i] = parcel
    energy[i] = 0.0
    contact[i] = wp.vec3i(-1)


@wp.kernel
def density(
    x: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    grid: wp.uint64,
    h: float,
):
    i = wp.tid()
    if mass[i] <= 0.0:
        rho[i] = 1000.0
        return
    query = wp.hash_grid_query(grid, x[i], h)
    j = int(0)
    total = float(0.0)
    while wp.hash_grid_query_next(query, j):
        r = wp.length(x[i] - x[j])
        q = wp.max(h * h - r * r, 0.0)
        total += mass[j] * (315.0 / (64.0 * 3.14159265 * wp.pow(h, 9.0))) * q * q * q
    rho[i] = total


@wp.func
def drag_increment(
    gas: wp.vec3, liquid: wp.vec3, gas_mass: float, liquid_mass: float, dt: float
):
    ratio = liquid_mass / wp.max(gas_mass, 0.00000001)
    fraction = 1.0 - wp.exp(-dt / 0.8 * (1.0 + ratio))
    return (gas - liquid) * fraction / (1.0 + ratio)


@wp.kernel
def move(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    contact: wp.array(dtype=wp.vec3i),
    grid: wp.uint64,
    h: float,
    u: wp.array3d(dtype=float),
    vg: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    sdf: wp.array3d(dtype=float),
    wood: wp.array3d(dtype=int),
    loading: wp.array3d(dtype=float),
    impulse: wp.array3d(dtype=wp.vec3),
    ledger: wp.array(dtype=wp.float64),
    c: Parameters,
    dt: float,
):
    i = wp.tid()
    m = mass[i]
    if m <= 0.0:
        return
    p = x[i]
    vi = v[i]
    q = (p - c.origin) / c.dx
    gas = velocity(u, vg, w, q)
    # Parcel drag represents submillimeter spray droplets; the renderer uses
    # the parcel volume independently from the heat-transfer droplet radius.
    cell = wp.vec3i(q)
    load = float(0.0)
    if inside(cell[0], cell[1], cell[2], c):
        load = loading[cell[0], cell[1], cell[2]]
    drag = drag_increment(gas, vi, 1.2 * c.dx * c.dx * c.dx, load, dt)
    acceleration = wp.vec3(0.0, 0.0, -9.81)
    query = wp.hash_grid_query(grid, p, h)
    j = int(0)
    pi = 120.0 * wp.max(rho[i] - 1000.0, 0.0)
    while wp.hash_grid_query_next(query, j):
        d = p - x[j]
        r = wp.length(d)
        if j != i and r > 0.00001 and r < h and mass[j] > 0.0:
            pj = 120.0 * wp.max(rho[j] - 1000.0, 0.0)
            grad = -45.0 / (3.14159265 * wp.pow(h, 6.0)) * wp.pow(h - r, 2.0) * d / r
            acceleration -= (
                mass[j]
                * (
                    pi / wp.pow(wp.max(rho[i], 1000.0), 2.0)
                    + pj / wp.pow(wp.max(rho[j], 1000.0), 2.0)
                )
                * grad
            )
    acc_len = wp.length(acceleration)
    # CFL bound applies only to unresolved close-particle pressure impulses.
    if acc_len > h / (dt * dt) * 0.25:
        acceleration *= h / (dt * dt) * 0.25 / acc_len
    vn = vi + drag + acceleration * dt
    old = p
    hit = wp.vec3i(-1)
    for sub in range(3):
        candidate = p + vn * (dt / 3.0)
        qq = (candidate - c.origin) / c.dx - wp.vec3(0.5)
        dist = sample(sdf, qq)
        if dist < 0.006:
            grad = wp.vec3(
                sample(sdf, qq + wp.vec3(0.5, 0.0, 0.0))
                - sample(sdf, qq - wp.vec3(0.5, 0.0, 0.0)),
                sample(sdf, qq + wp.vec3(0.0, 0.5, 0.0))
                - sample(sdf, qq - wp.vec3(0.0, 0.5, 0.0)),
                sample(sdf, qq + wp.vec3(0.0, 0.0, 0.5))
                - sample(sdf, qq - wp.vec3(0.0, 0.0, 0.5)),
            )
            normal = grad / wp.max(wp.length(grad), 0.000001)
            candidate += normal * (0.006 - dist)
            normal_speed = wp.dot(vn, normal)
            vn -= normal * (1.08 * wp.min(normal_speed, 0.0))
            vn *= 0.78
            index = wp.vec3i((candidate - normal * c.dx * 0.8 - c.origin) / c.dx)
            if (
                inside(index[0], index[1], index[2], c)
                and wood[index[0], index[1], index[2]] == 1
            ):
                hit = index
        p = candidate
    index = wp.vec3i((old - c.origin) / c.dx)
    if inside(index[0], index[1], index[2], c):
        wp.atomic_add(impulse, index[0], index[1], index[2], -m * drag)
    outside = (
        p[0] < c.origin[0] + c.dx
        or p[1] < c.origin[1] + c.dx
        or p[2] < c.origin[2]
        or p[0] > c.origin[0] + float(c.nx - 1) * c.dx
        or p[1] > c.origin[1] + float(c.ny - 1) * c.dx
        or p[2] > c.origin[2] + float(c.nz - 1) * c.dx
    )
    if outside:
        wp.atomic_add(ledger, 0, wp.float64(m))
        wp.atomic_add(ledger, 1, wp.float64(energy[i]))
        mass[i] = 0.0
        energy[i] = 0.0
        x[i] = wp.vec3(1000.0 + float(i) * 0.02, 1000.0, 1000.0)
    else:
        x[i] = p
        v[i] = vn
        contact[i] = hit


@wp.kernel
def deposit(
    mass: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    contact: wp.array(dtype=wp.vec3i),
    film: wp.array3d(dtype=float),
    film_energy: wp.array3d(dtype=float),
    capacity: wp.array3d(dtype=float),
):
    i = wp.tid()
    q = contact[i]
    if mass[i] > 0.0 and q[0] >= 0:
        request = 0.65 * mass[i]
        previous = wp.atomic_add(film, q[0], q[1], q[2], request)
        retained = wp.clamp(capacity[q[0], q[1], q[2]] - previous, 0.0, request)
        wp.atomic_add(film, q[0], q[1], q[2], retained - request)
        heat = energy[i] * retained / mass[i]
        wp.atomic_add(film_energy, q[0], q[1], q[2], heat)
        mass[i] -= retained
        energy[i] -= heat


@wp.kernel
def wick(
    film: wp.array3d(dtype=float),
    energy: wp.array3d(dtype=float),
    fn: wp.array3d(dtype=float),
    en: wp.array3d(dtype=float),
    wood: wp.array3d(dtype=int),
    c: Parameters,
    dt: float,
):
    i, j, k = wp.tid()
    m = film[i, j, k]
    e = energy[i, j, k]
    if wood[i, j, k] == 1:
        a = wp.min(0.004 * dt / (c.dx * c.dx), 0.12)
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if inside(q[0], q[1], q[2], c) and wood[q[0], q[1], q[2]] == 1:
                    m += a * (film[q[0], q[1], q[2]] - film[i, j, k])
                    e += a * (energy[q[0], q[1], q[2]] - energy[i, j, k])
    fn[i, j, k] = m
    en[i, j, k] = e


@wp.kernel
def film_gather(
    film: wp.array3d(dtype=float),
    energy: wp.array3d(dtype=float),
    fraction: wp.array3d(dtype=float),
    specific: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    film[i, j, k] *= fraction[i, j, k]
    energy[i, j, k] = film[i, j, k] * specific[i, j, k]


@wp.kernel
def scatter(
    x: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    contact: wp.array(dtype=wp.vec3i),
    gm: wp.array3d(dtype=float),
    ge: wp.array3d(dtype=float),
    c: Parameters,
):
    i = wp.tid()
    if mass[i] <= 0.0:
        return
    index = wp.vec3i((x[i] - c.origin) / c.dx)

    if inside(index[0], index[1], index[2], c):
        wp.atomic_add(gm, index[0], index[1], index[2], mass[i])
        wp.atomic_add(ge, index[0], index[1], index[2], energy[i])


@wp.kernel
def exchange(
    gm: wp.array3d(dtype=float),
    ge: wp.array3d(dtype=float),
    fraction: wp.array3d(dtype=float),
    specific: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    Y: wp.array3d(dtype=Species),
    mist: wp.array3d(dtype=float),
    wm: wp.array3d(dtype=float),
    wt: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    wood: wp.array3d(dtype=int),
    vapor: wp.array3d(dtype=float),
    exits: wp.array3d(dtype=wp.vec3i),
    wet: wp.array3d(dtype=float),
    ledger: wp.array(dtype=wp.float64),
    radius: float,
    c: Parameters,
    dt: float,
):
    i, j, k = wp.tid()
    water = gm[i, j, k]
    if water <= 0.0:
        return
    y = Y[i, j, k]
    temp = T[i, j, k]
    cp = heat_capacity(y)
    mass = gas_density(y, temp) * c.dx * c.dx * c.dx
    iswood = wood[i, j, k] == 1
    if solid[i, j, k] == 1 and not iswood:
        specific[i, j, k] = ge[i, j, k] / water
        wet[i, j, k] += water * dt
        return
    if iswood:
        mass = wp.max(wm[i, j, k], 0.000001)
        cp = 1600.0
        temp = wt[i, j, k]
    area = 3.0 * water / (1000.0 * radius)
    conductance = 80.0 * area
    if iswood:
        conductance = 500.0 * area
    dm, remaining_energy, sensible, heat = water_exchange(
        mass, cp, temp, water, ge[i, j, k], conductance, dt
    )
    fraction[i, j, k] = wp.clamp(1.0 - dm / water, 0.0, 1.0)
    specific[i, j, k] = remaining_energy / wp.max(water - dm, 0.00000001)
    wp.atomic_add(ledger, 2, wp.float64(dm))
    wp.atomic_add(ledger, 3, wp.float64(heat))
    if iswood:
        wt[i, j, k] = 293.15 + (sensible - dm * 1860.0 * 80.0) / (mass * cp)
        wet[i, j, k] += water * dt
        count = int(0)
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if inside(q[0], q[1], q[2], c) and solid[q[0], q[1], q[2]] == 0:
                    count += 1
        if count > 0:
            for axis in range(3):
                for sign in range(2):
                    q = wp.vec3i(i, j, k)
                    q[axis] += 2 * sign - 1
                    if inside(q[0], q[1], q[2], c) and solid[q[0], q[1], q[2]] == 0:
                        wp.atomic_add(vapor, q[0], q[1], q[2], dm / float(count))
        else:
            q = exits[i, j, k]
            wp.atomic_add(vapor, q[0], q[1], q[2], dm)
    else:
        total = mass + dm
        y *= mass / total
        y[3] += dm / total
        T[i, j, k] = 293.15 + sensible / (total * heat_capacity(y))
        Y[i, j, k] = y
        mist[i, j, k] += dm / (c.dx * c.dx * c.dx)


@wp.kernel
def add_vapor(
    T: wp.array3d(dtype=float),
    Y: wp.array3d(dtype=Species),
    mist: wp.array3d(dtype=float),
    vapor: wp.array3d(dtype=float),
    c: Parameters,
):
    i, j, k = wp.tid()
    dm = vapor[i, j, k]
    if dm > 0.0:
        y = Y[i, j, k]
        temp = T[i, j, k]
        mass = gas_density(y, temp) * c.dx * c.dx * c.dx
        sensible = mass * heat_capacity(y) * (temp - 293.15) + dm * 1860.0 * 80.0
        y *= mass / (mass + dm)
        y[3] += dm / (mass + dm)
        Y[i, j, k] = y
        T[i, j, k] = 293.15 + sensible / ((mass + dm) * heat_capacity(y))
        mist[i, j, k] += dm / (c.dx * c.dx * c.dx)


@wp.kernel
def gather(
    x: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    contact: wp.array(dtype=wp.vec3i),
    fraction: wp.array3d(dtype=float),
    specific: wp.array3d(dtype=float),
    c: Parameters,
):
    i = wp.tid()
    if mass[i] <= 0.0:
        return
    index = wp.vec3i((x[i] - c.origin) / c.dx)

    if inside(index[0], index[1], index[2], c):
        mass[i] *= fraction[index[0], index[1], index[2]]
        energy[i] = mass[i] * specific[index[0], index[1], index[2]]


class Water:
    def __init__(self, sim, max_particles=50000):
        self.sim = sim
        self.cfg = sim.cfg
        self.n = max_particles
        self.count = 0
        self.emitted_mass = 0.0
        self.credit = 0.0
        self.x = wp.array(
            np.full((max_particles, 3), 1000, dtype="f"),
            dtype=wp.vec3,
            device=sim.device,
        )
        self.v = wp.zeros_like(self.x)
        self.mass = wp.zeros(max_particles, dtype=float, device=sim.device)
        self.energy = wp.zeros_like(self.mass)
        self.rho = wp.zeros_like(self.mass)
        self.contact = wp.zeros(max_particles, dtype=wp.vec3i, device=sim.device)
        self.grid = wp.HashGrid(64, 64, 64, device=sim.device)
        self.h = 0.018
        z = lambda: wp.zeros(sim.shape, dtype=float, device=sim.device)
        self.gm = z()
        self.ge = z()
        self.fraction = z()
        self.specific = z()
        self.vapor = z()
        self.wet = z()
        self.film = z()
        self.film_energy = z()
        self.fn = z()
        self.en = z()
        wood = sim.wood.numpy()
        exposed = np.zeros_like(wood)
        for axis in range(3):
            for shift in [-1, 1]:
                exposed += (1 - np.roll(sim.solid.numpy(), shift, axis)) * wood
        capacity = 0.45 * sim.wm.numpy() + 1000.0 * 0.0001 * sim.c.dx**2 * exposed
        self.capacity = wp.array(capacity.astype("f"), device=sim.device)
        self.ledger = wp.zeros(4, dtype=wp.float64, device=sim.device)
        sim.film = self.film
        sim.film_energy = self.film_energy

    def step(self, dt, time, enabled=True, aim="base"):
        s = self.sim
        c = self.cfg
        if enabled and c["water_start"] <= time < c["water_stop"]:
            self.credit += c["flow"] * dt / c["parcel_mass"]
            n = int(self.credit)
            self.credit -= n
            if n:
                if self.count + n > self.n:
                    raise RuntimeError(
                        "Increase particle capacity for this nozzle duration"
                    )
                target = np.array(c["target"], float)
                target[0] += c["sweep"] * math.sin((time - c["water_start"]) * 1.8)
                if aim == "top":
                    target[2] += 1.55
                origin = np.array(c["nozzle"], float)
                direction = target - origin
                direction /= np.linalg.norm(direction)
                wp.launch(
                    emit,
                    n,
                    [
                        self.x,
                        self.v,
                        self.mass,
                        self.energy,
                        self.contact,
                        self.count,
                        wp.vec3(*origin),
                        wp.vec3(*direction),
                        c["speed"],
                        c["cone"],
                        c["parcel_mass"],
                        1367,
                    ],
                    device=s.device,
                )
                self.count += n
                self.emitted_mass += n * c["parcel_mass"]
        if self.count == 0:
            return
        self.gm.zero_()
        self.ge.zero_()
        wp.launch(
            scatter,
            self.count,
            [self.x, self.mass, self.energy, self.contact, self.gm, self.ge, s.c],
            device=s.device,
        )
        self.grid.build(self.x, self.h)
        wp.launch(
            density,
            self.count,
            [self.x, self.mass, self.rho, self.grid.id, self.h],
            device=s.device,
        )
        wp.launch(
            move,
            self.count,
            [
                self.x,
                self.v,
                self.mass,
                self.energy,
                self.rho,
                self.contact,
                self.grid.id,
                self.h,
                s.u,
                s.v,
                s.w,
                s.sdf,
                s.wood,
                self.gm,
                s.impulse,
                self.ledger,
                s.c,
                dt,
            ],
            device=s.device,
        )
        wp.launch(
            deposit,
            self.count,
            [
                self.mass,
                self.energy,
                self.contact,
                self.film,
                self.film_energy,
                self.capacity,
            ],
            device=s.device,
        )
        wp.launch(
            wick,
            s.shape,
            [self.film, self.film_energy, self.fn, self.en, s.wood, s.c, dt],
            device=s.device,
        )
        self.film, self.fn = self.fn, self.film
        self.film_energy, self.en = self.en, self.film_energy
        s.film = self.film
        s.film_energy = self.film_energy
        wp.copy(self.gm, self.film)
        wp.copy(self.ge, self.film_energy)
        self.fraction.fill_(1.0)
        self.specific.zero_()
        self.vapor.zero_()
        wp.launch(
            scatter,
            self.count,
            [self.x, self.mass, self.energy, self.contact, self.gm, self.ge, s.c],
            device=s.device,
        )
        wp.launch(
            exchange,
            s.shape,
            [
                self.gm,
                self.ge,
                self.fraction,
                self.specific,
                s.T,
                s.Y,
                s.mist,
                s.wm,
                s.wt,
                s.solid,
                s.wood,
                self.vapor,
                s.exits,
                self.wet,
                self.ledger,
                c["droplet_radius"],
                s.c,
                dt,
            ],
            device=s.device,
        )
        wp.launch(
            add_vapor, s.shape, [s.T, s.Y, s.mist, self.vapor, s.c], device=s.device
        )
        wp.launch(
            gather,
            self.count,
            [
                self.x,
                self.mass,
                self.energy,
                self.contact,
                self.fraction,
                self.specific,
                s.c,
            ],
            device=s.device,
        )
        wp.launch(
            film_gather,
            s.shape,
            [self.film, self.film_energy, self.fraction, self.specific],
            device=s.device,
        )

    def diagnostics(self):
        mass = float(self.mass.numpy().sum())
        film = float(self.film.numpy().sum())
        led = self.ledger.numpy()
        return dict(
            parcels=self.count,
            water_emitted_kg=self.emitted_mass,
            water_remaining_kg=mass,
            water_retained_kg=film,
            water_escaped_kg=float(led[0]),
            water_evaporated_kg=float(led[2]),
            water_mass_balance_error_kg=float(
                mass + film + led[0] + led[2] - self.emitted_mass
            ),
            water_heat_absorbed_j=float(led[3]),
        )
