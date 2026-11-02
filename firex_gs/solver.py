"""Open-boundary MAC flow, reactive gas species, and finite-capacity solid fuel."""
from dataclasses import dataclass
import numpy as np
import warp as wp
from storm_gs.solver import sample, sample3, velocity
from .chemistry import Species, ambient, heat_capacity, gas_density, reaction


@wp.struct
class Parameters:
    origin: wp.vec3
    dx: float
    nx: int
    ny: int
    nz: int
    wind: wp.vec3
    pyrolysis: float
    transfer: float
    feedback: float


@wp.func
def inside(i: int, j: int, k: int, c: Parameters):
    return i >= 0 and j >= 0 and k >= 0 and i < c.nx and j < c.ny and k < c.nz


@wp.func
def open_cell(s: wp.array3d(dtype=int), i: int, j: int, k: int, c: Parameters):
    result = int(1)
    if inside(i, j, k, c):
        result = 1 - s[i, j, k]
    return result


@wp.func
def value(p: wp.array3d(dtype=float), i: int, j: int, k: int, c: Parameters):
    x = float(0.0)
    if inside(i, j, k, c):
        x = p[i, j, k]
    return x


@wp.func
def sample_species(a: wp.array3d(dtype=Species), p: wp.vec3):
    q = wp.vec3(
        wp.clamp(p[0], 0.0, float(a.shape[0] - 1) - 0.001),
        wp.clamp(p[1], 0.0, float(a.shape[1] - 1) - 0.001),
        wp.clamp(p[2], 0.0, float(a.shape[2] - 1) - 0.001),
    )
    base = wp.vec3i(q)
    f = q - wp.vec3(base)
    out = Species(0.0)
    for n in range(8):
        offset = wp.vec3i(n % 2, (n / 2) % 2, n / 4)
        weight = wp.vec3(1.0) - f
        for d in range(3):
            if offset[d] == 1:
                weight[d] = f[d]
        out += a[base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]] * (
            weight[0] * weight[1] * weight[2]
        )
    return out


@wp.kernel
def initialize(
    T: wp.array3d(dtype=float),
    Y: wp.array3d(dtype=Species),
    wood: wp.array3d(dtype=int),
    solid: wp.array3d(dtype=int),
    wm: wp.array3d(dtype=float),
    wt: wp.array3d(dtype=float),
    density: float,
    temperature: float,
    c: Parameters,
):
    i, j, k = wp.tid()
    Y[i, j, k] = ambient()
    T[i, j, k] = 293.15
    if wood[i, j, k] == 1:
        wm[i, j, k] = density * c.dx * c.dx * c.dx
        wt[i, j, k] = 310.0
        exposed = int(0)
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if inside(q[0], q[1], q[2], c) and solid[q[0], q[1], q[2]] == 0:
                    exposed += 1
        if exposed > 0:
            wt[i, j, k] = temperature
    else:
        wt[i, j, k] = 293.15
    if solid[i, j, k] == 0:
        near = int(0)
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if inside(q[0], q[1], q[2], c):
                    near += wood[q[0], q[1], q[2]]
        if near > 0:
            T[i, j, k] = 750.0


@wp.kernel
def curl_field(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    omega: wp.array3d(dtype=wp.vec3),
    mag: wp.array3d(dtype=float),
    dx: float,
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    vx = (
        velocity(u, v, w, p + wp.vec3(1.0, 0.0, 0.0))
        - velocity(u, v, w, p - wp.vec3(1.0, 0.0, 0.0))
    ) / (2.0 * dx)
    vy = (
        velocity(u, v, w, p + wp.vec3(0.0, 1.0, 0.0))
        - velocity(u, v, w, p - wp.vec3(0.0, 1.0, 0.0))
    ) / (2.0 * dx)
    vz = (
        velocity(u, v, w, p + wp.vec3(0.0, 0.0, 1.0))
        - velocity(u, v, w, p - wp.vec3(0.0, 0.0, 1.0))
    ) / (2.0 * dx)
    curl = wp.vec3(vy[2] - vz[1], vz[0] - vx[2], vx[1] - vy[0])
    omega[i, j, k] = curl
    mag[i, j, k] = wp.length(curl)


@wp.kernel
def flow(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    un: wp.array3d(dtype=float),
    vn: wp.array3d(dtype=float),
    wn: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    omega: wp.array3d(dtype=wp.vec3),
    mag: wp.array3d(dtype=float),
    impulse: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=int),
    c: Parameters,
    dt: float,
    time: float,
):
    i, j, k = wp.tid()
    base = wp.vec3(float(i), float(j), float(k))
    for axis in range(3):
        offset = wp.vec3(0.5)
        offset[axis] = 0.0
        p = base + offset
        current = velocity(u, v, w, p)
        mid = p - current * (0.5 * dt / c.dx)
        back = p - velocity(u, v, w, mid) * (dt / c.dx)
        prev = velocity(u, v, w, back)
        q = p - wp.vec3(0.5)
        temp = sample(T, q)
        grad = wp.vec3(
            sample(mag, q + wp.vec3(1.0, 0.0, 0.0))
            - sample(mag, q - wp.vec3(1.0, 0.0, 0.0)),
            sample(mag, q + wp.vec3(0.0, 1.0, 0.0))
            - sample(mag, q - wp.vec3(0.0, 1.0, 0.0)),
            sample(mag, q + wp.vec3(0.0, 0.0, 1.0))
            - sample(mag, q - wp.vec3(0.0, 0.0, 1.0)),
        )
        confinement = wp.cross(
            grad / wp.max(wp.length(grad), 0.0001), sample3(omega, q)
        ) * (0.65 * c.dx)
        world = c.origin + p * c.dx
        turbulence = wp.vec3(
            wp.noise(wp.uint32(419), world * 3.0 + wp.vec3(time * 0.7, 0.0, 0.0)),
            wp.noise(wp.uint32(752), world * 3.0 + wp.vec3(0.0, time * 0.9, 0.0)),
            wp.noise(wp.uint32(291), world * 3.0 + wp.vec3(0.0, 0.0, time * 0.5)),
        )
        force = (
            wp.vec3(0.0, 0.0, 9.81 * (1.0 - 293.15 / wp.max(temp, 293.15)))
            + confinement
            + (c.wind - current) * 0.15
            + turbulence * 1.8 * wp.smoothstep(350.0, 700.0, temp)
        )
        lap = (
            velocity(u, v, w, back + wp.vec3(1.0, 0.0, 0.0))
            + velocity(u, v, w, back - wp.vec3(1.0, 0.0, 0.0))
            + velocity(u, v, w, back + wp.vec3(0.0, 1.0, 0.0))
            + velocity(u, v, w, back - wp.vec3(0.0, 1.0, 0.0))
            + velocity(u, v, w, back + wp.vec3(0.0, 0.0, 1.0))
            + velocity(u, v, w, back - wp.vec3(0.0, 0.0, 1.0))
            - prev * 6.0
        )
        speed = (
            prev[axis]
            + dt * (force[axis] + 0.02 * lap[axis] / (c.dx * c.dx))
            + sample3(impulse, q)[axis] / (1.2 * c.dx * c.dx * c.dx)
        )
        # Open atmospheric boundaries retain outward flux and ambient inflow.
        a = wp.vec3i(i, j, k)
        b = a
        b[axis] -= 1
        if (
            open_cell(solid, a[0], a[1], a[2], c)
            * open_cell(solid, b[0], b[1], b[2], c)
            == 0
        ):
            speed = 0.0
        if axis == 0:
            un[i, j, k] = speed
        elif axis == 1:
            vn[i, j, k] = speed
        else:
            wn[i, j, k] = speed


@wp.kernel
def divergence(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
):
    i, j, k = wp.tid()
    out[i, j, k] = (
        (
            u[i + 1, j, k]
            - u[i, j, k]
            + v[i, j + 1, k]
            - v[i, j, k]
            + w[i, j, k + 1]
            - w[i, j, k]
        )
        / c.dx
        * float(1 - solid[i, j, k])
    )


@wp.kernel
def pressure(
    p: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
    color: int,
):
    i, j, k = wp.tid()
    if (i + j + k) % 2 == color and solid[i, j, k] == 0:
        total = float(0.0)
        count = int(0)
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if open_cell(solid, q[0], q[1], q[2], c) == 1:
                    total += value(p, q[0], q[1], q[2], c)
                    count += 1
        if count > 0:
            p[i, j, k] += 1.65 * (
                (total - div[i, j, k] * c.dx * c.dx) / float(count) - p[i, j, k]
            )


@wp.kernel
def project(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
):
    i, j, k = wp.tid()
    for axis in range(3):
        a = wp.vec3i(i, j, k)
        b = a
        b[axis] -= 1
        grad = (value(p, a[0], a[1], a[2], c) - value(p, b[0], b[1], b[2], c)) / c.dx
        allowed = open_cell(solid, a[0], a[1], a[2], c) * open_cell(
            solid, b[0], b[1], b[2], c
        )
        if axis == 0:
            u[i, j, k] = (u[i, j, k] - grad) * float(allowed)
        elif axis == 1:
            v[i, j, k] = (v[i, j, k] - grad) * float(allowed)
        else:
            w[i, j, k] = (w[i, j, k] - grad) * float(allowed)


@wp.kernel
def transport(
    T: wp.array3d(dtype=float),
    Y: wp.array3d(dtype=Species),
    mist: wp.array3d(dtype=float),
    tn: wp.array3d(dtype=float),
    yn: wp.array3d(dtype=Species),
    mn: wp.array3d(dtype=float),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
    dt: float,
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    mid = p - velocity(u, v, w, p) * (0.5 * dt / c.dx)
    back = p - velocity(u, v, w, mid) * (dt / c.dx)
    index = wp.vec3i(back)
    temp = float(293.15)
    y = ambient()
    m = float(0.0)
    if inside(index[0], index[1], index[2], c):
        if solid[index[0], index[1], index[2]] == 1:
            back = p
        at = back - wp.vec3(0.5)
        base = wp.vec3i(
            int(wp.floor(at[0])), int(wp.floor(at[1])), int(wp.floor(at[2]))
        )
        f = at - wp.vec3(base)
        temp = 0.0
        y = Species(0.0)
        m = 0.0
        weight_sum = float(0.0)
        for n in range(8):
            offset = wp.vec3i(n % 2, (n / 2) % 2, n / 4)
            q = base + offset
            weight = wp.vec3(1.0) - f
            for d in range(3):
                if offset[d] == 1:
                    weight[d] = f[d]
            weight_scalar = weight[0] * weight[1] * weight[2]
            if inside(q[0], q[1], q[2], c):
                if solid[q[0], q[1], q[2]] == 0:
                    temp += weight_scalar * T[q[0], q[1], q[2]]
                    y += weight_scalar * Y[q[0], q[1], q[2]]
                    m += weight_scalar * mist[q[0], q[1], q[2]]
                    weight_sum += weight_scalar
            else:
                temp += weight_scalar * 293.15
                y += weight_scalar * ambient()
                weight_sum += weight_scalar
        if weight_sum > 0.000001:
            temp /= weight_sum
            y /= weight_sum
            m /= weight_sum
        else:
            temp = T[i, j, k]
            y = Y[i, j, k]
            m = mist[i, j, k]
    if solid[i, j, k] == 1:
        temp = 293.15
        y = ambient()
        m = 0.0
    tn[i, j, k] = temp
    yn[i, j, k] = y
    mn[i, j, k] = m * wp.exp(-0.12 * dt)


@wp.kernel
def mix(
    T: wp.array3d(dtype=float),
    Y: wp.array3d(dtype=Species),
    mist: wp.array3d(dtype=float),
    tn: wp.array3d(dtype=float),
    yn: wp.array3d(dtype=Species),
    mn: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    c: Parameters,
    dt: float,
):
    i, j, k = wp.tid()
    temp = T[i, j, k]
    y = Y[i, j, k]
    m = mist[i, j, k]
    a = wp.min(0.030 * dt / (c.dx * c.dx), 0.15)
    if solid[i, j, k] == 0:
        for axis in range(3):
            for sign in range(2):
                q = wp.vec3i(i, j, k)
                q[axis] += 2 * sign - 1
                if inside(q[0], q[1], q[2], c):
                    if solid[q[0], q[1], q[2]] == 0:
                        temp += a * (T[q[0], q[1], q[2]] - T[i, j, k])
                        y += a * (Y[q[0], q[1], q[2]] - Y[i, j, k])
                        m += a * (mist[q[0], q[1], q[2]] - mist[i, j, k])
                else:
                    temp += a * (293.15 - T[i, j, k])
                    y += a * (ambient() - Y[i, j, k])
                    m -= a * mist[i, j, k]
    total = float(0.0)
    for d in range(6):
        total += y[d]
    tn[i, j, k] = temp
    yn[i, j, k] = y / wp.max(total, 0.000001)
    mn[i, j, k] = m


@wp.kernel
def wood_step(
    wm: wp.array3d(dtype=float),
    wt: wp.array3d(dtype=float),
    film: wp.array3d(dtype=float),
    film_energy: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    T: wp.array3d(dtype=float),
    Y: wp.array3d(dtype=Species),
    source: wp.array3d(dtype=float),
    energy: wp.array3d(dtype=float),
    char: wp.array3d(dtype=float),
    ledger: wp.array(dtype=wp.float64),
    c: Parameters,
    dt: float,
):
    i, j, k = wp.tid()
    mass = wm[i, j, k]
    if mass <= 0.0000001:
        return
    temp = wt[i, j, k]
    heat = float(0.0)
    count = int(0)
    volume = c.dx * c.dx * c.dx
    for axis in range(3):
        for sign in range(2):
            q = wp.vec3i(i, j, k)
            q[axis] += 2 * sign - 1
            if inside(q[0], q[1], q[2], c) and solid[q[0], q[1], q[2]] == 0:
                tg = T[q[0], q[1], q[2]]
                yg = Y[q[0], q[1], q[2]]
                capacity = gas_density(yg, tg) * volume * heat_capacity(yg)
                transfer = c.transfer * c.dx * c.dx * (tg - temp) * dt
                transfer = wp.clamp(
                    transfer,
                    -mass * 1600.0 * wp.max(temp - 293.15, 0.0) / 8.0,
                    capacity * wp.max(tg - 293.15, 0.0) * 0.12,
                )
                heat += transfer
                wp.atomic_add(energy, q[0], q[1], q[2], -transfer)
                count += 1
    if count == 0:
        return
    temp = wp.max(293.15, temp + heat / (mass * 1600.0))
    surface_temperature = temp
    water_mass = film[i, j, k]
    if water_mass > 0.0:
        water_temperature = 293.15 + film_energy[i, j, k] / (water_mass * 4184.0)
        coverage = wp.clamp(
            water_mass / (1000.0 * float(count) * c.dx * c.dx * 0.00005), 0.0, 1.0
        )
        # Thermal resistances across a 3.75 mm wood skin (k=0.15 W/m/K)
        # and its liquid coating set the pyrolysis-interface temperature.
        interface = (40.0 * temp + 500.0 * water_temperature) / 540.0
        surface_temperature = wp.lerp(temp, interface, coverage)
    release = mass * (
        1.0
        - wp.exp(-c.pyrolysis * wp.smoothstep(450.0, 760.0, surface_temperature) * dt)
    )
    pyrolysis_heat = release * 300000.0
    temp = wp.max(293.15, temp - pyrolysis_heat / (mass * 1600.0))
    wm[i, j, k] = mass - release
    wt[i, j, k] = temp
    char[i, j, k] += release
    wp.atomic_add(ledger, 0, wp.float64(release))
    wp.atomic_add(ledger, 1, wp.float64(pyrolysis_heat))
    for axis in range(3):
        for sign in range(2):
            q = wp.vec3i(i, j, k)
            q[axis] += 2 * sign - 1
            if inside(q[0], q[1], q[2], c) and solid[q[0], q[1], q[2]] == 0:
                dm = release / float(count)
                wp.atomic_add(source, q[0], q[1], q[2], dm)
                wp.atomic_add(
                    energy,
                    q[0],
                    q[1],
                    q[2],
                    dm * (0.96 * 2220.0 + 0.04 * 710.0) * (temp - 293.15),
                )


@wp.kernel
def gas_step(
    T: wp.array3d(dtype=float),
    Y: wp.array3d(dtype=Species),
    source: wp.array3d(dtype=float),
    energy: wp.array3d(dtype=float),
    burn: wp.array3d(dtype=float),
    hrr: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=int),
    ledger: wp.array(dtype=wp.float64),
    c: Parameters,
    dt: float,
):
    i, j, k = wp.tid()
    if solid[i, j, k] == 1:
        burn[i, j, k] = 0.0
        return
    temp = T[i, j, k]
    y = Y[i, j, k]
    mass = gas_density(y, temp) * c.dx * c.dx * c.dx
    sensible = mass * heat_capacity(y) * (temp - 293.15) + energy[i, j, k]
    dm = source[i, j, k]
    y *= mass / (mass + dm)
    y[0] += 0.96 * dm / (mass + dm)
    y[5] += 0.04 * dm / (mass + dm)
    temp = 293.15 + wp.max(sensible, 0.0) / ((mass + dm) * heat_capacity(y))
    next_y, released, burned = reaction(y, temp, dt)
    sensible = (mass + dm) * (heat_capacity(y) * (temp - 293.15) + released)
    temp = 293.15 + sensible / ((mass + dm) * heat_capacity(next_y))
    # Resolved boundary radiation/convection is represented by an explicit loss.
    loss = (
        (mass + dm)
        * heat_capacity(next_y)
        * (temp - 293.15)
        * (1.0 - wp.exp(-0.18 * dt))
    )
    temp -= loss / ((mass + dm) * heat_capacity(next_y))
    hrr[i, j, k] = (mass + dm) * released / dt / (c.dx * c.dx * c.dx)
    T[i, j, k] = temp
    Y[i, j, k] = next_y
    burn[i, j, k] = burned * (mass + dm) / dt / (c.dx * c.dx * c.dx)
    wp.atomic_add(ledger, 2, wp.float64((mass + dm) * released))
    wp.atomic_add(ledger, 3, wp.float64(loss))


class Fire:
    def __init__(self, cfg, geometry, device="cuda:0"):
        wp.init()
        self.cfg = cfg
        self.device = device
        self.shape = tuple(cfg["shape"])
        self.time = 0.0
        self.c = Parameters()
        self.c.origin = wp.vec3(*cfg["origin"])
        self.c.dx = cfg["dx"]
        self.c.nx, self.c.ny, self.c.nz = self.shape
        self.c.wind = wp.vec3(*cfg["wind"])
        self.c.pyrolysis = cfg["pyrolysis_rate"]
        self.c.transfer = cfg["wood_transfer"]
        self.c.feedback = cfg["wood_feedback"]
        shape = self.shape
        z = lambda dtype=float: wp.zeros(shape, dtype=dtype, device=device)
        self.T = z()
        self.tn = z()
        self.Y = z(Species)
        self.yn = z(Species)
        self.mist = z()
        self.mn = z()
        self.solid = wp.array(geometry["solid"], dtype=int, device=device)
        self.wood = wp.array(geometry["wood"], dtype=int, device=device)
        self.sdf = wp.array(geometry["sdf"], dtype=float, device=device)
        self.exits = wp.array(geometry["exits"], dtype=wp.vec3i, device=device)
        self.wm = z()
        self.wt = z()
        self.film = z()
        self.film_energy = z()
        self.char = z()
        self.source = z()
        self.energy = z()
        self.burn = z()
        self.hrr = z()
        self.omega = z(wp.vec3)
        self.mag = z()
        self.impulse = z(wp.vec3)
        fshape = tuple(n + 1 for n in shape)
        self.fshape = fshape
        self.u = wp.zeros(fshape, dtype=float, device=device)
        self.v = wp.zeros_like(self.u)
        self.w = wp.zeros_like(self.u)
        self.un = wp.zeros_like(self.u)
        self.vn = wp.zeros_like(self.u)
        self.wn = wp.zeros_like(self.u)
        self.p = z()
        self.div = z()
        self.ledger = wp.zeros(8, dtype=wp.float64, device=device)
        wp.launch(
            initialize,
            shape,
            [
                self.T,
                self.Y,
                self.wood,
                self.solid,
                self.wm,
                self.wt,
                cfg["wood_density"],
                cfg["wood_initial_temperature"],
                self.c,
            ],
            device=device,
        )
        self.initial_wood_mass = float(self.wm.numpy().sum())
        from .wood_thermal import WoodThermal

        self.wood_thermal = WoodThermal(self)

    def project(self, iterations=28):
        wp.launch(
            divergence,
            self.shape,
            [self.u, self.v, self.w, self.div, self.solid, self.c],
            device=self.device,
        )
        for _ in range(iterations):
            for color in [0, 1]:
                wp.launch(
                    pressure,
                    self.shape,
                    [self.p, self.div, self.solid, self.c, color],
                    device=self.device,
                )
        wp.launch(
            project,
            self.fshape,
            [self.u, self.v, self.w, self.p, self.solid, self.c],
            device=self.device,
        )

    def step(self, dt):
        wp.launch(
            curl_field,
            self.shape,
            [self.u, self.v, self.w, self.omega, self.mag, self.c.dx],
            device=self.device,
        )
        wp.launch(
            flow,
            self.fshape,
            [
                self.u,
                self.v,
                self.w,
                self.un,
                self.vn,
                self.wn,
                self.T,
                self.omega,
                self.mag,
                self.impulse,
                self.solid,
                self.c,
                dt,
                self.time,
            ],
            device=self.device,
        )
        self.u, self.un = self.un, self.u
        self.v, self.vn = self.vn, self.v
        self.w, self.wn = self.wn, self.w
        self.impulse.zero_()
        self.project()
        wp.launch(
            transport,
            self.shape,
            [
                self.T,
                self.Y,
                self.mist,
                self.tn,
                self.yn,
                self.mn,
                self.u,
                self.v,
                self.w,
                self.solid,
                self.c,
                dt,
            ],
            device=self.device,
        )
        self.T, self.tn = self.tn, self.T
        self.Y, self.yn = self.yn, self.Y
        self.mist, self.mn = self.mn, self.mist
        wp.launch(
            mix,
            self.shape,
            [
                self.T,
                self.Y,
                self.mist,
                self.tn,
                self.yn,
                self.mn,
                self.solid,
                self.c,
                dt,
            ],
            device=self.device,
        )
        self.T, self.tn = self.tn, self.T
        self.Y, self.yn = self.yn, self.Y
        self.mist, self.mn = self.mn, self.mist
        self.source.zero_()
        self.energy.zero_()
        self.wood_thermal.step(dt)
        wp.launch(
            wood_step,
            self.shape,
            [
                self.wm,
                self.wt,
                self.film,
                self.film_energy,
                self.solid,
                self.T,
                self.Y,
                self.source,
                self.energy,
                self.char,
                self.ledger,
                self.c,
                dt,
            ],
            device=self.device,
        )
        wp.launch(
            gas_step,
            self.shape,
            [
                self.T,
                self.Y,
                self.source,
                self.energy,
                self.burn,
                self.hrr,
                self.solid,
                self.ledger,
                self.c,
                dt,
            ],
            device=self.device,
        )
        self.time += dt

    def diagnostics(self):
        t = self.T.numpy()
        y = self.Y.numpy()
        wood = self.wood.numpy().astype(bool)
        wt = self.wt.numpy()
        mass = self.wm.numpy()
        return dict(
            time=self.time,
            temperature_max=float(t.max()),
            fuel_mass_fraction_max=float(y[..., 0].max()),
            species_sum_error=float(np.max(np.abs(y.sum(-1) - 1))),
            species_min=float(y.min()),
            heat_release_kw=float(self.hrr.numpy().sum() * self.c.dx**3 / 1000),
            wood_temperature_mean=float(
                np.average(wt[wood], weights=mass[wood] + 1e-12)
            ),
            wood_remaining_kg=float(mass.sum()),
            wood_initial_kg=self.initial_wood_mass,
            velocity_max=max(
                float(np.abs(a.numpy()).max()) for a in [self.u, self.v, self.w]
            ),
            ledger=self.ledger.numpy().tolist(),
        )
