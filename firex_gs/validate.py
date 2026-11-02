"""Tests of implemented reaction, vaporization energy, and species balance."""
import argparse, json
from pathlib import Path
import numpy as np
import warp as wp
from .chemistry import Species, reaction, heat_capacity, water_exchange
from .water import drag_increment


@wp.kernel
def reaction_probe(
    y: wp.array(dtype=Species),
    t: wp.array(dtype=float),
    out: wp.array(dtype=Species),
    released: wp.array(dtype=float),
):
    i = wp.tid()
    result, energy, burned = reaction(y[i], t[i], 0.02)
    out[i] = result
    released[i] = energy


@wp.kernel
def exchange_probe(out: wp.array2d(dtype=float)):
    i = wp.tid()
    gm = 0.001
    cp = 1200.0
    temp = 1300.0
    wm = 0.0001
    we = float(i % 2) * wm * 4184.0 * 80.0
    conductance = 2.0
    if i >= 2:
        conductance = 2000.0
    dm, e, h, q = water_exchange(gm, cp, temp, wm, we, conductance, 0.02)
    out[i, 0] = dm
    out[i, 1] = e
    out[i, 2] = h
    out[i, 3] = q
    out[i, 4] = we


@wp.kernel
def drag_probe(out: wp.array2d(dtype=float)):
    i = wp.tid()
    mg = 0.0002592
    mw = wp.pow(10.0, float(i) - 5.0)
    gas = wp.vec3(0.0, 0.0, 0.0)
    liquid = wp.vec3(10.0, 0.0, 0.0)
    delta = drag_increment(gas, liquid, mg, mw, 0.008333333)
    water_new = liquid + delta
    gas_new = gas - delta * mw / mg
    momentum = mg * gas_new + mw * water_new - mw * liquid
    before = 0.5 * mw * wp.dot(liquid, liquid)
    after = 0.5 * mw * wp.dot(water_new, water_new) + 0.5 * mg * wp.dot(
        gas_new, gas_new
    )
    out[i, 0] = wp.length(momentum)
    out[i, 1] = after / before
    out[i, 2] = gas_new[0]
    out[i, 3] = water_new[0]


def projection_probe(device):
    from .solver import Fire, divergence
    from .scenes import PRESETS

    cfg = dict(PRESETS["firewoods"])
    cfg.update(shape=[18, 18, 18], origin=[0.0, 0.0, 0.0], dx=0.1)
    shape = tuple(cfg["shape"])
    solid = np.zeros(shape, dtype="i")
    solid[:, :, 0] = 1
    solid[7:11, 7:11, 1:6] = 1
    geo = dict(
        solid=solid,
        wood=np.zeros(shape, "i"),
        sdf=np.ones(shape, "f"),
        exits=np.zeros(shape + (3,), "i"),
    )
    sim = Fire(cfg, geo, device)
    rng = np.random.default_rng(7)
    fields = []
    for axis in range(3):
        values = rng.normal(size=sim.fshape).astype("f")
        for index in np.ndindex(sim.fshape):
            a = np.array(index)
            b = a.copy()
            b[axis] -= 1
            for q in [a, b]:
                if ((q >= 0) & (q < np.array(shape))).all() and solid[tuple(q)]:
                    values[index] = 0.0
        fields.append(values)
    for field, values in zip([sim.u, sim.v, sim.w], fields):
        field.assign(values)
    wp.launch(
        divergence,
        shape,
        [sim.u, sim.v, sim.w, sim.div, sim.solid, sim.c],
        device=device,
    )
    before = float(np.sqrt(np.mean(sim.div.numpy() ** 2)))
    sim.project(180)
    wp.launch(
        divergence,
        shape,
        [sim.u, sim.v, sim.w, sim.div, sim.solid, sim.c],
        device=device,
    )
    after = float(np.sqrt(np.mean(sim.div.numpy() ** 2)))
    wall = 0.0
    for axis, field in enumerate([sim.u, sim.v, sim.w]):
        values = field.numpy()
        for index in np.ndindex(sim.fshape):
            a = np.array(index)
            b = a.copy()
            b[axis] -= 1
            for q in [a, b]:
                if ((q >= 0) & (q < np.array(shape))).all() and solid[tuple(q)]:
                    wall = max(wall, abs(float(values[index])))
    assert after / before < 0.002 and wall < 1e-6, (before, after, wall)
    from .solver import flow, transport

    sim.c.dx = 0.035
    sim.c.wind = wp.vec3(0.0)
    sim.solid.zero_()
    sim.T.fill_(293.15)
    dt = 1 / 180
    checker = (
        np.broadcast_to(
            (np.arange(sim.fshape[1]) % 2 * 2 - 1)[None, :, None], sim.fshape
        )
        .astype("f")
        .copy()
    )
    sim.u.assign(checker)
    sim.v.fill_(sim.c.dx / dt)
    sim.w.zero_()
    wp.launch(
        flow,
        sim.fshape,
        [
            sim.u,
            sim.v,
            sim.w,
            sim.un,
            sim.vn,
            sim.wn,
            sim.T,
            sim.omega,
            sim.mag,
            sim.impulse,
            sim.solid,
            sim.c,
            dt,
            0.0,
        ],
        device=device,
    )
    checker_after = float(np.abs(sim.un.numpy()[2:-2, 2:-2, 2:-2]).max())
    assert checker_after < 0.85, checker_after
    # A solid wall must not act as an oxygen source during interpolation.
    wallmask = np.zeros(shape, "i")
    wallmask[8:, :, :] = 1
    sim.solid.assign(wallmask)
    composition = np.zeros(shape + (6,), "f")
    composition[..., 4] = 1.0
    composition[8:, :, :, 1] = 0.233
    composition[8:, :, :, 4] = 0.767
    sim.Y.assign(composition)
    sim.T.fill_(600.0)
    sim.u.fill_(-0.75)
    sim.v.zero_()
    sim.w.zero_()
    wp.launch(
        transport,
        shape,
        [
            sim.T,
            sim.Y,
            sim.mist,
            sim.tn,
            sim.yn,
            sim.mn,
            sim.u,
            sim.v,
            sim.w,
            sim.solid,
            sim.c,
            dt,
        ],
        device=device,
    )
    oxygen = float(sim.yn.numpy()[7, 2:-2, 2:-2, 1].max())
    assert oxygen == 0.0, oxygen
    return dict(
        before=before,
        after=after,
        ratio=after / before,
        solid_normal_speed=wall,
        viscous_checker_amplitude=checker_after,
        solid_oxygen_leak=oxygen,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    wp.init()
    y = np.array(
        [
            [0.04, 0.233, 0, 0, 0.727, 0],
            [0.15, 0.03, 0, 0, 0.82, 0],
            [0.04, 0, 0, 0, 0.96, 0],
            [0.04, 0.233, 0, 0, 0.727, 0],
        ],
        "f",
    )
    temp = np.array([1300, 1300, 1300, 293.15], "f")
    yn = wp.zeros(4, dtype=Species, device=a.device)
    q = wp.zeros(4, dtype=float, device=a.device)
    wp.launch(
        reaction_probe,
        4,
        [
            wp.array(y, dtype=Species, device=a.device),
            wp.array(temp, device=a.device),
            yn,
            q,
        ],
        device=a.device,
    )
    result = yn.numpy()
    heat = q.numpy()
    # Element mass fractions of C,H,O,N in CH4/O2/CO2/H2O/N2/C.
    elements = np.array(
        [
            [0.75, 0.25, 0, 0],
            [0, 0, 1, 0],
            [12 / 44, 0, 32 / 44, 0],
            [0, 2 / 18, 16 / 18, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )
    error = float(np.max(np.abs(result @ elements - y @ elements)))
    assert (
        error < 1e-7
        and result.min() >= -1e-7
        and np.max(np.abs(result.sum(1) - 1)) < 1e-6
    )
    assert heat[2] == 0 and heat[0] > heat[3] * 1000
    probe = wp.zeros((4, 5), dtype=float, device=a.device)
    wp.launch(exchange_probe, 4, [probe], device=a.device)
    vals = probe.numpy()
    errors = []
    for dm, e, h, transfer, we in vals:
        initial = 0.001 * 1200 * (1300 - 293.15) + we
        final = h + e + dm * (2256000 + (4184 - 1860) * 80)
        errors.append(float(abs(final - initial) / initial))
        assert 0 <= dm <= 0.0001 and e >= 0 and h >= 0
    assert max(errors) < 2e-6
    assert np.allclose(vals[2:, 0], 0.0001) and (vals[2:, 1] == 0).all()
    drag = wp.zeros((6, 4), dtype=float, device=a.device)
    wp.launch(drag_probe, 6, [drag], device=a.device)
    drag_values = drag.numpy()
    assert drag_values[:, 0].max() < 2e-6 and drag_values[:, 1].max() <= 1.000001
    assert drag_values[:, 2:].min() > -1e-5 and drag_values[:, 2:].max() < 10.00001
    projection = projection_probe(a.device)
    report = dict(
        passed=True,
        drag_momentum_error=float(drag_values[:, 0].max()),
        drag_kinetic_energy_ratio=drag_values[:, 1].tolist(),
        pressure_projection=projection,
        elemental_mass_error=error,
        cold_to_hot_heat_ratio=float(heat[3] / heat[0]),
        oxygen_free_heat=float(heat[2]),
        water_enthalpy_relative_errors=errors,
        evaporated_kg=vals[:, 0].tolist(),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
