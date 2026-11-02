"""Mass-balanced methane-proxy chemistry and enthalpy-limited water exchange."""
import warp as wp

Species = wp.vec(length=6, dtype=wp.float32)
# Fuel, oxygen, carbon dioxide, water vapor, nitrogen, carbonaceous residual.


@wp.func
def heat_capacity(y: Species):
    return (
        y[0] * 2220.0
        + y[1] * 918.0
        + y[2] * 840.0
        + y[3] * 1860.0
        + y[4] * 1040.0
        + y[5] * 710.0
    )


@wp.func
def gas_density(y: Species, temperature: float):
    reciprocal = (
        y[0] / 0.016
        + y[1] / 0.032
        + y[2] / 0.044
        + y[3] / 0.018
        + y[4] / 0.028
        + y[5] / 0.012
    )
    return 101325.0 / (
        8.3144626 * wp.max(temperature, 293.15) * wp.max(reciprocal, 1.0)
    )


@wp.func
def ambient():
    return Species(0.0, 0.233, 0.0, 0.0, 0.767, 0.0)


@wp.func
def reaction(y: Species, temperature: float, dt: float):
    # Fire-X Eqs. 4-7. One mole CH4 needs two moles O2; heat is 834 kJ/mol.
    # The oxygen-poor branch CH4 + O2 -> C + 2H2O preserves elemental mass.
    ratio = y[1] / wp.max(4.0 * y[0], 0.000001)
    complete = wp.clamp(ratio, 0.0, 1.0)
    oxygen_ratio = 2.0 + 2.0 * complete
    available = wp.min(y[0], y[1] / oxygen_ratio)
    rate = 3500.0 * wp.exp(-5200.0 / wp.max(temperature, 293.15))
    burned = wp.max(available, 0.0) * (1.0 - wp.exp(-rate * dt))
    out = y
    out[0] -= burned
    out[1] -= oxygen_ratio * burned
    out[2] += 2.75 * complete * burned
    out[3] += 2.25 * burned
    out[5] += 0.75 * (1.0 - complete) * burned
    released = burned * (52125000.0 * complete + 19000000.0 * (1.0 - complete))
    return out, released, burned


@wp.func
def water_exchange(
    gas_mass: float,
    gas_cp: float,
    temperature: float,
    water_mass: float,
    water_energy: float,
    conductance: float,
    dt: float,
):
    # Liquid enthalpy uses ambient water as zero. Vapor reference enthalpy is
    # L + (cp_liquid-cp_vapor)*(Tb-Ta); this keeps phase energy accounting exact.
    ta = 293.15
    tb = 373.15
    cp = 4184.0
    latent = 2256000.0
    capacity = wp.max(gas_mass * gas_cp, 0.00000001)
    water_capacity = wp.max(water_mass * cp, 0.00000001)
    tw = ta + water_energy / water_capacity
    fraction = 1.0 - wp.exp(-conductance * dt * (1.0 / capacity + 1.0 / water_capacity))
    transfer = (temperature - tw) * fraction / (1.0 / capacity + 1.0 / water_capacity)
    # Once the water reaches boiling, excess sensible heat pays latent heat.
    energy = wp.max(0.0, water_energy + transfer)
    evaporated = wp.min(
        water_mass, wp.max(energy - water_mass * cp * (tb - ta), 0.0) / latent
    )
    energy -= evaporated * (latent + cp * (tb - ta))
    if evaporated >= water_mass:
        # Fully evaporated water cannot retain sensible energy. Return the
        # excess to the hot phase before adding the vapor's sensible enthalpy.
        transfer -= wp.max(energy, 0.0)
        energy = 0.0
    sensible_gas = (
        capacity * (temperature - ta) - transfer + evaporated * 1860.0 * (tb - ta)
    )
    return evaporated, wp.max(energy, 0.0), sensible_gas, transfer
