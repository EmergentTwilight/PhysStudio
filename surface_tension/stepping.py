"""Rollback and local timestep refinement for driven capillary scenes."""
from copy import deepcopy


def capture(system):
    result = dict(h=system.h.copy(), v=system.v.copy(), time=system.time,
                  last=deepcopy(system.last),
                  bodies=[tuple(getattr(b, key).copy() for key in
                          ['position','rotation','velocity','omega']) for b in system.bodies])
    if system.stirrer is not None:
        result['center'] = system.stirrer.center.copy()
    if system.flow is not None:
        result['flow'] = {key:deepcopy(getattr(system.flow,key)) for key in
                          ['u','w','tracers','last','height_flux','momentum_error']
                          if hasattr(system.flow,key)}
    return result


def restore(system, state):
    system.h = state['h']; system.v = state['v']; system.time = state['time']
    system.last = state['last']
    for body, values in zip(system.bodies, state['bodies']):
        for key, value in zip(['position','rotation','velocity','omega'], values):
            setattr(body, key, value)
    if system.stirrer is not None:system.stirrer.center = state['center']
    if system.flow is not None:
        for key, value in state['flow'].items():setattr(system.flow,key,value)


def advance(system, iterations, level=0):
    """Retry a rejected step as two half steps without clipping the velocity."""
    if system.flow is None:
        result = dict(system.step(iterations)); result['integration_dt'] = system.dt
        return [result]
    state = capture(system); dt = system.dt
    reason = None
    try:
        result = dict(system.step(iterations))
        if result['flow_cfl'] > .85:
            reason = f"Surface flow CFL {result['flow_cfl']:.3f}"
        elif max(result['continuous_probe_penetration'],result['continuous_pair_penetration']) > 5.e-6:
            reason = 'Continuous solid contact exceeds 5 micrometers'
        else:
            result.update(integration_dt=dt, refinement_level=level)
            return [result]
    except RuntimeError as error:
        if not str(error).startswith('Surface flow CFL'):raise
        reason = str(error)
    restore(system,state)
    if level >= 5:raise RuntimeError(f'Adaptive step failed at dt={dt}: {reason}')
    system.dt = dt/2
    try:
        return advance(system,iterations,level+1)+advance(system,iterations,level+1)
    finally:
        system.dt = dt
