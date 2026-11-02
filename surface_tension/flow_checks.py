"""Checks for continuous contacts, staggered currents, and cached trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .scenes import make_scene
from .solver import CapillarySystem
from .surface_flow import SurfaceFlow
from .wire_contact import closest_segments, wire_contacts, rigid_probe_contacts, rigid_pair_contacts
from .stirrer import Stirrer


def numerics():
    p, q = closest_segments(np.array([[-1., 0, 0]]), np.array([[1., 0, 0]]),
                            np.array([[0., -1., .1]]), np.array([[0., 1., .1]]))
    assert np.allclose(p, [[0, 0, 0]]) and np.allclose(q, [[0, 0, .1]])
    a, b = make_scene('paperclips')[0][:2]
    q1 = np.zeros(3); q2 = np.array([.0054, .00043, .00024])
    R1, R2 = a.rotation, b.rotation
    contacts = wire_contacts(a, q1, R1, b, q2, R2)
    grad = sum(gap*np.r_[j1, j2] for gap, j1, j2 in contacts)
    fd = []; eps = 1.e-9
    for dof in range(12):
        values = []
        for sign in [-1, 1]:
            p = q1.copy(); q = q2.copy(); A = R1.copy(); B = R2.copy(); axis = dof % 6
            if axis < 3:
                (p if dof < 6 else q)[axis] += sign*eps
            else:
                dr = np.zeros(3); dr[axis-3] = sign*eps/(a.scale if dof < 6 else b.scale)
                if dof < 6:A = Rotation.from_rotvec(dr).as_matrix()@A
                else:B = Rotation.from_rotvec(dr).as_matrix()@B
            values.append(sum(.5*g*g for g, _, _ in wire_contacts(a, p, A, b, q, B)))
        fd.append((values[1]-values[0])/(2*eps))
    error = float(np.max(np.abs(grad-fd)))
    assert error < 1.e-7
    assert max(np.linalg.norm(j1[:3]+j2[:3]) for _, j1, j2 in contacts) < 1.e-12
    s = CapillarySystem([], nx=40, ny=32); flow = SurfaceFlow(s)
    rng = np.random.default_rng(4); target = rng.normal(size=(40, 32)); target -= target.mean()
    residual = flow.correct_divergence(target)
    assert residual < 1.e-10
    assert np.max(np.abs(flow.u[[0, -1]])) == 0
    assert np.max(np.abs(flow.w[:, [0, -1]])) == 0
    # The discrete gradient/divergence and finite-depth inverse must agree.
    rngfield = rng.normal(size=target.shape)
    gx, gy = flow.gradient(rngfield)
    laplacian = flow.filter(rngfield, -flow.k2)
    laperror = float(np.max(np.abs(flow.divergence(gx, gy)-laplacian)) / np.max(np.abs(laplacian)))
    assert laperror < 1.e-12
    # Finite-difference the refined triangle interpolation and moving support
    # envelope together, independent of the assembled contact Jacobian.
    body=make_scene('orchard')[0][:1]
    refined=CapillarySystem(body,nx=32,ny=26);refined.refine_liquid_contact()
    pos=np.asarray([body[0].position]);rot=np.asarray([body[0].rotation])
    B,idx,z,_=refined.contact(pos,rot)
    direction=rng.normal(size=refined.size)
    h=rng.normal(0,.0001,refined.n);values=[];eps=1.e-9
    for sign in [-1,1]:
        q=pos+sign*eps*direction[refined.n:refined.n+3]
        dr=sign*eps*direction[refined.n+3:]/body[0].scale
        R=np.asarray([Rotation.from_rotvec(dr).as_matrix()@rot[0]])
        _,ii,zz,_=refined.contact(q,R)
        assert np.array_equal(ii,idx)
        values.append((refined.contact_interpolation@(h+sign*eps*direction[:refined.n]))[idx]-zz)
    refined_error=float(np.max(np.abs((values[1]-values[0])/(2*eps)-B@direction)))
    assert refined_error<1.e-5
    reactions=B.T@rng.uniform(size=len(idx))
    refined_balance=float(abs(reactions[:refined.n].sum()+reactions[refined.n+2]))
    assert refined_balance<1.e-10
    # Independently exercise a driven fluid patch: motion must enter height
    # and vorticity through solver state, with no renderer involved.
    driven = CapillarySystem([], nx=40, ny=32, dt=1/720)
    driven.stirrer = Stirrer('paperclips', flow_coupled=True)
    current = SurfaceFlow(driven)
    previous = np.array([-.006, 0., -.0005])
    for step in range(120):
        t = (step+1)*driven.dt
        tip = np.array([-.006+.04*t, .002*np.sin(20*t), -.0005])
        old_h = driven.h.copy()
        predictor = current.prepare(tip, previous)
        stiffness = driven.rho*driven.gravity*driven.area+driven.sigma*driven.area*driven.k**2
        symbol = driven.mass_symbol/(driven.mass_symbol+driven.dt**2*stiffness)
        driven.h = driven.project_surface(current.filter(predictor.reshape(40,32), symbol).ravel())
        current.finish(old_h)
        previous = tip
    assert current.last['flow_rms_speed'] > .001
    assert current.last['flow_rms_vorticity'] > .1
    assert np.std(driven.h) > 1.e-7
    assert abs(driven.h.sum()) < 1.e-12
    return dict(continuous_wire_energy_gradient_error=error,
                staggered_continuity_error=residual, laplacian_relative_error=laperror,
                refined_contact_directional_error=refined_error,
                refined_contact_force_balance_error=refined_balance,
                driven_patch_speed_mm_s=current.last['flow_rms_speed']*1000,
                driven_patch_vorticity=current.last['flow_rms_vorticity'],
                driven_patch_height_rms_um=float(np.std(driven.h)*1.e6))


def triangle_height(height, points, cfg):
    nx,ny=height.shape
    x=(points[:,0]+cfg['length']/2)/cfg['length']*nx-.5
    y=(points[:,1]+cfg['width']/2)/cfg['width']*ny-.5
    inside=(x>=0)&(y>=0)&(x<=nx-1)&(y<=ny-1)
    x=x[inside];y=y[inside]
    i=np.minimum(x.astype(int),nx-2);j=np.minimum(y.astype(int),ny-2)
    fx=x-i;fy=y-j
    z=np.where(fx+fy<=1,(1-fx-fy)*height[i,j]+fx*height[i+1,j]+fy*height[i,j+1],
               (fx+fy-1)*height[i+1,j+1]+(1-fx)*height[i,j+1]+(1-fy)*height[i+1,j])
    return float(np.maximum(z-points[inside,2],0).max(initial=0.))


def mesh_contact_penetration(state, bodies, cfg):
    maximum=0.
    disk=np.array([[f*np.cos(a),f*np.sin(a),-np.sqrt(1-f*f)]
                   for f in [0.,.25,.5,.75,.90,.97]
                   for a in np.linspace(0,2*np.pi,24,endpoint=False)])
    for body,q,R in zip(bodies,state['positions'],state['rotations']):
        if len(body.centers)>1:
            line=np.concatenate([body.centers,.5*(body.centers[:-1]+body.centers[1:])])
            centers=line@R.T+q
            points=(centers[:,None]+disk[None]*body.radii[0,0]).reshape(-1,3)
        else:
            points=(disk*body.radii[0])@R.T+q
        maximum=max(maximum,triangle_height(state['height'],points,cfg))
    points=state['probe_position']+disk*cfg['stirrer']['radius']
    maximum=max(maximum,triangle_height(state['height'],points,cfg))
    return maximum


def trajectory(folder):
    cfg = json.loads((folder/'config.json').read_text())
    assert cfg.get('surface_flow')
    states = sorted((folder/'states').glob('*.npz'))
    assert len(states) == cfg['frames']
    records = json.loads((folder/'audit.json').read_text())
    bodies = make_scene(cfg['scene'])[0]
    probe_max = 0.; pairs_max = 0.; water_max = 0.; flow = []; wave = []
    nx, ny = cfg['nx'], cfg['ny']
    xx, yy = np.meshgrid((np.arange(nx)+.5)*cfg['length']/nx-cfg['length']/2,
                         (np.arange(ny)+.5)*cfg['width']/ny-cfg['width']/2, indexing='ij')
    for f in states:
        with np.load(f) as state:
            for key in state.files:assert np.isfinite(state[key]).all(), (f, key)
            water_max=max(water_max,mesh_contact_penetration(state,bodies,cfg))
            q, R, tip = state['positions'], state['rotations'], state['probe_position']
            for i, b in enumerate(bodies):
                c = rigid_probe_contacts(b, q[i], R[i], tip, cfg['stirrer']['radius'], cfg['stirrer']['length'])
                probe_max = max([probe_max, *[-x[0] for x in c]])
                for j in range(i):
                    c = rigid_pair_contacts(b, q[i], R[i], bodies[j], q[j], R[j])
                    pairs_max = max([pairs_max, *[-x[0] for x in c]])
            u = .5*(state['flow_u'][1:]+state['flow_u'][:-1])
            v = .5*(state['flow_v'][:, 1:]+state['flow_v'][:, :-1])
            far = (xx-tip[0])**2+(yy-tip[1])**2 > .004**2
            for pos in q:
                far &= (xx-pos[0])**2+(yy-pos[1])**2 > .0045**2
            flow.append(float(np.sqrt(np.mean((u*u+v*v)[far]))))
            wave.append(float(np.sqrt(np.mean(state['velocity'][far]**2))))
    subprobe = max(row['substep_maxima']['continuous_probe_penetration'] for row in records)
    subpair = max(row['substep_maxima']['continuous_pair_penetration'] for row in records)
    cfl = max(row['substep_maxima']['flow_cfl'] for row in records)
    assert max(probe_max, pairs_max, subprobe, subpair) < .00001
    assert cfl < 1
    assert water_max < .00010, water_max
    volume = max(abs(row['volume_change']) for row in records)/(cfg['length']*cfg['width']*cfg['depth'])
    assert volume < 1.e-10
    residual = max(row['flow_continuity_error'] for row in records)
    assert residual < 1.e-7
    return dict(simulation=str(folder.resolve()), frames=len(states),
                saved_continuous_probe_penetration_um=probe_max*1.e6,
                saved_continuous_pair_penetration_um=pairs_max*1.e6,
                all_substep_probe_penetration_um=subprobe*1.e6,
                all_substep_pair_penetration_um=subpair*1.e6,
                dense_triangle_water_penetration_um=water_max*1.e6,
                volume_relative_drift=volume, maximum_cfl=cfl,
                continuity_residual=residual,
                final_half_far_water_rms_speed_mm_s=float(np.mean(flow[len(flow)//2:])*1.e3),
                final_half_far_surface_rms_speed_mm_s=float(np.mean(wave[len(wave)//2:])*1.e3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--simulations', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = dict(numerics=numerics(), trajectories=[trajectory(f) for f in args.simulations], status='passed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
