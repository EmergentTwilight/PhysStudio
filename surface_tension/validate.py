"""Energy-gradient, rigid-contact Jacobian, and trajectory checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .solver import CapillarySystem
from .scenes import make_scene
from .stirrer import Stirrer
from .interaction_checks import check_interactions


def numerical_checks():
    bodies,_=make_scene('orchard');bodies=bodies[:1]
    system=CapillarySystem(bodies,nx=32,ny=26)
    rng=np.random.default_rng(10)
    h=rng.normal(0,.00015,system.n)
    direction=rng.normal(size=system.n);direction/=np.linalg.norm(direction)
    eps=1.e-8
    analytic=np.dot(system.surface(h)[1],direction)
    finite=(system.surface(h+eps*direction)[0]-system.surface(h-eps*direction)[0])/(2*eps)
    err=abs(analytic-finite)/max(abs(analytic),1.e-12)
    assert err<1.e-5,err
    q=np.array([b.position for b in bodies]);R=np.array([b.rotation for b in bodies])
    B,idx,z,owners=system.contact(q,R)
    jac=-B[:,system.n:].toarray()
    jac_errors=[]
    for dof in range(6):
        values=[]
        for sign in [-1,1]:
            qq=q.copy();RR=R.copy()
            if dof<3:qq[0,dof]+=sign*eps
            else:
                dr=np.zeros(3);dr[dof-3]=sign*eps/bodies[0].scale
                RR[0]=Rotation.from_rotvec(dr).as_matrix()@R[0]
            _,ii,zz,_=system.contact(qq,RR)
            assert np.array_equal(ii,idx)
            values.append(zz)
        fd=(values[1]-values[0])/(2*eps)
        jac_errors.append(float(np.max(np.abs(fd-jac[:,dof]))))
    assert max(jac_errors)<1.e-4,jac_errors
    impulses=rng.uniform(0,1,len(idx))
    reaction=B.T@impulses
    cancellation=float(reaction[:system.n].sum()+reaction[system.n+2::6].sum())
    assert abs(cancellation)<1.e-10
    a=rng.normal(size=system.n);b=rng.normal(size=system.n)
    symmetry=abs(np.dot(a,system.mass(b))-np.dot(b,system.mass(a)))
    assert symmetry<1.e-9
    assert np.dot(a,system.mass(a))>0
    wire_bodies,_=make_scene('paperclips')
    wire=CapillarySystem(wire_bodies,nx=64,ny=51)
    q=np.array([b.position for b in wire_bodies]);R=np.array([b.rotation for b in wire_bodies])
    fast=wire.contact(q,R);reference=wire.contact(q,R,fast_spheres=False)
    assert np.array_equal(fast[1],reference[1])
    sphere_error=float(np.max(np.abs(fast[2]-reference[2])))
    jacobian_error=float(np.max(np.abs((fast[0]-reference[0]).toarray())))
    assert sphere_error<1.e-12 and jacobian_error<1.e-10
    probe=Stirrer('paperclips');body=wire_bodies[0]
    pos=body.position.copy();rot=body.rotation.copy()
    tip=pos+np.array([.002,.001,.0001])
    _,probe_jac=probe.solid_contact(body,pos,rot,tip)
    probe_errors=[]
    for dof in range(6):
        values=[]
        for sign in [-1,1]:
            qq=pos.copy();RR=rot.copy()
            if dof<3:qq[dof]+=sign*eps
            else:
                dr=np.zeros(3);dr[dof-3]=sign*eps/body.scale
                RR=Rotation.from_rotvec(dr).as_matrix()@rot
            values.append(probe.solid_contact(body,qq,RR,tip)[0])
        probe_errors.append(abs((values[1]-values[0])/(2*eps)-probe_jac[dof]))
    assert max(probe_errors)<1.e-5,probe_errors
    return dict(surface_gradient_relative_error=err,contact_jacobian_max_errors=jac_errors,
                contact_vertical_action_reaction_error=cancellation,fluid_mass_symmetry_error=float(symmetry),
                fluid_mass_positive=True,batched_wire_height_error=sphere_error,
                batched_wire_jacobian_error=jacobian_error,
                stirrer_solid_jacobian_error=float(max(probe_errors)))


def trajectory(path):
    cfg=json.loads((path/'config.json').read_text())
    records=json.loads((path/'audit.json').read_text())
    files=sorted((path/'states').glob('*.npz'))
    assert len(files)==cfg['frames']
    for file in files:
        with np.load(file) as s:
            for k in s.files:assert np.isfinite(s[k]).all(),(file,k)
            for R in s['rotations']:assert np.linalg.norm(R.T@R-np.eye(3))<1.e-6
            if cfg.get('pinned_rim'):
                h=s['height']
                assert np.max(np.abs(np.r_[h[0],h[-1],h[:,0],h[:,-1]]))<1.e-12
    first=np.load(files[0]);last=np.load(files[-1])
    relative_volume=max(abs(r['volume_change']) for r in records)/(cfg['length']*cfg['width']*cfg['depth'])
    max_penetration=max(r['max_penetration'] for r in records)
    assert relative_volume<1.e-4,relative_volume
    assert max_penetration<.00010,max_penetration
    def spread(p):return float(np.mean(np.linalg.norm(p[:,:2]-p[:,:2].mean(0),axis=1)))
    result=dict(path=str(path),frames=len(files),physical_seconds=float(last['time']),
        max_relative_volume_change=relative_volume,max_penetration_um=max_penetration*1.e6,
        initial_spread_mm=spread(first['positions'])*1.e3,final_spread_mm=spread(last['positions'])*1.e3,
        final_height_mm=(last['positions'][:,2]*1.e3).tolist(),
        max_cg_iterations=max(it['cg'] for r in records for it in r['iterations']))
    tail=files[3*len(files)//4:]
    speeds=[];angular=[];heights=[]
    for file in tail:
        with np.load(file) as state:
            speeds.append(np.linalg.norm(state['body_velocity'],axis=1))
            angular.append(np.linalg.norm(state['body_omega'],axis=1))
            heights.append(np.sqrt(np.mean(state['velocity']**2)))
    result.update(final_quarter_mean_speed_mm_s=float(np.mean(speeds)*1.e3),
                  final_quarter_mean_angular_speed_rad_s=float(np.mean(angular)),
                  final_quarter_surface_rms_speed_mm_s=float(np.mean(heights)*1.e3))
    if cfg.get('stirrer'):
        result['probe_solid_penetration_um']=max(r['probe_solid_penetration'] for r in records)*1.e6
        assert result['probe_solid_penetration_um']<100.,result
        if 'wire_pair_penetration' in records[0]:
            result['wire_pair_penetration_um']=max(r['wire_pair_penetration'] for r in records)*1.e6
            assert result['wire_pair_penetration_um']<100.,result
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--simulations',type=Path,nargs='*',default=[])
    args=p.parse_args()
    report=dict(numerics=numerical_checks(),interactions=check_interactions(),
                trajectories=[trajectory(s) for s in args.simulations],status='passed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
