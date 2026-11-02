"""Cache independently replayable capillary-contact trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .solver import CapillarySystem
from .scenes import make_scene
from .stirrer import Stirrer
from .surface_flow import SurfaceFlow
from .stepping import advance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=['paperclips','capillary_cluster','orchard'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=180)
    p.add_argument('--resolution',type=int,default=96)
    p.add_argument('--sigma',type=float,default=.072)
    p.add_argument('--viscosity',type=float,default=.003)
    p.add_argument('--dt',type=float,default=1/240)
    p.add_argument('--frame-dt',type=float,default=1/60)
    p.add_argument('--newton',type=int,default=6)
    p.add_argument('--stirrer',action='store_true')
    p.add_argument('--surface-flow',action='store_true')
    p.add_argument('--settle',type=float,default=0.,help='Physical seconds before recording')
    p.add_argument('--pinned-rim',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'states').mkdir()
    bodies,visual=make_scene(args.scene)
    system=CapillarySystem(bodies,nx=args.resolution,ny=round(args.resolution*.8),sigma=args.sigma,
                           viscosity=args.viscosity,dt=args.dt,pinned_rim=args.pinned_rim)
    for _ in range(round(args.settle/args.dt)):
        system.step(args.newton)
    system.time=0.
    if args.stirrer:
        system.stirrer=Stirrer(args.scene,flow_coupled=args.surface_flow)
    if args.surface_flow:
        if not args.stirrer:raise ValueError("--surface-flow requires --stirrer")
        system.refine_liquid_contact()
        system.flow=SurfaceFlow(system)
        system.flow.initialize()
    config=dict(scene=args.scene,nx=system.nx,ny=system.ny,length=system.length,width=system.width,
        depth=system.depth,sigma=args.sigma,rho=system.rho,viscosity=system.viscosity,
        membrane_thickness=system.membrane_thickness,dt=args.dt,frame_dt=args.frame_dt,
        frames=args.frames,visual=visual,settle_seconds=args.settle,pinned_rim=args.pinned_rim,
        liquid_contact_refinement=2 if args.surface_flow else 1,
        liquid_contact_offset=system.contact_offset,
        solid_contact_margin=system.contact_margin if args.surface_flow else 0.,
        adaptive_flow_steps=args.surface_flow,
        surface_flow=args.surface_flow,flow_tracers=160 if args.surface_flow else 0,stirrer=system.stirrer.config() if system.stirrer else None,
        bodies=[dict(name=b.name,mass=b.mass,inertia=b.inertia.tolist(),
        scale=b.scale,centers=b.centers.tolist(),radii=b.radii.tolist()) for b in bodies],
        solver='implicit graph-membrane / 3D potential-fluid / six-DOF rigid contact',
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    np.savez_compressed(args.output/'mesh.npz',xy=system.xy,faces=system.faces)
    started=time.time();audit=[]
    steps=round(args.frame_dt/args.dt)
    assert abs(steps*args.dt-args.frame_dt)<1.e-10
    for frame in range(args.frames):
        substeps=[]
        for _ in range(steps):
            substeps.extend(advance(system,args.newton))
        maxima={key:max(row.get(key,0.) for row in substeps) for key in [
            'max_penetration','probe_solid_penetration','wire_pair_penetration',
            'continuous_probe_penetration','continuous_pair_penetration','flow_cfl']}
        np.savez_compressed(args.output/'states'/f'{frame:05d}.npz',**system.snapshot())
        audit.append(dict(frame=frame,**system.last,substep_maxima=maxima,
            accepted_substeps=len(substeps),minimum_dt=min(r['integration_dt'] for r in substeps)))
        if frame%10==0:
            print(json.dumps(dict(frame=frame,seconds=round(time.time()-started,2),
                z_mm=[round(b.position[2]*1000,3) for b in bodies],
                penetration_um=round(system.last['max_penetration']*1.e6,2))),flush=True)
            (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    (args.output/'complete.json').write_text(json.dumps(dict(frames=args.frames,seconds=time.time()-started),indent=2))
    print('Complete',args.output,flush=True)


if __name__=='__main__':main()
