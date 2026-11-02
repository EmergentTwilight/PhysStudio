"""Simulate ice melting and cache thermomechanical states."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from .scenes import build,PRESETS
from .solver import IceSystem


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=list(PRESETS),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=240);p.add_argument('--seconds',type=float,default=4.)
    p.add_argument('--dx',type=float);p.add_argument('--dt',type=float,default=1/1440)
    p.add_argument('--iterations',type=int,default=7);p.add_argument('--thermal-scale',type=float)
    p.add_argument('--device',default='cuda:0');p.add_argument('--no-heating',action='store_true')
    p.add_argument('--settings',type=Path,help='JSON overrides for wetting and surface reconstruction')
    args=p.parse_args()
    if args.frames<2 or args.seconds<=0:p.error('Use at least two frames and positive seconds.')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'states').mkdir()
    x,g,e,cfg=build(args.scene,args.dx)
    if args.settings:
        overrides=json.loads(args.settings.read_text())
        allowed={'surface_tension','adhesion','ice_drag','viscosity','wall_film_drag','wall_film_depth',
                 'water_kernel','camera_frames'}
        if set(overrides)-allowed:raise ValueError(f'Unknown settings: {set(overrides)-allowed}')
        for key,value in overrides.items():
            if key=='water_kernel':
                if value not in ['isotropic','wetting']:raise ValueError('Unknown water kernel')
            elif not isinstance(value,(int,float)) or not np.isfinite(value) or value<0.:
                raise ValueError(f'{key} must be finite and nonnegative')
        if overrides.get('wall_film_depth',2.)<=0.:raise ValueError('Film depth must be positive')
        if overrides.get('camera_frames',240)<=1:raise ValueError('Camera timeline needs at least two frames')
        cfg.update(overrides)
    if args.thermal_scale is not None:cfg['thermal_scale']=args.thermal_scale
    steps=int(np.ceil((args.seconds/(args.frames-1))/args.dt));dt=args.seconds/(args.frames-1)/steps
    cfg.update(frames=args.frames,seconds=args.seconds,dt=dt,substeps=steps,iterations=args.iterations,
        playback_fps=30,heating=not args.no_heating,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2))
    np.savez_compressed(args.output/'initial.npz',positions=x,enthalpy=e,groups=g)
    system=IceSystem(x,g,e,cfg,args.device)
    print(json.dumps(dict(scene=args.scene,particles=len(x),ice=cfg['initial_ice_particles'],dt=dt,steps=steps)),flush=True)
    audit=[];started=time.time()
    for frame in range(args.frames):
        if frame:
            for _ in range(steps):system.step(dt,args.iterations,not args.no_heating)
        system.update_components()
        record=system.audit();record.update(frame=frame,elapsed=time.time()-started)
        np.savez_compressed(args.output/'states'/f'{frame:05d}.npz',**system.snapshot())
        audit.append(record)
        if frame%10==0 or frame==args.frames-1:
            (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
            print(json.dumps(record),flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(frames=args.frames,seconds=time.time()-started),indent=2))


if __name__=='__main__':main()
