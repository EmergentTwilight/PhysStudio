"""Reconstruct separate ice and water meshes from cached particle states."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from .thermal import LATENT


@wp.kernel(enable_backward=False)
def film_shape(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),grid:wp.uint64,
               dx:float,flatness:wp.array(dtype=float)):
    i=wp.tid();weight=float(0.);mean=float(0.);second=float(0.)
    if e[i]>=333500.:
        for j in wp.hash_grid_query(grid,x[i],3.*dx):
            if e[j]>=333500.:
                r=(x[j]-x[i])/dx
                w=wp.pow(wp.max(1.-wp.dot(r,r)/9.,0.),3.)
                weight+=w;mean+=w*r[2];second+=w*r[2]*r[2]
        variance=wp.max(second/wp.max(weight,1.e-6)-wp.pow(mean/wp.max(weight,1.e-6),2.),0.)
        # Flatten single-layer films; recover a volumetric kernel in deeper pools.
        t=wp.clamp((.12-variance)/.10,0.,1.)
        flatness[i]=t*t*(3.-2.*t)


@wp.kernel(enable_backward=False)
def deposit(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),ice:wp.array3d(dtype=float),water:wp.array3d(dtype=float),
            origin:wp.vec3,ds:float,dx:float,floor:float,wetting:int,flatness:wp.array(dtype=float)):
    p=wp.tid();point=x[p];b=wp.vec3i(int(wp.floor((point[0]-origin[0])/ds)),
        int(wp.floor((point[1]-origin[1])/ds)),int(wp.floor((point[2]-origin[2])/ds)))
    tangent=float(.65);normal=float(.65)
    if wetting==1 and e[p]>=333500.:
        height=wp.max(point[2]-floor-.46*dx,0.)
        weight=wp.pow(wp.max(1.-height/(2.*dx),0.),2.)*flatness[p]
        tangent+=.35*weight
        # Preserve each particle's Gaussian integral while resolving a thin wall film.
        normal=.65*.65*.65/(tangent*tangent)
    reach_xy=int(wp.ceil((2./.65)*tangent*dx/ds))
    reach_z=int(wp.ceil((2./.65)*normal*dx/ds))
    for i in range(-reach_xy,reach_xy+1):
        for j in range(-reach_xy,reach_xy+1):
            for k in range(-reach_z,reach_z+1):
                a=b+wp.vec3i(i,j,k)
                if a[0]>=0 and a[1]>=0 and a[2]>=0 and a[0]<ice.shape[0] and a[1]<ice.shape[1] and a[2]<ice.shape[2]:
                    q=origin+ds*wp.vec3(float(a[0]),float(a[1]),float(a[2]))-point
                    qn=wp.vec3(q[0]/(tangent*dx),q[1]/(tangent*dx),q[2]/(normal*dx))
                    if wp.dot(qn,qn)<(2./.65)*(2./.65):
                        w=wp.exp(-wp.dot(qn,qn)/2.)/(wp.pow(2.*3.141592653589793,1.5)*tangent*tangent*normal)
                        if e[p]<333500.:wp.atomic_add(ice,a[0],a[1],a[2],w)
                        else:wp.atomic_add(water,a[0],a[1],a[2],w*.917)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,nargs='*')
    p.add_argument('--stride',type=int,default=1);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resolution-scale',type=float)
    p.add_argument('--water-kernel',choices=['isotropic','wetting'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((args.input/'config.json').read_text());dx=cfg['dx']
    water_kernel=args.water_kernel or cfg.get('water_kernel','isotropic')
    scale=args.resolution_scale if args.resolution_scale is not None else (.4 if water_kernel=='wetting' else .6)
    if not np.isfinite(scale) or scale<=0.:p.error('Resolution scale must be finite and positive.')
    ds=dx*scale
    origin=np.array(cfg['bounds'][0])-dx*2.5;upper=np.array(cfg['bounds'][1])+dx*2.5
    # Only the occupied vertical range needs to be meshed.
    files=sorted((args.input/'states').glob('*.npz'))
    if len(files)!=cfg['frames']:raise ValueError('Simulation states are incomplete')
    indices=args.frames if args.frames is not None else list(range(0,len(files),args.stride))
    zmax=max(float(np.load(files[i])['positions'][:,2].max()) for i in indices)
    upper[2]=zmax+dx*3.
    shape=tuple(int(v) for v in np.ceil((upper-origin)/ds).astype(int)+1)
    wp.init();ice=wp.zeros(tuple(shape),dtype=float,device=args.device);water=wp.zeros_like(ice)
    grid=wp.HashGrid(128,128,128,device=args.device)
    mc=wp.MarchingCubes(*shape,max_verts=1500000,max_tris=3000000,device=args.device)
    records=[];start=time.time()
    for index in indices:
        state=np.load(files[index]);x=wp.array(state['positions'],dtype=wp.vec3,device=args.device)
        e=wp.array(state['enthalpy'],dtype=float,device=args.device)
        flatness=wp.zeros(len(x),dtype=float,device=args.device)
        if water_kernel=='wetting':
            grid.build(x,3.*dx)
            wp.launch(film_shape,len(x),inputs=[x,e,grid.id,dx,flatness],device=args.device)
        ice.zero_();water.zero_()
        wp.launch(deposit,len(x),inputs=[x,e,ice,water,wp.vec3(*origin),ds,dx,
            float(cfg['bounds'][0][2]),int(water_kernel=='wetting'),flatness],device=args.device)
        mesh={};counts={}
        for name,field in [('ice',ice),('water',water)]:
            mc.surface(field,.48)
            vertices=mc.verts.numpy()*ds+origin if mc.verts.shape[0] else np.empty((0,3),np.float32)
            # The density is positive inside; reverse Warp's low-to-high winding.
            faces=mc.indices.numpy().reshape(-1,3)[:,[0,2,1]] if mc.indices.shape[0] else np.empty((0,3),np.int32)
            mesh[name+'_vertices']=vertices.astype(np.float32);mesh[name+'_faces']=faces.astype(np.int32)
            counts[name]=len(faces)
        np.savez_compressed(args.output/f'{index:05d}.npz',**mesh)
        records.append(dict(frame=index,triangles=counts))
        if index%20==0:print(json.dumps(dict(frame=index,elapsed=time.time()-start,triangles=counts)),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(input=str(args.input.resolve()),frames=indices,spacing=ds,
        threshold=.48,kernel_sigma=.65*dx,water_kernel=water_kernel,
        film_model='local_vertical_variance_v2' if water_kernel=='wetting' else None,records=records),indent=2))


if __name__=='__main__':main()
