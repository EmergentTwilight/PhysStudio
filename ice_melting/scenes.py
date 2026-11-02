"""Three thermal boundary conditions and volumetric ice geometries."""
from physstudio.paths import data_path
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
import trimesh
from .thermal import CP_ICE,LATENT,CP_WATER,RHO_ICE,RHO_WATER

PRESETS={
    'bunny':dict(label='Ice atelier',dx=.0016,air_t=45.,air_h=100.,plate_t=65.,plate_h=1400.,
        radiant=1800.,spot=[-.07,-.035,.065],spot_width=.085,thermal_scale=80.,
        wall_film_drag=80.,wall_film_depth=2.,water_kernel='wetting',
        seconds=4.,frames=240,bounds=[[-.105,-.080,0.],[.105,.080,.18]],target=[0.,0.,.032]),
    'arch':dict(label='Thawing arch',dx=.0018,air_t=38.,air_h=50.,plate_t=35.,plate_h=180.,
        radiant=30000.,spot=[0.,-.025,.100],spot_width=.025,thermal_scale=95.,
        seconds=4.,frames=240,bounds=[[-.115,-.075,0.],[.115,.075,.18]],target=[0.,0.,.034]),
    'cubes':dict(label='Warm-water drift',dx=.0020,air_t=32.,air_h=35.,plate_t=55.,plate_h=1200.,
        radiant=0.,spot=[0.,0.,.05],spot_width=.1,thermal_scale=90.,
        seconds=4.,frames=240,bounds=[[-.090,-.062,0.],[.090,.062,.15]],target=[0.,0.,.026]),
}


def lattice(lo,hi,dx):
    return np.stack(np.meshgrid(*[np.arange(a+dx*.5,b,dx) for a,b in zip(lo,hi)],indexing='ij'),-1).reshape(-1,3)


def bunny(dx):
    path=data_path('ice_melting','bunny.ply')
    mesh=trimesh.load(path,process=False)
    # The Stanford scan has five open boundary loops. Cap them before filling;
    # otherwise a finer voxelization can silently become a hollow ice shell.
    edges=mesh.edges
    _,first,counts=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
    boundary=edges[first[counts==1]]
    if len(boundary):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        degree=np.bincount(boundary.ravel(),minlength=len(mesh.vertices))
        if np.any((degree!=0)&(degree!=2)):
            raise ValueError('Bunny boundary is not a collection of simple loops')
        graph=coo_matrix((np.ones(len(boundary)),(boundary[:,0],boundary[:,1])),
                         shape=(len(mesh.vertices),len(mesh.vertices)))
        _,labels=connected_components(graph,directed=False)
        vertices=mesh.vertices.tolist();faces=mesh.faces.tolist()
        for label in np.unique(labels[boundary[:,0]]):
            loop=boundary[labels[boundary[:,0]]==label]
            center=mesh.vertices[np.unique(loop)].mean(0)
            index=len(vertices);vertices.append(center.tolist())
            faces.extend([[int(b),int(a),index] for a,b in loop])
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    if not mesh.is_watertight:
        raise ValueError('Bunny must enclose a solid volume before voxelization')
    v=mesh.vertices.copy();v=v[:,[0,2,1]];v[:,1]*=-1
    v-=np.array([(v[:,0].max()+v[:,0].min())/2,(v[:,1].max()+v[:,1].min())/2,v[:,2].min()])
    v*=.095/v[:,2].max();v[:,2]+=dx*.5
    mesh.vertices=v
    vox=mesh.voxelized(dx)
    a=np.pad(vox.matrix,3)
    a=ndimage.binary_fill_holes(a)
    if a.sum()<.85*abs(mesh.volume)/dx**3:
        raise ValueError('Voxelization failed to fill the bunny interior')
    p=vox.indices_to_points(np.argwhere(a)-3)
    p[:,2]+=dx*.5-p[:,2].min()
    return p


def build(name,dx=None):
    cfg=dict(PRESETS[name]);cfg['scene']=name
    if dx is not None:cfg['dx']=dx
    dx=cfg['dx'];lo,hi=np.array(cfg['bounds'])
    if name=='bunny':
        ice=bunny(dx);groups=np.zeros(len(ice),np.int32)
    elif name=='arch':
        p=lattice([-.065,-.020,0.],[.065,.020,.112],dx)
        radial=np.sqrt(p[:,0]**2+(p[:,2]-.037)**2)
        taper=np.exp(-(p[:,0]/.020)**2)
        arc=(radial<(.059-.002*taper))&(radial>(.039+.007*taper))&(p[:,2]>=.037)
        legs=(np.abs(p[:,0])>.039)&(np.abs(p[:,0])<.059)&(p[:,2]<.039)
        width=.013+.003*np.cos(p[:,2]*32.)
        ice=p[(arc|legs)&(np.abs(p[:,1])<width)]
        groups=np.zeros(len(ice),np.int32)
    else:
        parts=[];ids=[]
        for index,(center,angles,size) in enumerate([
            ((-.037,-.016,.040),(8,14,-15),.029),((.002,.014,.044),(-8,-12,18),.033),
            ((.038,-.013,.037),(5,-15,30),.026)]):
            p=lattice([-size/2]*3,[size/2]*3,dx)
            # A rounded-box mask retains broad ice facets and softened edges.
            q=np.maximum(np.abs(p)-(size/2-.004),0.)
            p=p[np.linalg.norm(q,axis=1)<.004]
            p=p@Rotation.from_euler('xyz',angles,degrees=True).as_matrix().T+center
            parts.append(p);ids.extend([index]*len(p))
        ice=np.concatenate(parts);groups=np.array(ids,np.int32)
    e=np.full(len(ice),-7.*CP_ICE,np.float32)
    if name=='cubes':
        water_dx=dx*(RHO_ICE/RHO_WATER)**(1/3)
        water=lattice(lo+np.array([dx,dx,0.]),[hi[0]-dx,hi[1]-dx,.024],water_dx)
        from scipy.spatial import cKDTree
        water=water[cKDTree(ice).query(water)[0]>dx*1.15]
        x=np.vstack([ice,water]);groups=np.r_[groups,np.full(len(water),-1,np.int32)]
        e=np.r_[e,np.full(len(water),LATENT+CP_WATER*42.,np.float32)]
    else:x=ice
    cfg.update(particles=len(x),initial_ice_particles=len(ice),mass=RHO_ICE*dx**3,
        fluid_model='PBF with symmetric volume-weighted pressure and particle rigid shape matching',
        thermal_model='specific enthalpy with latent heat and symmetric SPH conduction')
    return x.astype(np.float32),groups,e,cfg
