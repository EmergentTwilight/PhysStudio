"""Millimeter-scale rigid assets and physical scene presets."""
import numpy as np
from scipy.spatial.transform import Rotation
from .solver import Body


def paperclip_path():
    pieces=[]
    def line(a,b):
        count=max(2,int(np.linalg.norm(np.asarray(a)-b)/.0002)+1)
        pieces.append(np.linspace(a,b,count))
    def arc(center,r,start,end):
        theta=np.linspace(start,end,max(8,int(abs(end-start)*r/.0002)+1))
        pieces.append(np.asarray(center)+r*np.column_stack([np.cos(theta),np.sin(theta)]))
    line([-.0016,-.00035],[.0018,-.00035])
    arc([.0018,.0002],.00055,-np.pi/2,np.pi/2)
    line([.0018,.00075],[-.0025,.00075])
    arc([-.0025,-.00025],.001,np.pi/2,3*np.pi/2)
    line([-.0025,-.00125],[.0024,-.00125])
    arc([.0024,.0002],.00145,-np.pi/2,np.pi/2)
    line([.0024,.00165],[-.002,.00165])
    points=np.concatenate(pieces)
    points=np.column_stack([points,np.zeros(len(points))])
    points-=points.mean(axis=0)
    return points


def make_scene(name):
    bodies=[];visual=[]
    if name=='paperclips':
        path=paperclip_path();wire=.00032
        length=np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
        for i,(xy,density,angle) in enumerate([
            ((-.010,-.004),2700,-.45),((0.,.002),5400,.25),((.010,-.002),7900,-.15)]):
            mass=np.pi*wire**2*length*density
            inertia=mass*np.mean(np.sum(path*path,axis=1)[:,None]-path*path,axis=0)+mass*wire**2*.5
            body=Body(f'clip_{i}',np.array([*xy,wire+.0004+i*.00025]),
                Rotation.from_euler('z',angle).as_matrix(),mass,inertia,.003,
                path,np.full_like(path,wire),.003)
            bodies.append(body)
            visual.append(dict(kind='paperclip',path=path.tolist(),radius=wire,density=density,
                color=[[.45,.59,.64],[.53,.32,.11],[.32,.39,.44]][i]))
    elif name=='capillary_cluster':
        for i in range(7):
            theta=2*np.pi*(i-1)/6
            xy=np.array([np.cos(theta),np.sin(theta)])*(.0082 if i else 0)
            axes=np.array([.00185,.00185,.0008])
            mass=4/3*np.pi*np.prod(axes)*2200
            inertia=mass/5*(np.sum(axes*axes)-axes*axes)
            bodies.append(Body(f'pin_{i}',np.array([*xy,.0013]),np.eye(3),mass,inertia,.00185,
                               np.zeros((1,3)),axes[None,:],.00187,
                               velocity=np.array([-xy[1],xy[0],0])*.15))
            visual.append(dict(kind='pin',axes=axes.tolist(),color=[
                [.035,.28,.32],[.7,.14,.065],[.65,.4,.09],[.05,.31,.24],
                [.26,.13,.35],[.55,.07,.15],[.1,.21,.40]][i]))
    elif name=='orchard':
        for i,(xy,radius,tilt) in enumerate([
            ((-.007,-.003),.0026,12),((.001,.003),.0029,-8),((.009,-.003),.0024,10)]):
            axes=np.array([radius,radius*.94,radius*1.13])
            mass=4/3*np.pi*np.prod(axes)*1250
            inertia=mass/5*(np.sum(axes*axes)-axes*axes)
            R=Rotation.from_euler('xyz',[tilt,0,i*60],degrees=True).as_matrix()
            bodies.append(Body(f'apple_{i}',np.array([*xy,axes[2]+.0008+i*.0005]),R,
                mass,inertia,radius,np.zeros((1,3)),axes[None,:],radius*.94))
            visual.append(dict(kind='apple',axes=axes.tolist(),asset='food_apple_01'))
    else:
        raise ValueError(name)
    return bodies,visual


DESCRIPTIONS={
    'paperclips': ('Held by a surface', 'Three metal densities, one water membrane'),
    'capillary_cluster': ('An invisible attraction', 'Rigid bodies gather along shared menisci'),
    'orchard': ('A miniature orchard', 'Scanned apples meet capillary waves'),
}
