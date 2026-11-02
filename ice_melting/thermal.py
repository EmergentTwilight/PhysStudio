"""Specific enthalpy with latent heat and symmetric SPH heat exchange."""
import numpy as np
import warp as wp

LATENT=333500.0
CP_ICE=2100.0
CP_WATER=4180.0
RHO_ICE=917.0
RHO_WATER=1000.0


def temperature(enthalpy):
    e=np.asarray(enthalpy)
    return np.where(e<0,e/CP_ICE,np.where(e>LATENT,(e-LATENT)/CP_WATER,0.))


def fraction(enthalpy):
    return np.clip(np.asarray(enthalpy)/LATENT,0.,1.)


@wp.func
def temp(e:float):
    t=float(0.0)
    if e<0.0:t=e/2100.0
    elif e>333500.0:t=(e-333500.0)/4180.0
    return t


@wp.func
def liquid(e:float):
    return wp.clamp(e/333500.0,0.0,1.0)


@wp.func
def volume(e:float, m:float):
    f=liquid(e)
    return m*((1.0-f)/917.0+f/1000.0)


@wp.kernel(enable_backward=False)
def rate(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),grid:wp.uint64,
         h:float,norm:float,dx:float,m:float,air_t:float,air_h:float,plate_t:float,
         plate_h:float,radiant:float,spot:wp.vec3,spot_width:float,
         result:wp.array(dtype=float),external:wp.array(dtype=float),diagonal:wp.array(dtype=float)):
    i=wp.tid();p=x[i];ti=temp(e[i]);fi=liquid(e[i]);ki=2.2*(1.0-fi)+.6*fi
    vi=volume(e[i],m);heat=float(0.);diag=float(0.);occupancy=float(0.)
    normal=wp.vec3(0.)
    for j in wp.hash_grid_query(grid,p,h):
        r=p-x[j];r2=wp.dot(r,r)
        if r2<h*h:
            q=1.0-r2/(h*h);vj=volume(e[j],m)
            w=q*q*q/(norm*dx*dx*dx)
            occupancy+=vj*w
            grad=-6.0*q*q/(h*h*norm*dx*dx*dx)*r
            normal-=vj*grad
            if j!=i:
                fj=liquid(e[j]);kj=2.2*(1.0-fj)+.6*fj
                conduct=4.0*ki*kj/(ki+kj)*vi*vj*(-wp.dot(r,grad))/(r2+.01*h*h)
                heat+=conduct*(temp(e[j])-ti)
                diag+=conduct
    area=3.0*dx*dx*wp.clamp((1.0-occupancy)/.5,0.,1.)
    conv=air_h*area
    plate=plate_h*dx*dx*wp.clamp((1.3*dx-p[2])/(.9*dx),0.,1.)
    delta=p-spot
    illumination=wp.exp(-wp.dot(delta,delta)/(2.*spot_width*spot_width))
    incidence=float(.4)
    if wp.length(normal)>1.e-6 and wp.length(delta)>1.e-6:
        incidence=wp.max(wp.dot(wp.normalize(normal),-wp.normalize(delta)),0.)
    qext=conv*(air_t-ti)+plate*(plate_t-ti)+radiant*area*illumination*incidence
    external[i]=qext
    result[i]=(heat+qext)/m
    diagonal[i]=(diag+conv+plate)/(m*2100.0)


@wp.kernel(enable_backward=False)
def integrate(e:wp.array(dtype=float),rates:wp.array(dtype=float),external:wp.array(dtype=float),
              accumulated:wp.array(dtype=float),dt:float):
    i=wp.tid();e[i]+=dt*rates[i];accumulated[i]+=dt*external[i]
