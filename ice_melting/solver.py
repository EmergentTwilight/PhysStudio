"""Warp particle dynamics with thermal phase change and rigid ice fragments."""
import math
import numpy as np
import warp as wp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from . import thermal

wp.set_module_options({'enable_backward':False})


@wp.func
def kernel(r:wp.vec3,h:float):
    q=wp.max(0.,1.-wp.dot(r,r)/(h*h));return q*q*q


@wp.func
def grad(r:wp.vec3,h:float,norm:float):
    length=wp.length(r);out=wp.vec3(0.)
    if length>1.e-8 and length<h:
        q=1.-length/h;out=-9.142857142857*q*q/(h*norm*length)*r
    return out


@wp.func
def solid(e:float,g:int):
    return e<333500. and g>=0


@wp.kernel
def predict(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),old:wp.array(dtype=wp.vec3),dt:float):
    i=wp.tid();old[i]=x[i];v[i]+=wp.vec3(0.,0.,-9.81)*dt;x[i]+=dt*v[i]


@wp.kernel
def density(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),groups:wp.array(dtype=int),
            grid:wp.uint64,h:float,norm:float,lam:wp.array(dtype=float),rho:wp.array(dtype=float)):
    i=wp.tid();p=x[i];d=float(0.);gsum=wp.vec3(0.);g2=float(0.)
    for j in wp.hash_grid_query(grid,p,h):
        volume=1.-.083*thermal.liquid(e[j])
        r=p-x[j];d+=volume*kernel(r,h)/norm
        g=volume*grad(r,h,norm);gsum+=g
        if j!=i:g2+=wp.dot(g,g)
    rho[i]=d;lam[i]=0.
    if not solid(e[i],groups[i]):
        lam[i]=-wp.max(d-1.,0.)/(g2+wp.dot(gsum,gsum)+.02/(h*h))


@wp.kernel
def corrections(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),groups:wp.array(dtype=int),
                grid:wp.uint64,h:float,norm:float,dx:float,lam:wp.array(dtype=float),
                delta:wp.array(dtype=wp.vec3),limited:wp.array(dtype=int)):
    i=wp.tid();p=x[i];d=wp.vec3(0.);ice=solid(e[i],groups[i]);vi=1.-.083*thermal.liquid(e[i])
    for j in wp.hash_grid_query(grid,p,h):
        if j!=i:
            r=p-x[j];length=wp.length(r);icej=solid(e[j],groups[j])
            vj=1.-.083*thermal.liquid(e[j])
            d+=(lam[i]*vj+lam[j]*vi)*grad(r,h,norm)
            contact=(ice or icej) and not (ice and icej and groups[i]==groups[j])
            if contact and length>1.e-8 and length<dx*.90:
                d+=.5*(dx*.90-length)*r/length
    if wp.length(d)>dx*.35:
        d*=dx*.35/wp.length(d);wp.atomic_add(limited,0,1)
    delta[i]=d


@wp.kernel
def apply(x:wp.array(dtype=wp.vec3),d:wp.array(dtype=wp.vec3),lo:wp.vec3,hi:wp.vec3,radius:float):
    i=wp.tid();p=x[i]+d[i]
    x[i]=wp.vec3(wp.clamp(p[0],lo[0]+radius,hi[0]-radius),
        wp.clamp(p[1],lo[1]+radius,hi[1]-radius),wp.clamp(p[2],lo[2]+radius,hi[2]-radius))


@wp.func
def as_double(v:wp.vec3):
    return wp.vec3d(wp.float64(v[0]),wp.float64(v[1]),wp.float64(v[2]))


@wp.func
def as_float(v:wp.vec3d):
    return wp.vec3(wp.float32(v[0]),wp.float32(v[1]),wp.float32(v[2]))


@wp.kernel
def sum_centers(x:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),
                groups:wp.array(dtype=int),s:wp.array(dtype=wp.vec3d),r:wp.array(dtype=wp.vec3d),count:wp.array(dtype=wp.float64)):
    i=wp.tid();g=groups[i]
    if solid(e[i],g):
        wp.atomic_add(s,g,as_double(x[i]));wp.atomic_add(r,g,as_double(rest[i]));wp.atomic_add(count,g,wp.float64(1.))


@wp.kernel
def covariance(x:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),
                groups:wp.array(dtype=int),s:wp.array(dtype=wp.vec3d),r:wp.array(dtype=wp.vec3d),
                count:wp.array(dtype=wp.float64),a:wp.array(dtype=wp.mat33d)):
    i=wp.tid();g=groups[i]
    if solid(e[i],g):
        wp.atomic_add(a,g,wp.outer(as_double(x[i])-s[g]/count[g],as_double(rest[i])-r[g]/count[g]))


@wp.kernel
def rotations(a:wp.array(dtype=wp.mat33d),count:wp.array(dtype=wp.float64),rotation:wp.array(dtype=wp.mat33d)):
    g=wp.tid();R=wp.identity(n=3,dtype=wp.float64)
    if count[g]>2.:
        u=wp.mat33d(wp.float64(0.));v=wp.mat33d(wp.float64(0.));sigma=wp.vec3d(wp.float64(0.))
        wp.svd3(a[g],u,sigma,v)
        sign=wp.float64(1.)
        if wp.determinant(u*wp.transpose(v))<0.:sign=wp.float64(-1.)
        R=u*wp.diag(wp.vec3d(wp.float64(1.),wp.float64(1.),sign))*wp.transpose(v)
    rotation[g]=R


@wp.kernel
def match(x:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),
          groups:wp.array(dtype=int),s:wp.array(dtype=wp.vec3d),r:wp.array(dtype=wp.vec3d),
          count:wp.array(dtype=wp.float64),rotation:wp.array(dtype=wp.mat33d)):
    i=wp.tid();g=groups[i]
    if solid(e[i],g):x[i]=as_float(s[g]/count[g]+rotation[g]*(as_double(rest[i])-r[g]/count[g]))


@wp.kernel
def group_bounds(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),groups:wp.array(dtype=int),
                 minimum:wp.array(dtype=wp.vec3),maximum:wp.array(dtype=wp.vec3)):
    i=wp.tid();g=groups[i]
    if solid(e[i],g):
        wp.atomic_min(minimum,g,x[i]);wp.atomic_max(maximum,g,x[i])


@wp.kernel
def group_contact(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),groups:wp.array(dtype=int),
                  minimum:wp.array(dtype=wp.vec3),maximum:wp.array(dtype=wp.vec3),lo:wp.vec3,hi:wp.vec3,radius:float):
    i=wp.tid();g=groups[i]
    if solid(e[i],g):
        a=lo+wp.vec3(radius)-minimum[g];b=hi-wp.vec3(radius)-maximum[g]
        x[i]+=wp.vec3(wp.max(a[0],0.)+wp.min(b[0],0.),wp.max(a[1],0.)+wp.min(b[1],0.),
                      wp.max(a[2],0.)+wp.min(b[2],0.))


@wp.kernel
def velocities(x:wp.array(dtype=wp.vec3),old:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),
               dt:float,lo:wp.vec3,hi:wp.vec3,radius:float):
    i=wp.tid();vel=(x[i]-old[i])/dt
    # Coulomb-like floor friction acts only on contacting particles.
    if x[i][2]<lo[2]+radius+1.e-6:
        tangent=wp.vec3(vel[0],vel[1],0.);speed=wp.length(tangent)
        if speed>1.e-8:vel-=tangent*wp.min(.12*9.81*dt/speed,1.)
    v[i]=vel


@wp.kernel
def wall_film_viscosity(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),
                        groups:wp.array(dtype=int),floor:float,radius:float,depth:float,drag:float,dt:float):
    i=wp.tid()
    if not solid(e[i],groups[i]):
        distance=wp.max(x[i][2]-floor-radius,0.)
        weight=wp.pow(wp.max(1.-distance/depth,0.),2.)
        # Backward-Euler viscous relaxation against the stationary wetting wall.
        # The support is local to the unresolved film, and the update dissipates energy.
        v[i]/=1.+drag*dt*weight


@wp.func
def cohesion(r:float,h:float):
    value=float(0.)
    if r>0. and r<h:
        q=r/h;value=32./3.141592653589793/(h*h*h)*wp.pow(1.-q,3.)*wp.pow(q,3.)
        if q<.5:value=2.*value-32./3.141592653589793/(h*h*h)/64.
    return value


@wp.kernel
def normals(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),groups:wp.array(dtype=int),
            grid:wp.uint64,h:float,norm:float,out:wp.array(dtype=wp.vec3)):
    i=wp.tid();n=wp.vec3(0.)
    if not solid(e[i],groups[i]):
        for j in wp.hash_grid_query(grid,x[i],h):
            if not solid(e[j],groups[j]):n+=.917*h*grad(x[i]-x[j],h,norm)
    out[i]=n


@wp.kernel
def surface_forces(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),out:wp.array(dtype=wp.vec3),
                   e:wp.array(dtype=float),groups:wp.array(dtype=int),n:wp.array(dtype=wp.vec3),rho:wp.array(dtype=float),
                   grid:wp.uint64,h:float,norm:float,mass:float,dt:float,gamma:float,beta:float,
                   ice_drag:float,viscosity:float):
    i=wp.tid();ice=solid(e[i],groups[i]);a=wp.vec3(0.);dv=wp.vec3(0.)
    for j in wp.hash_grid_query(grid,x[i],h):
        if j!=i:
            r=x[i]-x[j];l=wp.length(r);icej=solid(e[j],groups[j])
            if l>1.e-8 and l<h:
                if not ice and not icej:
                    correction=2./wp.max(rho[i]+rho[j],.4)
                    a-=gamma*correction*(mass*cohesion(l,h)*r/l+(n[i]-n[j]))
                    dv+=(v[j]-v[i])*kernel(r,h)/norm
                elif ice!=icej:
                    # Equal/opposite short-range wetting force on ice and water.
                    q=l/h
                    if q>.5:
                        adhesion=.007/wp.pow(h,3.25)*wp.pow(wp.max(-4.*l*l/h+6.*l-2.*h,0.),.25)
                        a-=beta*mass*adhesion*r/l
                    dv+=ice_drag*(v[j]-v[i])*kernel(r,h)/norm
    out[i]=v[i]+dt*a+wp.min(viscosity*dt,.10)*dv


class IceSystem:
    def __init__(self,positions,groups,enthalpy,cfg,device='cuda:0'):
        wp.init();self.device=device;self.cfg=cfg;self.n=len(positions);self.dx=cfg['dx'];self.h=2.3*self.dx
        self.mass=cfg['mass'];self.time=0.;self.steps=0;self.group_count=int(groups.max())+1
        self.initial_ice=np.flatnonzero(groups>=0)
        pair=cKDTree(positions[self.initial_ice]).query_pairs(1.45*self.dx,output_type='ndarray')
        self.edges=self.initial_ice[pair]
        self.edges=self.edges[groups[self.edges[:,0]]==groups[self.edges[:,1]]]
        self.last_melt=np.zeros(len(self.initial_ice),bool)
        lattice=np.stack(np.meshgrid(*[np.arange(-3,4)*self.dx]*3,indexing='ij'),-1).reshape(-1,3)
        self.norm=float(np.sum(np.maximum(1.-(lattice*lattice).sum(-1)/self.h**2,0.)**3))
        self.x=wp.array(positions,dtype=wp.vec3,device=device);self.old=wp.clone(self.x);self.rest=wp.clone(self.x)
        self.v=wp.zeros_like(self.x);self.vout=wp.zeros_like(self.x);self.delta=wp.zeros_like(self.x);self.normal=wp.zeros_like(self.x)
        self.e=wp.array(enthalpy,dtype=float,device=device);self.groups=wp.array(groups,dtype=int,device=device)
        self.lam=wp.zeros(self.n,dtype=float,device=device);self.rho=wp.zeros_like(self.lam)
        self.heat_rate=wp.zeros_like(self.lam);self.external=wp.zeros_like(self.lam);self.heat_external=wp.zeros_like(self.lam)
        self.thermal_diag=wp.zeros_like(self.lam);self.limited=wp.zeros(1,dtype=int,device=device)
        self.capacity=max(len(self.initial_ice),1)
        self.sum=wp.zeros(self.capacity,dtype=wp.vec3d,device=device);self.sumrest=wp.zeros_like(self.sum)
        self.count=wp.zeros(self.capacity,dtype=wp.float64,device=device)
        self.matrix=wp.zeros(self.capacity,dtype=wp.mat33d,device=device);self.rotation=wp.zeros_like(self.matrix)
        self.minimum=wp.zeros(self.capacity,dtype=wp.vec3,device=device);self.maximum=wp.zeros_like(self.minimum)
        self.grid=wp.HashGrid(128,128,128,device=device)
        self.lo,self.hi=[wp.vec3(*p) for p in cfg['bounds']]
        self.initial_energy=float(np.sum(enthalpy.astype(np.float64))*self.mass)
        self.max_thermal_number=0.

    def launch(self,k,n,args):wp.launch(k,n,inputs=args,device=self.device)

    def update_components(self):
        e=self.e.numpy();gone=e[self.initial_ice]>=thermal.LATENT
        if np.array_equal(gone,self.last_melt):return
        active=np.zeros(self.n,bool);active[self.initial_ice[~gone]]=True
        edges=self.edges[active[self.edges[:,0]]&active[self.edges[:,1]]]
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(self.n,self.n)).tocsr()
        _,labels=connected_components(graph,directed=False)
        _,relabel=np.unique(labels[active],return_inverse=True)
        groups=np.full(self.n,-1,np.int32);groups[active]=relabel
        self.group_count=int(relabel.max())+1 if len(relabel) else 0
        self.groups.assign(groups);self.last_melt=gone

    def shape_match(self):
        if not self.group_count:return
        self.sum.zero_();self.sumrest.zero_();self.count.zero_();self.matrix.zero_()
        self.launch(sum_centers,self.n,[self.x,self.rest,self.e,self.groups,self.sum,self.sumrest,self.count])
        self.launch(covariance,self.n,[self.x,self.rest,self.e,self.groups,self.sum,self.sumrest,self.count,self.matrix])
        self.launch(rotations,self.group_count,[self.matrix,self.count,self.rotation])
        self.launch(match,self.n,[self.x,self.rest,self.e,self.groups,self.sum,self.sumrest,self.count,self.rotation])
        self.minimum.fill_(wp.vec3(1.e6));self.maximum.fill_(wp.vec3(-1.e6))
        self.launch(group_bounds,self.n,[self.x,self.e,self.groups,self.minimum,self.maximum])
        self.launch(group_contact,self.n,[self.x,self.e,self.groups,self.minimum,self.maximum,self.lo,self.hi,self.dx*.46])

    def heat(self,dt):
        cfg=self.cfg;total=dt*cfg['thermal_scale']
        # Explicit conduction and Robin heating have a conservative substep bound.
        max_rate=18.*2.2/(thermal.RHO_ICE*thermal.CP_ICE*self.dx**2)+(
            3.*cfg['air_h']+cfg['plate_h'])/(thermal.RHO_ICE*thermal.CP_ICE*self.dx)
        steps=max(1,math.ceil(total*max_rate/.28));hdt=total/steps
        for _ in range(steps):
            self.launch(thermal.rate,self.n,[self.x,self.e,self.grid.id,self.h,self.norm,self.dx,self.mass,
                cfg['air_t'],cfg['air_h'],cfg['plate_t'],cfg['plate_h'],cfg['radiant'],
                wp.vec3(*cfg['spot']),cfg['spot_width'],self.heat_rate,self.external,self.thermal_diag])
            self.launch(thermal.integrate,self.n,[self.e,self.heat_rate,self.external,self.heat_external,hdt])
        if self.steps%30==0:
            self.max_thermal_number=max(self.max_thermal_number,float(self.thermal_diag.numpy().max()*hdt))
            if self.max_thermal_number>.55:raise RuntimeError('Thermal timestep exceeds the monotonicity bound')

    def step(self,dt=1/360,iterations=7,heating=True):
        self.grid.build(self.x,self.h)
        if heating:self.heat(dt)
        self.launch(density,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.lam,self.rho])
        self.launch(normals,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.normal])
        self.launch(surface_forces,self.n,[self.x,self.v,self.vout,self.e,self.groups,self.normal,self.rho,
            self.grid.id,self.h,self.norm,self.mass,dt,self.cfg.get('surface_tension',.025),
            self.cfg.get('adhesion',.035),self.cfg.get('ice_drag',.25),self.cfg.get('viscosity',12.)])
        self.v,self.vout=self.vout,self.v
        self.launch(predict,self.n,[self.x,self.v,self.old,dt])
        for _ in range(iterations):
            self.grid.build(self.x,self.h)
            self.launch(density,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.lam,self.rho])
            self.launch(corrections,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.dx,self.lam,self.delta,self.limited])
            self.launch(apply,self.n,[self.x,self.delta,self.lo,self.hi,self.dx*.46])
            self.shape_match()
        self.launch(velocities,self.n,[self.x,self.old,self.v,dt,self.lo,self.hi,self.dx*.46])
        if self.cfg.get('wall_film_drag',0.)>0.:
            self.launch(wall_film_viscosity,self.n,[self.x,self.v,self.e,self.groups,self.lo[2],
                self.dx*.46,self.dx*self.cfg.get('wall_film_depth',2.),self.cfg['wall_film_drag'],dt])
        self.time+=dt;self.steps+=1

    def snapshot(self):
        return dict(positions=self.x.numpy(),velocities=self.v.numpy(),enthalpy=self.e.numpy(),
                    groups=self.groups.numpy(),density=self.rho.numpy(),time=np.array(self.time))

    def audit(self):
        s=self.snapshot();x=s['positions'];e=s['enthalpy'];v=s['velocities']
        for key,value in s.items():
            if not np.isfinite(value).all():raise RuntimeError(f'Nonfinite {key}')
        fluid=e>=thermal.LATENT
        energy=float(e.astype(np.float64).sum()*self.mass)
        external=float(self.heat_external.numpy().astype(np.float64).sum())
        counts=np.bincount(s['groups'][~fluid]) if np.any(~fluid) else np.zeros(0)
        maxspeed=float(np.linalg.norm(v,axis=1).max())
        if maxspeed>8.:raise RuntimeError(f'Unresolved fast motion: {maxspeed}')
        return dict(time=self.time,particles=self.n,mass_kg=self.n*self.mass,energy_J=energy,
            external_heat_J=external,energy_balance_J=energy-self.initial_energy-external,
            liquid_fraction=float(thermal.fraction(e).mean()),mobile_liquid_particles=int(fluid.sum()),
            solid_components=int(np.count_nonzero(counts)),largest_component=int(counts.max()) if len(counts) else 0,
            min_temperature=float(thermal.temperature(e).min()),max_temperature=float(thermal.temperature(e).max()),
            max_speed=maxspeed,ground_gap=float(x[:,2].min()-self.dx*.46),
            thermal_number=self.max_thermal_number,limited_density_corrections=int(self.limited.numpy()[0]),
            liquid_density_p95=float(np.quantile(s['density'][fluid],.95)) if fluid.any() else 0.)
