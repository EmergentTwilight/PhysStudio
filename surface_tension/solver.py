"""Implicit capillary height-membrane and six-DOF rigid contact.

The fluid inertia is the Schur complement of the finite-depth 3D potential
flow problem. A DCT diagonalizes its no-flux side-wall boundary conditions.
The surface remains a graph: there is no breakup or overhanging interface.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.transform import Rotation
from .wire_contact import rigid_probe_contacts, rigid_pair_contacts


@dataclass
class Body:
    name: str
    position: np.ndarray
    rotation: np.ndarray
    mass: float
    inertia: np.ndarray
    scale: float
    centers: np.ndarray
    radii: np.ndarray
    collision_radius: float
    velocity: np.ndarray = None
    omega: np.ndarray = None

    def __post_init__(self):
        self.velocity = np.zeros(3) if self.velocity is None else np.asarray(self.velocity, float)
        self.omega = np.zeros(3) if self.omega is None else np.asarray(self.omega, float)


def wire_pair_contacts(first, q1, R1, second, q2, R2):
    """Separated contact patches for two sphere-union wires."""
    offset1=first.centers@R1.T;offset2=second.centers@R2.T
    centers1=q1+offset1;centers2=q2+offset2
    radius1=first.radii[0,0];radius2=second.radii[0,0]
    bound1=np.linalg.norm(first.centers,axis=1).max()+radius1
    bound2=np.linalg.norm(second.centers,axis=1).max()+radius2
    if np.linalg.norm(q1-q2)>bound1+bound2:
        return []
    delta=centers1[:,None,:]-centers2[None,:,:]
    distance=np.linalg.norm(delta,axis=2)
    gap=distance-radius1-radius2
    active=np.argwhere(gap<0)
    if not len(active):return []
    order=np.argsort(gap[active[:,0],active[:,1]])
    chosen=[];contacts=[]
    for index in order:
        i,j=active[index]
        if any(np.linalg.norm(centers1[i]-centers1[k])<radius1*1.8 and
               np.linalg.norm(centers2[j]-centers2[l])<radius2*1.8 for k,l in chosen):
            continue
        normal=delta[i,j]/max(distance[i,j],1.e-12)
        jac1=np.r_[normal,np.cross(offset1[i],normal)/first.scale]
        jac2=-np.r_[normal,np.cross(offset2[j],normal)/second.scale]
        contacts.append((gap[i,j],jac1,jac2));chosen.append((i,j))
        if len(chosen)>=12:break
    return contacts


class CapillarySystem:
    def __init__(self, bodies, nx=80, ny=64, length=.040, width=.032,
                 depth=.012, sigma=.072, rho=1000., viscosity=.003,
                 dt=1/240, contact_stiffness=8.e6, stirrer=None, pinned_rim=False):
        self.bodies = bodies
        self.stirrer = stirrer
        self.flow = None
        self.contact_margin = .00006
        self.pinned_rim = pinned_rim
        self.nx, self.ny = nx, ny
        self.length, self.width, self.depth = length, width, depth
        self.dx, self.dy = length/nx, width/ny
        self.area = self.dx*self.dy
        self.sigma, self.rho, self.viscosity, self.dt = sigma, rho, viscosity, dt
        self.contact_stiffness = contact_stiffness * self.area
        xx, yy = np.meshgrid((np.arange(nx)+.5)*self.dx-length/2,
                             (np.arange(ny)+.5)*self.dy-width/2, indexing='ij')
        self.xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        self.n = nx*ny
        self.contact_xy = self.xy
        self.contact_vertices = np.arange(self.n)[:,None]
        self.contact_weights = np.ones((self.n,1))
        self.contact_interpolation = sparse.eye(self.n,format='csr')
        self.contact_offset = 0.
        self.free_surface=np.ones((nx,ny),dtype=bool)
        if pinned_rim:
            self.free_surface[[0,-1],:]=False
            self.free_surface[:,[0,-1]]=False
        self.free_surface=self.free_surface.ravel()
        self.size = self.n+6*len(bodies)
        self.h = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self.time = 0.
        self.gravity = 9.81
        self.membrane_thickness = .5*min(self.dx,self.dy)
        kx = 2*np.sin(np.pi*np.arange(nx)/(2*nx))/self.dx
        ky = 2*np.sin(np.pi*np.arange(ny)/(2*ny))/self.dy
        self.k = np.sqrt(kx[:,None]**2+ky[None,:]**2)
        safe = np.maximum(self.k,1.)
        self.fluid_mass = rho*self.area/(safe*np.tanh(safe*depth))
        # The closed basin has zero mean surface displacement. Its zero mode
        # is constrained exactly, rather than represented by a large mass.
        self.fluid_mass[0,0] = 0.
        self.mass_symbol = self.fluid_mass + rho*self.area*self.membrane_thickness
        # Two right triangles per quadrilateral, with exact graph-area energy.
        ids=np.arange(self.n).reshape(nx,ny)
        a,b,c,d=ids[:-1,:-1].ravel(),ids[1:,:-1].ravel(),ids[:-1,1:].ravel(),ids[1:,1:].ravel()
        self.faces=np.concatenate([np.stack([a,b,c],1),np.stack([d,c,b],1)])
        nt=len(self.faces)
        rows=np.repeat(np.arange(nt),3)
        gx=np.tile([-1/self.dx,1/self.dx,0],(len(a),1))
        gy=np.tile([-1/self.dy,0,1/self.dy],(len(a),1))
        gx=np.concatenate([gx,-gx]);gy=np.concatenate([gy,-gy])
        self.gx=sparse.csr_matrix((gx.ravel(),(rows,self.faces.ravel())),shape=(nt,self.n))
        self.gy=sparse.csr_matrix((gy.ravel(),(rows,self.faces.ravel())),shape=(nt,self.n))
        self.lap=(self.gx.T@self.gx+self.gy.T@self.gy)*(.5*self.area)
        self.last={}

    def refine_liquid_contact(self, factor=2, offset=.000050):
        """Constrain the triangle-interpolated surface on a finer contact grid."""
        ax=np.arange((self.nx-1)*factor+1)/factor
        ay=np.arange((self.ny-1)*factor+1)/factor
        xx,yy=np.meshgrid(ax,ay,indexing='ij')
        x=xx.ravel();y=yy.ravel()
        i=np.minimum(x.astype(int),self.nx-2);j=np.minimum(y.astype(int),self.ny-2)
        fx=x-i;fy=y-j;lower=fx+fy<=1
        a=i*self.ny+j;b=(i+1)*self.ny+j;c=i*self.ny+j+1;d=(i+1)*self.ny+j+1
        self.contact_vertices=np.where(lower[:,None],np.column_stack([a,b,c]),np.column_stack([d,c,b]))
        self.contact_weights=np.where(lower[:,None],np.column_stack([1-fx-fy,fx,fy]),
                                      np.column_stack([fx+fy-1,1-fx,1-fy]))
        self.contact_xy=np.column_stack([(x+.5)*self.dx-self.length/2,(y+.5)*self.dy-self.width/2])
        rows=np.repeat(np.arange(len(x)),3)
        self.contact_interpolation=sparse.csr_matrix((self.contact_weights.ravel(),
            (rows,self.contact_vertices.ravel())),shape=(len(x),self.n))
        self.contact_stiffness/=factor**2
        self.contact_offset=offset

    def mass(self,x):
        return idctn(dctn(x.reshape(self.nx,self.ny),norm='ortho')*self.mass_symbol,
                     norm='ortho').ravel()

    def project_surface(self, values):
        result=np.zeros_like(values)
        free=self.free_surface
        result[free]=values[free]-values[free].mean()
        return result

    def surface(self,h):
        sx,sy=self.gx@h,self.gy@h
        norm=np.sqrt(1+sx*sx+sy*sy)
        factor=.5*self.area*self.sigma
        energy=factor*np.sum(norm-1.) + .5*self.rho*self.gravity*self.area*np.dot(h,h)
        grad=factor*(self.gx.T@(sx/norm)+self.gy.T@(sy/norm))+self.rho*self.gravity*self.area*h
        return energy,grad

    def contact(self,positions,rotations,fast_spheres=True):
        """Lower envelope of rigid ellipsoids, with analytic six-DOF Jacobians."""
        rows,cols,vals=[],[],[]
        heights=[]
        indices=[]
        owners=[]
        count=0
        for bi,(body,q,R) in enumerate(zip(self.bodies,positions,rotations)):
            lower=np.full(len(self.contact_xy),np.inf)
            jac=np.zeros((len(self.contact_xy),6))
            components=zip(body.centers,body.radii)
            if fast_spheres and np.all(body.radii==body.radii[0,0]):
                centers=body.centers@R.T+q
                radius=body.radii[0,0]
                lo=centers[:,:2].min(0)-radius;hi=centers[:,:2].max(0)+radius
                selected=np.flatnonzero(np.all((self.contact_xy>=lo)&(self.contact_xy<=hi),axis=1))
                dxy=self.contact_xy[selected,None,:]-centers[None,:,:2]
                discr=radius**2-np.sum(dxy*dxy,axis=2)
                surface=np.where(discr>radius**2*.015,
                    centers[None,:,2]-np.sqrt(np.maximum(discr,0)),np.inf)
                nearest=surface.argmin(axis=1)
                height=surface[np.arange(len(selected)),nearest]
                finite=np.isfinite(height)
                selected=selected[finite];height=height[finite];nearest=nearest[finite]
                world=np.column_stack([self.contact_xy[selected],height])
                normal=world-centers[nearest]
                normalized=normal/normal[:,2,None]
                jac[selected,:3]=normalized
                jac[selected,3:]=np.cross(world-q,normalized)/body.scale
                lower[selected]=height
                components=[]
            for offset, radii in components:
                center=q+R@offset
                extent=np.sqrt((R*R)@(radii*radii))
                selected=np.flatnonzero(np.all(np.abs(self.contact_xy-center[:2])<extent[:2],axis=1))
                if not len(selected):continue
                A=(R/(radii*radii))@R.T
                d=self.contact_xy[selected]-center[:2]
                b=d@A[:2,2]
                c=np.einsum('ni,ij,nj->n',d,A[:2,:2],d)-1
                discr=b*b-A[2,2]*c
                valid=discr>A[2,2]*.015
                selected=selected[valid];d=d[valid];b=b[valid];discr=discr[valid]
                z=(-b-np.sqrt(discr))/A[2,2]
                world=np.column_stack([self.contact_xy[selected],center[2]+z])
                choose=world[:,2]<lower[selected]
                selected=selected[choose];world=world[choose]
                if not len(selected):continue
                normal=(world-center)@A
                normalized=normal/normal[:,2,None]
                jac[selected,:3]=normalized
                jac[selected,3:]=np.cross(world-q,normalized)/body.scale
                lower[selected]=world[:,2]
            chosen=np.flatnonzero(np.isfinite(lower))
            m=len(chosen)
            rr=np.arange(count,count+m)
            rows.extend(np.repeat(rr,self.contact_vertices.shape[1]))
            cols.extend(self.contact_vertices[chosen].ravel())
            vals.extend(self.contact_weights[chosen].ravel())
            for dof in range(6):
                rows.extend(rr);cols.extend(np.full(m,self.n+6*bi+dof));vals.extend(-jac[chosen,dof])
            heights.extend(lower[chosen]);indices.extend(chosen);owners.extend(np.full(m,bi))
            count+=m
        B=sparse.csr_matrix((vals,(rows,cols)),shape=(count,self.size))
        return B,np.asarray(indices,dtype=int),np.asarray(heights),np.asarray(owners,dtype=int)

    def step(self,newton_steps=6):
        dt=self.dt;n=self.n
        probe_data = None
        old_h = self.h.copy()
        flow_prediction = None
        if self.stirrer is not None:
            old_tip = self.stirrer.position(self.time)
            self.stirrer.advance(self.bodies, dt)
            tip = self.stirrer.position(self.time+dt)
            dxy = self.contact_xy-tip[:2]
            disc = self.stirrer.radius**2-np.sum(dxy*dxy, axis=1)
            pi = np.flatnonzero(disc > .015*self.stirrer.radius**2)
            pz = tip[2]-np.sqrt(disc[pi])
            PB = sparse.hstack([self.contact_interpolation[pi],
                                sparse.csr_matrix((len(pi),self.size-self.n))],format='csr')
            probe_data = tip, pi, pz, PB
            if self.flow is not None:
                flow_prediction = self.flow.prepare(tip, old_tip)
        self.v=idctn(dctn(self.v.reshape(self.nx,self.ny),norm='ortho')*
                       np.exp(-2*self.viscosity/self.rho*self.k**2*dt),norm='ortho').ravel()
        self.v=self.project_surface(self.v)
        q0=[];R0=[];blocks=[]
        for body in self.bodies:
            damping=6*np.pi*self.viscosity*body.scale if self.flow is None else 0.
            body.velocity/=1+dt*damping/body.mass
            body.omega/=1+dt*8*np.pi*self.viscosity*body.scale**3/np.mean(body.inertia)
            q0.append(body.position+dt*body.velocity)
            R0.append(Rotation.from_rotvec(dt*body.omega).as_matrix()@body.rotation)
            blocks.append(sparse.block_diag([np.eye(3)*body.mass,
                body.rotation@np.diag(body.inertia)@body.rotation.T/body.scale**2]))
        q0=np.asarray(q0);R0=np.asarray(R0)
        bm=sparse.block_diag(blocks,format='csr')
        bm_diag=bm.diagonal()
        hp=self.project_surface(self.h+dt*self.v) if flow_prediction is None else flow_prediction
        y=np.zeros(self.size);y[:n]=hp
        stiffness=self.sigma*self.lap+sparse.eye(n,format='csr')*(self.rho*self.gravity*self.area)
        weights=np.zeros(self.size)
        for i,body in enumerate(self.bodies):weights[n+i*6+2]=body.mass*self.gravity
        iterations=[]

        def poses(state):
            dq=state[n:].reshape(-1,6)
            positions=q0+dq[:,:3]
            rotations=np.asarray([Rotation.from_rotvec(d[3:]/body.scale).as_matrix()@R
                                  for d,body,R in zip(dq,self.bodies,R0)])
            return positions,rotations

        def evaluate(state,derivatives=True):
            h=state[:n];dh=h-hp;dq=state[n:]
            mass_h=self.mass(dh)/dt**2
            mass_b=bm@dq/dt**2
            energy,gs=self.surface(h)
            energy+=.5*np.dot(dh,mass_h)+.5*np.dot(dq,mass_b)+np.dot(weights,state)
            g=np.concatenate([mass_h+gs,mass_b])+weights
            q,R=poses(state)
            B,idx,bottom,owners=self.contact(q,R)
            if probe_data is not None:
                tip, pi, pz, PB = probe_data
                B=sparse.vstack([B,PB],format='csr')
                idx=np.r_[idx,pi];bottom=np.r_[bottom,pz]
                owners=np.r_[owners,np.full(len(pi),-1,dtype=int)]
            residual=(self.contact_interpolation@h)[idx]-bottom+self.contact_offset
            active=residual>0
            BA=B[active]
            force=self.contact_stiffness*np.maximum(residual,0)
            energy+=.5*self.contact_stiffness*np.dot(np.maximum(residual,0),np.maximum(residual,0))
            g+=B.T@force
            # Separate rigid bodies with a circular footprint.
            pairs=[]
            for i,body in enumerate(self.bodies):
                if probe_data is not None:
                    contacts=(rigid_probe_contacts(body,q[i],R[i],tip,self.stirrer.radius,
                              self.stirrer.length,self.contact_margin) if self.flow is not None
                              else [self.stirrer.solid_contact(body,q[i],R[i],tip)])
                    margin=self.contact_margin if self.flow is not None else 0.
                    for gap,jac in contacts:
                        gap-=margin
                        if gap<0:
                            row=np.zeros(self.size);row[n+6*i:n+6*i+6]=jac
                            rate=60. if self.flow is not None else 15.
                            g+=rate*gap*row;energy+=.5*rate*gap*gap
                            pairs.append(np.sqrt(rate)*row)
                    # A containing sphere keeps the stirred objects inside the basin.
                    bound=np.max(np.linalg.norm(body.centers,axis=1)+np.max(body.radii,axis=1))
                    for axis,extent in enumerate([self.length/2,self.width/2]):
                        wall=extent-bound-abs(q[i,axis])
                        if wall<0:
                            row=np.zeros(self.size);row[n+6*i+axis]=-np.sign(q[i,axis])
                            g+=20*wall*row;energy+=10*wall*wall
                            pairs.append(np.sqrt(20)*row)
                for j in range(i):
                    other=self.bodies[j]
                    if self.flow is not None:
                        for gap,ji,jj in rigid_pair_contacts(body,q[i],R[i],other,q[j],R[j],self.contact_margin):
                            gap-=self.contact_margin
                            row=np.zeros(self.size)
                            row[n+6*i:n+6*i+6]=ji;row[n+6*j:n+6*j+6]=jj
                            rate=60.
                            g+=rate*gap*row;energy+=.5*rate*gap*gap
                            pairs.append(np.sqrt(rate)*row)
                        continue
                    if self.stirrer is not None and len(body.centers)>1 and len(other.centers)>1:
                        for gap,ji,jj in wire_pair_contacts(body,q[i],R[i],other,q[j],R[j]):
                            row=np.zeros(self.size)
                            row[n+6*i:n+6*i+6]=ji;row[n+6*j:n+6*j+6]=jj
                            rate=8.
                            g+=rate*gap*row;energy+=.5*rate*gap*gap
                            pairs.append(np.sqrt(rate)*row)
                        continue
                    delta=q[i,:2]-q[j,:2];dist=np.linalg.norm(delta)
                    gap=dist-body.collision_radius-other.collision_radius
                    if gap<0 and dist>1.e-9:
                        direction=delta/dist;row=np.zeros(self.size)
                        row[n+6*i:n+6*i+2]=direction;row[n+6*j:n+6*j+2]=-direction
                        rate=8.
                        g+=rate*gap*row;energy+=.5*rate*gap*gap
                        pairs.append(np.sqrt(rate)*row)
                # Ground collision remains inactive in the high-tension demos.
                floor=q[i,2]+self.depth-body.scale*.4
                if floor<0:
                    row=np.zeros(self.size);row[n+6*i+2]=1
                    g+=20*floor*row;energy+=10*floor**2;pairs.append(np.sqrt(20)*row)
            g[:n]=self.project_surface(g[:n])
            C=sparse.csr_matrix(np.asarray(pairs)) if pairs else sparse.csr_matrix((0,self.size))
            return energy,g,BA,C,(q,R,idx,bottom,force,owners)

        for iteration in range(newton_steps):
            energy,g,B,C,aux=evaluate(y)
            contact_diag=self.contact_stiffness*np.asarray(B.multiply(B).sum(axis=0)).ravel()
            contact_diag+=np.asarray(C.multiply(C).sum(axis=0)).ravel()
            def mv(v):
                vv=v.copy();vv[:n]=self.project_surface(vv[:n])
                result=np.concatenate([self.mass(vv[:n])/dt**2+stiffness@vv[:n],bm@vv[n:]/dt**2])
                result+=self.contact_stiffness*(B.T@(B@vv))+C.T@(C@vv)
                result[:n]=self.project_surface(result[:n])+v[:n]-vv[:n]
                return result
            # The homogeneous fluid block is inverted spectrally; contact is
            # included in both the surface shift and the rigid diagonal.
            shift=np.percentile(contact_diag[:n],98)
            denom=(self.mass_symbol/dt**2+self.rho*self.gravity*self.area+
                   self.sigma*self.area*self.k**2+shift)
            body_denom=bm_diag/dt**2+contact_diag[n:]
            def pre(v):
                projected=self.project_surface(v[:n])
                sh=idctn(dctn(projected.reshape(self.nx,self.ny),norm='ortho')/denom,norm='ortho').ravel()
                sh=self.project_surface(sh)+v[:n]-projected
                return np.concatenate([sh,v[n:]/body_denom])
            operator=LinearOperator((self.size,self.size),matvec=mv)
            preconditioner=LinearOperator((self.size,self.size),matvec=pre)
            calls=[0]
            def cb(_):calls[0]+=1
            delta,info=cg(operator,-g,M=preconditioner,rtol=2.e-5,atol=1.e-12,maxiter=180,callback=cb)
            if info<0 or not np.isfinite(delta).all():raise RuntimeError('Non-finite coupled solve')
            alpha=1.
            slope=np.dot(g,delta)
            for line in range(10):
                candidate=y+alpha*delta
                candidate[:n]=self.project_surface(candidate[:n])
                new_energy=evaluate(candidate,False)[0]
                if new_energy<=energy+1.e-4*alpha*slope+1.e-16:break
                alpha*=.5
            y=candidate
            iterations.append(dict(cg=calls[0],info=int(info),step=float(alpha)))
            if np.max(np.abs(alpha*delta))<(3.e-8 if self.flow is not None else 2.e-7):break
        energy,g,B,C,aux=evaluate(y)
        q,R,idx,bottom,force,owners=aux
        self.v=(y[:n]-self.h)/dt
        self.h=y[:n].copy()
        for i,body in enumerate(self.bodies):
            body.velocity=(q[i]-body.position)/dt
            body.omega=Rotation.from_matrix(R[i]@body.rotation.T).as_rotvec()/dt
            body.position=q[i];body.rotation=R[i]
        if self.flow is not None:
            self.flow.finish(old_h)
        self.time+=dt
        self.last=dict(time=self.time,energy=float(energy),height_min=float(self.h.min()),
            height_max=float(self.h.max()),volume_change=float(self.area*self.h.sum()),
            max_penetration=float(np.maximum((self.contact_interpolation@self.h)[idx]-bottom,0).max(initial=0)),
            contact_force_z=[float(force[owners==i].sum()) for i in range(len(self.bodies))],
            gradient_max=float(np.abs(g).max()),iterations=iterations)
        if probe_data is not None:
            tip,pi,pz,PB=probe_data
            self.last.update(probe_position=tip.tolist(),
                probe_liquid_force_z=float(force[owners==-1].sum()),
                probe_liquid_penetration=float(np.maximum((self.contact_interpolation@self.h)[pi]-pz,0).max(initial=0)),
                probe_solid_penetration=float(max(0.,max(-self.stirrer.solid_contact(
                    b,q[i],R[i],tip)[0] for i,b in enumerate(self.bodies)))))
            wire_gaps=[]
            for i,b in enumerate(self.bodies):
                for j in range(i):
                    other=self.bodies[j]
                    if len(b.centers)>1 and len(other.centers)>1:
                        wire_gaps.extend(-entry[0] for entry in wire_pair_contacts(
                            b,q[i],R[i],other,q[j],R[j]))
            self.last['wire_pair_penetration']=float(max([0.,*wire_gaps]))
        if self.flow is not None:
            probe_gaps=[];pair_gaps=[]
            for i,b in enumerate(self.bodies):
                probe_gaps.extend(c[0] for c in rigid_probe_contacts(b,q[i],R[i],tip,
                                  self.stirrer.radius,self.stirrer.length))
                for j in range(i):
                    pair_gaps.extend(c[0] for c in rigid_pair_contacts(b,q[i],R[i],
                                     self.bodies[j],q[j],R[j]))
            self.last.update(self.flow.last)
            self.last.update(continuous_probe_penetration=max([0.,*[-v for v in probe_gaps]]),
                             continuous_pair_penetration=max([0.,*[-v for v in pair_gaps]]))
        if not np.isfinite(y).all():raise RuntimeError('Non-finite simulation state')
        return self.last

    def snapshot(self):
        result=dict(height=self.h.reshape(self.nx,self.ny).astype(np.float32),
                    velocity=self.v.reshape(self.nx,self.ny).astype(np.float32),
                    positions=np.asarray([b.position for b in self.bodies]),
                    rotations=np.asarray([b.rotation for b in self.bodies]),
                    body_velocity=np.asarray([b.velocity for b in self.bodies]),
                    body_omega=np.asarray([b.omega for b in self.bodies]),time=self.time)
        if self.stirrer is not None:
            result['probe_position']=self.stirrer.position(self.time)
        if self.flow is not None:
            result.update(self.flow.snapshot())
        return result
