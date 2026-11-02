"""Staggered surface currents coupled to the finite-depth capillary solve.

Velocity transport retains rotational flow. The divergence couples to the
height solve through tanh(k D)/k, preserving the existing linear wave
inertia. Conservative surface fluxes carry the nonlinear height perturbation.
"""
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import map_coordinates


class SurfaceFlow:
    def __init__(self, system):
        self.s = system
        nx, ny = system.nx, system.ny
        self.u = np.zeros((nx+1, ny))
        self.w = np.zeros((nx, ny+1))
        self.k2 = system.k**2
        safe = np.maximum(system.k, 1.e-12)
        self.transport = np.tanh(safe*system.depth)/safe
        self.transport[0, 0] = system.depth
        self.xu, self.yu = np.meshgrid(np.arange(nx+1)*system.dx-system.length/2,
            (np.arange(ny)+.5)*system.dy-system.width/2, indexing='ij')
        self.xw, self.yw = np.meshgrid((np.arange(nx)+.5)*system.dx-system.length/2,
            np.arange(ny+1)*system.dy-system.width/2, indexing='ij')
        self.last = {}
        self.coupling_depth = .0025
        rng = np.random.default_rng(719)
        self.tracers = rng.uniform([-.46*system.length, -.46*system.width],
                                  [.46*system.length, .46*system.width], (160, 2))

    def sample(self, array, x, y, axis):
        s = self.s
        i = (x+s.length/2)/s.dx-(.0 if axis == 0 else .5)
        j = (y+s.width/2)/s.dy-(.5 if axis == 0 else .0)
        return map_coordinates(array, [i, j], order=1, mode='nearest')

    def divergence(self, u, w):
        return np.diff(u, axis=0)/self.s.dx+np.diff(w, axis=1)/self.s.dy

    def gradient(self, scalar):
        u = np.zeros_like(self.u); w = np.zeros_like(self.w)
        u[1:-1] = np.diff(scalar, axis=0)/self.s.dx
        w[:, 1:-1] = np.diff(scalar, axis=1)/self.s.dy
        return u, w

    def filter(self, values, symbol):
        return idctn(dctn(values, norm='ortho')*symbol, norm='ortho')

    def correct_divergence(self, target):
        delta = target-self.divergence(self.u, self.w)
        inverse = np.zeros_like(self.k2)
        inverse[self.k2 > 0] = -1/self.k2[self.k2 > 0]
        potential = self.filter(delta, inverse)
        du, dw = self.gradient(potential)
        self.u += du; self.w += dw
        return float(np.max(np.abs(self.divergence(self.u, self.w)-target)))

    def initialize(self):
        self.correct_divergence(-self.filter(self.s.v.reshape(self.k2.shape), 1/self.transport))

    def prepare(self, tip, old_tip):
        s = self.s; dt = s.dt
        old_u = self.u.copy(); old_w = self.w.copy()
        # Midpoint semi-Lagrangian transport of face velocities.
        for array, x, y, axis in [(self.u, self.xu, self.yu, 0),
                                   (self.w, self.xw, self.yw, 1)]:
            vx = self.sample(old_u, x, y, 0); vy = self.sample(old_w, x, y, 1)
            mx = x-.5*dt*vx; my = y-.5*dt*vy
            vx = self.sample(old_u, mx, my, 0); vy = self.sample(old_w, mx, my, 1)
            original = old_u if axis == 0 else old_w
            array[:] = self.sample(original, x-dt*vx, y-dt*vy, axis)
            # Implicit spectral diffusion uses an even extension at side walls.
            ii, jj = np.indices(array.shape)
            k2 = (2*np.sin(np.pi*ii/(2*array.shape[0]))/s.dx)**2
            k2 += (2*np.sin(np.pi*jj/(2*array.shape[1]))/s.dy)**2
            array[:] = idctn(dctn(array, norm='ortho')/(1+dt*s.viscosity/s.rho*k2), norm='ortho')
        probe_velocity = (tip-old_tip)/dt
        immersed = np.clip((s.stirrer.radius-tip[2])/(2*s.stirrer.radius), 0, 1)
        radius = s.stirrer.radius
        # Brinkman relaxation imposes the moving immersed boundary. Its
        # horizontal momentum enters continuity before the pressure solve.
        probe_work = 0.
        for array, x, y, axis in [(self.u, self.xu, self.yu, 0),
                                   (self.w, self.xw, self.yw, 1)]:
            distance = np.sqrt((x-tip[0])**2+(y-tip[1])**2)
            mask = np.clip((radius+.5*s.dx-distance)/s.dx, 0, 1)
            relaxation = 1-np.exp(-dt*1200*immersed*mask)
            change = relaxation*(probe_velocity[axis]-array)
            probe_work += float(np.sum(np.abs(change))*s.rho*s.area*radius)
            array += change
        self.u[[0, -1]] = 0; self.w[:, [0, -1]] = 0
        self.couple_bodies()
        h = s.h.reshape(s.nx, s.ny)
        inverse = np.zeros_like(self.k2)
        inverse[self.k2 > 0] = -1/self.k2[self.k2 > 0]
        potential = self.filter(self.divergence(self.u, self.w), inverse)
        pu, pw = self.gradient(potential)
        ru, rw = self.u-pu, self.w-pw
        hx = np.zeros_like(self.u); hy = np.zeros_like(self.w)
        hx[1:-1] = np.where(ru[1:-1] >= 0, h[:-1], h[1:])
        hy[:, 1:-1] = np.where(rw[:, 1:-1] >= 0, h[:, :-1], h[:, 1:])
        # Both flux divergences telescope to zero at the basin walls.
        self.height_flux = self.divergence(hx*ru, hy*rw)
        dh = -self.filter(self.divergence(self.u, self.w), self.transport)-self.height_flux
        predicted = s.project_surface((h+dt*dh).ravel())
        self.last = dict(probe_momentum_impulse=probe_work)
        return predicted

    def couple_bodies(self):
        """Exchange horizontal drag impulses with equal opposite fluid impulses."""
        s = self.s; errors = []
        for body in s.bodies:
            offsets = body.centers @ body.rotation.T
            points = body.position+offsets
            # Sample the submerged wire or the floating disc's footprint.
            if len(points) == 1:
                angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
                local = np.column_stack([body.radii[0, 0]*.7*np.cos(angles),
                                         body.radii[0, 1]*.7*np.sin(angles), np.zeros(24)])
                offsets = local@body.rotation.T; points = body.position+offsets
                length = 2*np.pi*body.radii[0, 0]
                radius = body.radii[0, 2]
            else:
                length = np.linalg.norm(np.diff(body.centers, axis=0), axis=1).sum()
                radius = body.radii[0, 0]
            uv = np.column_stack([self.sample(self.u, points[:, 0], points[:, 1], 0),
                                  self.sample(self.w, points[:, 0], points[:, 1], 1)])
            bv = (body.velocity+np.cross(body.omega, offsets))[:, :2]
            rel = uv-bv
            # Linear viscous drag plus cross-flow drag, integrated over length.
            coefficient = (4*np.pi*s.viscosity+1.2*s.rho*radius*np.linalg.norm(rel, axis=1))*length/len(points)
            height = map_coordinates(s.h.reshape(s.nx, s.ny),
                [(points[:, 0]+s.length/2)/s.dx-.5, (points[:, 1]+s.width/2)/s.dy-.5],
                order=1, mode='nearest')
            wet = np.clip((height-points[:, 2]+radius)/(2*radius), 0, 1)
            impulse = s.dt*coefficient[:, None]*wet[:, None]*rel
            # Implicit relaxation for the translational reduced mass.
            impulse /= 1+s.dt*coefficient.sum()/body.mass
            impulse3 = np.column_stack([impulse, np.zeros(len(impulse))])
            total = impulse3.sum(0)
            body.velocity += total/body.mass
            inertia = body.rotation@np.diag(body.inertia)@body.rotation.T
            body.omega += np.linalg.solve(inertia, np.cross(offsets, impulse3).sum(0))
            mass = s.rho*s.area*self.coupling_depth
            before = np.array([self.u.sum(), self.w.sum()])*mass
            for axis, array in [(0, self.u), (1, self.w)]:
                x = (points[:, 0]+s.length/2)/s.dx-(0 if axis == 0 else .5)
                y = (points[:, 1]+s.width/2)/s.dy-(.5 if axis == 0 else 0)
                ix = np.floor(x).astype(int); iy = np.floor(y).astype(int)
                fx = x-ix; fy = y-iy
                for di, dj, weight in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                                       (0, 1, (1-fx)*fy), (1, 1, fx*fy)]:
                    i = np.clip(ix+di, 1 if axis == 0 else 0, array.shape[0]-2 if axis == 0 else array.shape[0]-1)
                    j = np.clip(iy+dj, 1 if axis == 1 else 0, array.shape[1]-2 if axis == 1 else array.shape[1]-1)
                    np.add.at(array, (i, j), -impulse[:, axis]*weight/mass)
            after = np.array([self.u.sum(), self.w.sum()])*mass
            errors.append(float(np.linalg.norm(total[:2]+after-before)))
        self.momentum_error = max(errors, default=0.)

    def finish(self, old_h):
        s = self.s
        rate = (s.h.reshape(s.nx, s.ny)-old_h.reshape(s.nx, s.ny))/s.dt
        target = -self.filter(rate+self.height_flux, 1/self.transport)
        target -= target.mean()
        error = self.correct_divergence(target)
        ux = .5*(self.u[:-1]+self.u[1:]); uy = .5*(self.w[:, :-1]+self.w[:, 1:])
        curl = np.gradient(uy, s.dx, axis=0)-np.gradient(ux, s.dy, axis=1)
        cfl = s.dt*(np.max(np.abs(self.u))/s.dx+np.max(np.abs(self.w))/s.dy)
        if cfl > 1.0:
            raise RuntimeError(f'Surface flow CFL {cfl:.3f} exceeds one; reduce dt.')
        x, y = self.tracers.T
        uv = np.column_stack([self.sample(self.u, x, y, 0), self.sample(self.w, x, y, 1)])
        mid = self.tracers+.5*s.dt*uv
        velocity = np.column_stack([self.sample(self.u, *mid.T, 0), self.sample(self.w, *mid.T, 1)])
        self.tracers += s.dt*velocity
        self.tracers = np.clip(self.tracers, [-s.length/2+s.dx, -s.width/2+s.dy],
                              [s.length/2-s.dx, s.width/2-s.dy])
        self.last.update(flow_rms_speed=float(np.sqrt(np.mean(ux*ux+uy*uy))),
                         flow_rms_vorticity=float(np.sqrt(np.mean(curl*curl))),
                         flow_continuity_error=error, flow_cfl=float(cfl),
                         drag_momentum_error=self.momentum_error)

    def snapshot(self):
        s = self.s
        z = map_coordinates(s.h.reshape(s.nx, s.ny),
            [(self.tracers[:, 0]+s.length/2)/s.dx-.5, (self.tracers[:, 1]+s.width/2)/s.dy-.5],
            order=1, mode='nearest')
        return dict(flow_u=self.u.astype(np.float32), flow_v=self.w.astype(np.float32),
                    tracer_positions=np.column_stack([self.tracers, z]).astype(np.float32))
