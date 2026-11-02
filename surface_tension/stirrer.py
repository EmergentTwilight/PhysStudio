"""Prescribed capsule motion and differentiable solid contact geometry."""
import numpy as np


class Stirrer:
    def __init__(self, scene, flow_coupled=False):
        self.flow_coupled = flow_coupled
        self.scene = scene
        self.radius = .00115 if scene == 'paperclips' else .00130
        self.length = .018
        self.period = (1.1 if scene == 'paperclips' else 1.25) if flow_coupled else (1.45 if scene == 'paperclips' else 2.2)
        self.center = np.zeros(2)

    def advance(self, bodies, dt):
        if self.scene != 'capillary_cluster':
            return
        target=np.mean([body.position[:2] for body in bodies],axis=0)
        target=np.clip(target,[-.0137,-.0108],[.0137,.0108])
        change=(target-self.center)*(1-np.exp(-dt/.35))
        speed=np.linalg.norm(change)/dt
        self.center+=change*min(1.,.008/max(speed,1.e-12))

    def position(self, time):
        phase = np.pi + 2*np.pi*max(time-.18, 0)/self.period
        if self.scene == 'paperclips':
            xy = [.0115*np.cos(phase), .006*np.sin(2*phase)]
        else:
            xy = self.center+np.array([(.0058 if self.flow_coupled else .0043)*np.cos(phase), (.0038 if self.flow_coupled else .0028)*np.sin(2*phase)])
        u = np.clip((time-.08)/.28, 0, 1)
        u = u*u*(3-2*u)
        z = (1-u)*.006 + u*((-.00045+.00055*np.cos(2*phase)) if self.flow_coupled else (.00048+.00022*np.cos(phase)))
        return np.array([*xy, z])

    def config(self):
        return dict(scene=self.scene, radius=self.radius, length=self.length,
                    period=self.period, flow_coupled=self.flow_coupled, motion='periodic capsule with centroid tracking'
                    if self.scene=='capillary_cluster' else 'prescribed periodic capsule')

    def solid_contact(self, body, position, rotation, tip):
        """Deepest sphere/capsule contact and scaled six-DOF gap derivative.

        Wire component spheres are exact. Other components use enclosing
        spheres for this probe contact. Dynamic liquid contact keeps ellipsoids.
        """
        offsets = body.centers @ rotation.T
        centers = position + offsets
        nearest = np.tile(tip, (len(centers), 1))
        nearest[:, 2] = np.clip(centers[:, 2], tip[2], tip[2]+self.length)
        delta = centers-nearest
        distances = np.linalg.norm(delta, axis=1)
        gaps = distances-self.radius-np.max(body.radii, axis=1)
        index = np.argmin(gaps)
        direction = delta[index]/max(distances[index], 1.e-12)
        jac = np.r_[direction, np.cross(offsets[index], direction)/body.scale]
        return gaps[index], jac
