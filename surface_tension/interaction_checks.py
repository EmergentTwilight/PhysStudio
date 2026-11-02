"""Independent gradient and constraint checks for driven-contact scenes."""
import numpy as np
from scipy.spatial.transform import Rotation
from .scenes import make_scene
from .solver import CapillarySystem, wire_pair_contacts


def check_interactions():
    first,second=make_scene('paperclips')[0][:2]
    q1=np.zeros(3);q2=np.array([.0054,.00043,.00024])
    R1=first.rotation;R2=second.rotation
    contacts=wire_pair_contacts(first,q1,R1,second,q2,R2)
    assert contacts
    gradient=sum(gap*np.r_[j1,j2] for gap,j1,j2 in contacts)
    eps=1.e-9;finite=[]
    for dof in range(12):
        values=[]
        for sign in [-1,1]:
            a=q1.copy();b=q2.copy();A=R1.copy();B=R2.copy();axis=dof%6
            if axis<3:
                if dof<6:a[axis]+=sign*eps
                else:b[axis]+=sign*eps
            else:
                dr=np.zeros(3)
                dr[axis-3]=sign*eps/(first.scale if dof<6 else second.scale)
                if dof<6:A=Rotation.from_rotvec(dr).as_matrix()@A
                else:B=Rotation.from_rotvec(dr).as_matrix()@B
            values.append(sum(.5*gap**2 for gap,_,_ in
                              wire_pair_contacts(first,a,A,second,b,B)))
        finite.append((values[1]-values[0])/(2*eps))
    error=float(np.max(np.abs(gradient-finite)))
    assert error<1.e-7,error
    force_error=max(float(np.linalg.norm(j1[:3]+j2[:3])) for _,j1,j2 in contacts)
    assert force_error<1.e-12

    system=CapillarySystem(make_scene('capillary_cluster')[0],nx=32,ny=26,pinned_rim=True)
    rng=np.random.default_rng(20);a=rng.normal(size=system.n);b=rng.normal(size=system.n)
    pa=system.project_surface(a);pb=system.project_surface(b)
    assert abs(pa.sum())<1.e-12
    assert np.max(np.abs(pa[~system.free_surface]))==0
    assert np.max(np.abs(system.project_surface(pa)-pa))<1.e-12
    symmetry=float(abs(np.dot(a,pb)-np.dot(pa,b)))
    assert symmetry<1.e-10
    return dict(wire_pair_energy_gradient_error=error,
                wire_pair_force_balance_error=force_error,
                pinned_projection_symmetry_error=symmetry)
