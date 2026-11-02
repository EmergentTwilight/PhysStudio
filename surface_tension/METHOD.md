# Capillary membrane and rigid contact

Reference: Ruan, Liu, Zhu, Sueda, Wang, and Chen,
[Solid-Fluid Interaction with Surface-Tension-Dominant Contact](https://arxiv.org/html/2105.08471v1),
ACM TOG 40(4), Article 120, SIGGRAPH 2021.

## Implemented system

The basin spans 40 × 32 mm and is 12 mm deep. Its top surface is represented by
fixed horizontal vertices and changing heights `h`. Each quadrilateral has two
triangles. Rigid objects carry position, orientation, linear velocity, angular
velocity, mass, and inertia. The demos use liquid density 1000 kg/m³, surface
tension 0.072 N/m, dynamic viscosity 0.003 Pa·s, and gravity 9.81 m/s².

For every triangle, constant matrices `Gx`, `Gy` give the two graph slopes. Its
energy and the gravitational energy of the displaced liquid are

```text
E_surface = sum_t sigma * A0_t * sqrt(1 + (Gx h)_t² + (Gy h)_t²)
          + (rho g / 2) * sum_i cell_area * h_i².
```

The constant flat area is subtracted only for numerical energy evaluation.
Forces use the exact derivative of this nonlinear energy. A positive flat-area
stiffness supplies the implicit descent direction, followed by an energy line
search. This follows the paper's area-energy mechanism (equations 4 and 9)
and positive implicit stiffness idea (equation 11), restricted to graph motion.

## Three-dimensional fluid response

The velocity potential solves the incompressible Laplace problem in a finite
depth rectangular liquid volume, with no-flux bottom and side boundaries. For
a nonconstant horizontal mode with wave number `k`, eliminating the interior
potential gives surface added mass

```text
M_fluid(k) = rho * cell_area / (k tanh(k H)).
M_membrane = rho * (0.5 min(dx,dy)) * cell_area.
```

A discrete cosine transform diagonalizes the lateral no-flux operator;
`k² = (2 sin(pi mx/(2 Nx))/dx)² + (2 sin(pi my/(2 Ny))/dy)²`.
This is a nonlocal inertia operator: displacing one surface patch moves the
underlying liquid. It is not an independent spring or local water column.
The constant mode is removed exactly, so `sum(h)=0` in the closed basin.
Linear viscous decay damps each mode by `exp(-2 nu k² dt)`. Rigid Stokes drag
uses `6 pi mu r` for translation and `8 pi mu r³` for rotation.

The baseline fluid reduction assumes potential flow and a fixed mean
free-surface domain. The updated stirred variant adds the surface-current
transport described below. It is different from
the paper's fully advected MAC fluid and triangle/dual-cell intersection
operator `W`.

## Coupled rigid contact

The lower envelope of each rigid proxy gives a support height `s_i(q,R)` over
a surface vertex. Wire proxies are unions of spheres; apples and pin bases use
ellipsoids. The sphere batch path is numerically checked against the general
ellipsoid implementation. The unilateral contact energy is

```text
E_contact = 0.5 * k_contact * sum_i max(h_i - s_i(q,R), 0)²,
k_contact = 8e6 N/m³ * cell_area.
```

With surface normal `n` and offset `r` from the center of mass, the support
height Jacobian is

```text
J = [nx/nz, ny/nz, 1, (r cross n)/nz].
B = [I_surface, -J_rigid].
```

The six-component rigid rotation increment is scaled by its object radius
internally to improve conditioning; the corresponding inertia is transformed
by the same scaling. `Bᵀ` transfers a contact reaction to both the membrane and
rigid body, including torque. As in the paper's equation 20, forces applied
away from the center of mass therefore affect angular motion.

All membrane heights and rigid increments enter the same implicit Euler
objective:

```text
0.5/dt² * (h - h_predict)ᵀ M_surface (h - h_predict)
+ 0.5/dt² * delta_qᵀ M_rigid delta_q
+ E_surface + E_contact + E_rigid_gravity + E_body_contact.
```

Each iteration rebuilds the active contacts, solves a matrix-free symmetric
system using preconditioned conjugate gradients, and performs an energy line
search. The homogeneous membrane-fluid block has a spectral preconditioner.
Rigid-body collisions use a circular footprint penalty, and the
basin floor supplies a unilateral penalty. The high-tension demos stay well
above the floor.

## Driven stirring and horizontal currents

The updated stirred presets use `SurfaceFlow`, a horizontal staggered velocity
field on the membrane domain. Face velocities are transported with midpoint
semi-Lagrangian tracing and damped by implicit viscous diffusion. Zero normal
velocity closes the basin walls. A smooth Brinkman footprint relaxes immersed
rod velocities toward the prescribed capsule velocity before the pressure solve.
The rod inserts smoothly, sweeps a figure-eight path and changes immersion depth.
Paperclips use a 1.1 s sweep; the cluster uses 1.25 s and follows its centroid
through a 0.35 s filter bounded at 8 mm/s.

Let `D` be the staggered divergence, `G` its compatible gradient, and
`T(k)=tanh(k H)/k`. The predictor is

```text
h_predict = h_old - dt * T D u_star - dt * D(h_upwind u_rot).
```

`u_rot` is the discrete divergence-free part of the transported velocity.
Only this component advects the height perturbation; compressive wave motion
already enters the finite-depth operator. The upwind fluxes telescope at the
closed basin boundary, conserving mean height. The existing implicit capillary
and rigid-body objective solves for the new height. A Neumann Poisson correction
then makes the face velocity satisfy that height change with the same `T`.
The divergence/gradient identity and correction residual are tested separately.
This preserves the baseline finite-depth linear wave response while retaining
transported rotational currents. It is a dispersive surface-current model,
not a volumetric 3D Navier-Stokes solver.

Bodies exchange horizontal viscous and cross-flow drag impulses with the grid.
The reaction is scattered with bilinear weights; the fluid coupling layer is
2.5 mm deep. The body's translation and rotation respond to the summed impulse
and torque. Actual grid momentum changes are audited against body impulses.
160 sparse massless surface tracers follow the saved velocity field using
midpoint integration. They visualize flow and do not create waves or forces.

## Continuous solid and refined liquid contact

Paperclips use continuous capsule segments along the exact rendered polyline.
Closest-segment queries include interior, endpoint and parallel cases. Spatially
separated patches transfer equal and opposite forces and contact torques.
The rod is a capsule, and all relevant patches enter the implicit objective;
there is no single-deepest-point reduction. Pin bounds include the bottom disc,
colored handle and connecting stem. These pin bounds are conservative spheres.

Solid contact activates at a 60 micrometer separation with 60 N/m stiffness.
This resolution allowance keeps finite penalty displacement outside visible
surfaces. The lower liquid envelope is evaluated on a contact grid with twice
the membrane resolution in each direction. Its surface height is interpolated
with the actual triangle barycentric coordinates. The interpolation and rigid
Jacobian appear in the same contact matrix, preserving force transfer. Contact
quadrature area scales with the refined sampling density. A 50 micrometer
liquid offset accounts for finite penalties and between-sample curvature.

The cluster pins the outer membrane vertices and projects out mean interior
height. Contact and water diagnostics include maxima over every integration
substep. `flow_checks.py` separately recomputes solid gaps from saved poses;
`validate.py` retains the original energy and rigid/liquid derivative checks.
The potential-only variant remains available through `--legacy-stirrer`.

## Relation to the original algorithm

| Component | Current implementation |
| --- | --- |
| Triangle area energy and capillary force | Nonlinear, differentiated exactly on a graph mesh |
| Massive surface membrane | Mass per vertex from half-cell virtual thickness |
| Liquid response | Finite-depth wave inertia; stirred variants add transported surface currents |
| Rigid dynamics | Translation and rotation with mass and inertia |
| Coupled contact | One implicit membrane/rigid solve with unilateral penalty and analytic Jacobian |
| Volume conservation | Exact zero-mean surface constraint |
| Original MAC advection and dual-cell `W` | Replaced by the surface-current adapter and triangle contact interpolation |
| Paper's two-stage contact/advection solve | Replaced by an active-contact variational step |
| Remeshing, overhangs, interface breakup | Outside the graph adapter |

## Numerical and visual audit

`validate.py` checks independent finite differences of the energy gradient and
six contact derivatives. It also checks mass symmetry/positivity, vertical
action/reaction, and batched wire-query equivalence. Trajectory audits require
finite values, orthogonal rigid rotations, relative volume drift below 1e-4,
and sampled contact penetration below 0.1 mm. Updated stirred runs also store penetration and CFL extrema over every
integration substep, including those between saved output frames.

Rendering is independent of simulation. Water is a closed mesh built from the
cached membrane and fixed basin depth. Object transforms come from the cached
six-DOF states. Cycles handles transmission and reflection; submerged joints
make refractive distortion visible. All scene measurements are in meters;
rendering uniformly scales geometry by 100 for camera and lighting setup.

Surface-current discretization background: Robert Bridson's
[height-field fluid lecture notes](https://www.cs.ubc.ca/~rbridson/courses/533d-fall-2005/nov17-cs533d-slides.pdf)
and the conservative flux discussion in
[Wong's shallow-water thesis](https://www.cs.ubc.ca/~rbridson/docs/awong_msc.pdf).
The implementation keeps the finite-depth spectral wave operator above instead
of substituting the shallow-water limit.
