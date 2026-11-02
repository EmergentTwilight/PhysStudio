"""Continuous capsule contacts matching the rendered polyline centerlines."""
import numpy as np


def segments(body, q, rotation):
    points = body.centers @ rotation.T + q
    valid = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.e-12
    return points[:-1][valid], points[1:][valid]


def closest_segments(a, b, c, d):
    """Closest points for broadcast segment pairs, including parallel segments."""
    u, v, w = b-a, d-c, a-c
    aa = np.sum(u*u, axis=-1); bb = np.sum(u*v, axis=-1)
    cc = np.sum(v*v, axis=-1); dd = np.sum(u*w, axis=-1)
    ee = np.sum(v*w, axis=-1)
    determinant = aa*cc-bb*bb
    safe = np.maximum(determinant, 1.e-40)
    s = (bb*ee-cc*dd)/safe; t = (aa*ee-bb*dd)/safe
    interior = (determinant > 1.e-12*aa*cc) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    ss = np.stack([np.zeros_like(s), np.ones_like(s),
                   np.clip(-dd/aa, 0, 1), np.clip((bb-dd)/aa, 0, 1), s])
    tt = np.stack([np.clip(ee/cc, 0, 1), np.clip((ee+bb)/cc, 0, 1),
                   np.zeros_like(t), np.ones_like(t), t])
    p = a+ss[..., None]*u; q = c+tt[..., None]*v
    distance2 = np.sum((p-q)**2, axis=-1)
    distance2[-1] = np.where(interior, distance2[-1], np.inf)
    best = np.argmin(distance2, axis=0)
    shape = best.shape
    indices = np.indices(shape)
    selection = (best, *indices)
    return p[selection], q[selection]


def capsule_contacts(a, b, c, d, radius, margin=0.):
    lo1 = np.minimum(a, b); hi1 = np.maximum(a, b)
    lo2 = np.minimum(c, d); hi2 = np.maximum(c, d)
    near = np.all((lo1[:, None] <= hi2[None]+radius+margin) &
                  (lo2[None] <= hi1[:, None]+radius+margin), axis=-1)
    i, j = np.nonzero(near)
    if not len(i):
        return []
    p, q = closest_segments(a[i], b[i], c[j], d[j])
    delta = p-q; distance = np.linalg.norm(delta, axis=1)
    gap = distance-radius
    active = np.flatnonzero(gap < margin)
    contacts = []
    for k in active[np.argsort(gap[active])]:
        # Separate surface patches instead of accumulating coincident endpoints.
        if any(np.linalg.norm(p[k]-x[1]) < radius*.65 and
               np.linalg.norm(q[k]-x[2]) < radius*.65 for x in contacts):
            continue
        if distance[k] < 1.e-14:
            raise RuntimeError('Intersecting centerlines: reduce the time step.')
        contacts.append((float(gap[k]), p[k], q[k], delta[k]/distance[k]))
    return contacts


def wire_contacts(first, q1, r1, second, q2, r2, margin=0.):
    a, b = segments(first, q1, r1); c, d = segments(second, q2, r2)
    radius = first.radii[0, 0]+second.radii[0, 0]
    result = []
    for gap, p, q, normal in capsule_contacts(a, b, c, d, radius, margin):
        j1 = np.r_[normal, np.cross(p-q1, normal)/first.scale]
        j2 = -np.r_[normal, np.cross(q-q2, normal)/second.scale]
        result.append((gap, j1, j2))
    return result


def probe_contacts(body, position, rotation, tip, radius, length, margin=0.):
    a, b = segments(body, position, rotation)
    c = tip[None]; d = (tip+np.array([0., 0., length]))[None]
    result = []
    for gap, p, _, normal in capsule_contacts(a, b, c, d, body.radii[0, 0]+radius, margin):
        result.append((gap, np.r_[normal, np.cross(p-position, normal)/body.scale]))
    return result


def solid_spheres(body, position, rotation):
    """Conservative bounds for the disc, enamel handle and connecting stem."""
    centers = body.centers.copy()
    radii = np.max(body.radii, axis=1)
    if body.name.startswith('pin_'):
        centers = np.vstack([centers, [0., 0., .0019]])
        radii = np.r_[radii, .0012]
        # The two bounding spheres also contain the narrow stem between them.
    return centers@rotation.T+position, radii


def rigid_probe_contacts(body, position, rotation, tip, radius, length, margin=0.):
    if len(body.centers) > 1:
        return probe_contacts(body, position, rotation, tip, radius, length, margin)
    centers, radii = solid_spheres(body, position, rotation)
    nearest = np.tile(tip, (len(centers), 1))
    nearest[:, 2] = np.clip(centers[:, 2], tip[2], tip[2]+length)
    delta = centers-nearest; dist = np.linalg.norm(delta, axis=1)
    result = []
    for i in np.flatnonzero(dist-radii-radius < margin):
        normal = delta[i]/max(dist[i], 1.e-12)
        result.append((dist[i]-radii[i]-radius,
                       np.r_[normal, np.cross(centers[i]-position, normal)/body.scale]))
    return result


def rigid_pair_contacts(first, q1, r1, second, q2, r2, margin=0.):
    if len(first.centers) > 1 and len(second.centers) > 1:
        return wire_contacts(first, q1, r1, second, q2, r2, margin)
    a, ra = solid_spheres(first, q1, r1); b, rb = solid_spheres(second, q2, r2)
    delta = a[:, None]-b[None]; dist = np.linalg.norm(delta, axis=2)
    gap = dist-ra[:, None]-rb[None]
    result = []
    for i, j in np.argwhere(gap < margin):
        normal = delta[i, j]/max(dist[i, j], 1.e-12)
        result.append((gap[i, j], np.r_[normal, np.cross(a[i]-q1, normal)/first.scale],
                       -np.r_[normal, np.cross(b[j]-q2, normal)/second.scale]))
    return result
