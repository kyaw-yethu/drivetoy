"""Plain-NumPy geometry shared by the Blender script and the tests (no bpy import)."""
import numpy as np


def seg_dist(p, a, b):
    """Distance from point p to the segment a-b (3D)."""
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def crown_blocks(c, r, segs, margin=0.3):
    """Whether a crown of radius r centred at c comes within ``margin`` of any of the
    sightlines ``segs`` (rows x0, y0, z0, x1, y1, z1)."""
    segs = np.asarray(segs, float).reshape(-1, 6)
    if not len(segs):
        return False
    c = np.asarray(c, float)
    a, ab = segs[:, :3], segs[:, 3:] - segs[:, :3]
    t = np.clip(np.einsum("ij,ij->i", c - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    return bool((np.linalg.norm(c - (a + t[:, None] * ab), axis=1) < r + margin).any())
