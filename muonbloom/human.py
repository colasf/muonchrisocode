"""The standing figure of YOU / FLOOD / OUTRO: a real body mesh instead of lofted ellipses.

data/human.npz is baked by tools/build_human.py from the MakeHuman base mesh (CC0): 1.80 m, feet on y = 0,
facing +z, its left side on +x, arms down. Nothing heavy runs here: the mesh is only sampled into a point
cloud; the slice contours, the voxels of the hit test and the points of the silhouette come ready.
"""
from __future__ import annotations

import functools

import numpy as np

from .layout import DATA

_D = np.load(DATA / "human.npz")

PART_NAMES = [str(p) for p in _D["parts"]]
HEIGHT = float(_D["height"])
HEART = _D["heart"].astype(np.float64)

VERTS, TRIS, VNRM, VPART = _D["verts"], _D["tris"], _D["vnrm"], _D["vpart"]
LEVELS, CA, CB, CLEV = _D["levels"], _D["ca"], _D["cb"], _D["clev"]       # slice contours every 3 cm
_VOX, _ORG, _STEP = _D["vox"], _D["vox_org"], float(_D["vox_step"])
_CORE, _CORE_PART = _D["core_pts"], _D["core_part"]
_O_Y, _O_XZ, _O_START = _D["o_levels"], _D["o_xz"].astype(np.float64), _D["o_start"]
_O_GLEV = _D["o_lev"][_O_START]

_AREA = 0.5 * np.linalg.norm(np.cross(VERTS[TRIS[:, 1]] - VERTS[TRIS[:, 0]], VERTS[TRIS[:, 2]] - VERTS[TRIS[:, 0]]),
                             axis=1).astype(np.float64)
AREA = float(_AREA.sum())                   # m2 of skin


def cloud(rng, density):
    """Points spread evenly over the skin (`density` per m2) -> (points, normals, part of each point)."""
    n = int(rng.poisson(AREA * density))
    f = rng.choice(len(TRIS), n, p=_AREA / AREA)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u, v = np.where(flip, 1.0 - u, u), np.where(flip, 1.0 - v, v)
    w = np.stack([1.0 - u - v, u, v], 1).astype(np.float32)[:, :, None]
    t = TRIS[f]
    nrm = (VNRM[t] * w).sum(1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    part = VPART[t[np.arange(n), w[:, :, 0].argmax(1)]]
    return (VERTS[t] * w).sum(1), nrm, part.astype(np.int16)


def inside(p):
    """Which part contains each point (N, 3) -> index in PART_NAMES, or -1."""
    q = np.floor((np.asarray(p, np.float64) - _ORG[None]) / _STEP).astype(np.int64)
    ok = ((q >= 0) & (q < np.array(_VOX.shape)[None])).all(1)
    out = np.full(len(q), -1, np.int32)
    out[ok] = _VOX[q[ok, 0], q[ok, 1], q[ok, 2]]
    return out


def core(name, rng):
    """A point well inside a part (a hero muon is sent through it)."""
    pts = _CORE[_CORE_PART == PART_NAMES.index(name)]
    return pts[rng.integers(0, len(pts))].astype(np.float64)


@functools.lru_cache(maxsize=1)
def _inside_voxels():
    idx = np.argwhere(_VOX >= 0)
    return idx, _VOX[idx[:, 0], idx[:, 1], idx[:, 2]]


def random_inside(rng, n):
    """n points anywhere in the body (uniform in volume) -> (points, part of each)."""
    idx, part = _inside_voxels()
    k = rng.integers(0, len(idx), n)
    return _ORG[None] + (idx[k] + rng.random((n, 3))) * _STEP, part[k].astype(np.int32)


def outline(rx, rz, gap=0.004):
    """Silhouette seen along a horizontal direction, (rx, rz) = the right vector of the view:
    -> (y, offset) of its points, every 1.2 cm of height: both ends of every run the figure covers."""
    proj = _O_XZ[:, 0] * rx + _O_XZ[:, 1] * rz
    lo, hi = np.minimum.reduceat(proj, _O_START), np.maximum.reduceat(proj, _O_START)
    order = np.lexsort((lo, _O_GLEV))
    ys, xs = [], []
    lev, a, b = -1, 0.0, 0.0
    for k in order:                         # the runs of the parts that touch are merged: one outline
        if _O_GLEV[k] != lev or lo[k] > b + gap:
            if lev >= 0:
                ys += [_O_Y[lev]] * 2
                xs += [a, b]
            lev, a, b = _O_GLEV[k], lo[k], hi[k]
        else:
            b = max(b, hi[k])
    ys += [_O_Y[lev]] * 2
    xs += [a, b]
    return np.array(ys), np.array(xs)


@functools.lru_cache(maxsize=512)
def front_spans(v):
    """Front silhouette at height v (0 = feet, 1 = top of the head) -> [(centre, half width)] in heights."""
    j = int(v * HEIGHT / _STEP)
    if not 0 <= j < _VOX.shape[1]:
        return []
    row = np.r_[False, (_VOX[:, j, :] >= 0).any(1), False]
    edges = np.nonzero(row[1:] != row[:-1])[0]
    x = (_ORG[0] + edges * _STEP) / HEIGHT
    return [(0.5 * (a + b), 0.5 * (b - a)) for a, b in zip(x[0::2], x[1::2])]
