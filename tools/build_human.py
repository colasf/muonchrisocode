"""Bake data/human.npz: the standing figure of YOU / FLOOD / OUTRO, from a real body mesh.

Source (data/human/, all CC0, https://github.com/makehumancommunity/makehuman, makehuman/data):
  makehuman_base.obj              3dobjs/base.obj              the MakeHuman base mesh (body + eyeballs used)
  makehuman_default.mhskel        rigs/default.mhskel          its skeleton
  makehuman_default_weights.mhw   rigs/default_weights.mhw     its skinning weights
  <ethnicity>-<gender>-young.target   targets/macrodetails/    the shapes MakeHuman blends into its default person

SHAPE blends the six young adult shapes in equal parts (nobody in particular: neither a man nor a woman).
The mesh stands in an A pose (arms out, feet apart): POSE brings the arms down along the body and the feet
under the hips with the skinning weights. The figure is scaled to 1.80 m, feet on y = 0, facing +z, its left
side on +x. Then everything the scenes need is baked once, so that nothing heavy runs at show time:

  verts, tris, vnrm, vpart   what is seen of the mesh (the inside of the mouth and the eye sockets are left
                             out, the eyeballs are in), vertex normals, body part of every vertex (PARTS)
  levels, ca, cb, clev       CT-like slice contours every 3 cm (segments, chained along each loop)
  o_xz, o_lev, o_part, ...   contour points every 1.2 cm, grouped by (level, part): the dotted silhouette
  vox, vox_org, vox_step     5 mm voxels: body part of each voxel inside the body, -1 outside (hit test)
  core_pts, core_part        points well inside each part (where a hero muon is sent)
  heart                      the heart

  python tools/build_human.py                 bake data/human.npz
  python tools/build_human.py --preview x.png also draw front / side / three-quarter views of the result
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "human"
OUT = ROOT / "data" / "human.npz"

HEIGHT = 1.80
PARTS = ["FOOT_L", "FOOT_R", "LEG_L", "LEG_R", "TORSO", "NECK", "HEAD", "ARM_L", "ARM_R", "HAND_L", "HAND_R"]
# the bone a part starts at (its descendants belong to it); anything else is TORSO
PART_ROOT = {"foot.L": "FOOT_L", "foot.R": "FOOT_R", "upperleg01.L": "LEG_L", "upperleg01.R": "LEG_R",
             "neck01": "NECK", "head": "HEAD", "upperarm01.L": "ARM_L", "upperarm01.R": "ARM_R",
             "wrist.L": "HAND_L", "wrist.R": "HAND_R"}

SHAPE = {f"{e}-{g}-young": 1.0 / 6.0 for e in ("caucasian", "african", "asian") for g in ("female", "male")}

# the pose, left side (mirrored for .R): bone -> (axis, degrees) in the rest frame, about the head of the bone.
# "hinge" = the axis of the elbow (upper arm x forearm).
POSE_L = {
    "clavicle.L": ((0.0, 0.0, 1.0), -3.0),          # the shoulders drop a little with the arms
    "upperarm01.L": ((0.0, 0.0, 1.0), -24.0),       # arms down along the body
    "lowerarm01.L": ("hinge", -30.0),               # the elbows almost straight
    "upperleg01.L": ((0.0, 0.0, 1.0), -4.6),        # feet under the hips
    "foot.L": ((0.0, 0.0, 1.0), 4.6),               # soles flat again
}

SLICE_STEP = 0.03
OUTLINE_STEP = 0.012
VOX = 0.005
EYES = ("helper-l-eye", "helper-r-eye")


# ----------------------------------------------------------------------------
# source
# ----------------------------------------------------------------------------

def load_obj(path):
    V, groups, cur = [], {}, None
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            cur = line.split()[1]
        elif line.startswith("f "):
            groups.setdefault(cur, []).append([int(p.split("/")[0]) - 1 for p in line.split()[1:]])
    return np.array(V), groups


def rot(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def tri(quads):
    q = np.asarray(quads)
    return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def outward(V, T):
    vol = (V[T[:, 0]] * np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6.0
    return T if vol > 0 else T[:, ::-1].copy()


def posed_mesh():
    """-> verts (N, 3) in metres (every vertex of the file), part of every vertex, triangles of the body and of
    the two eyeballs (each a closed shell, facing out)."""
    V, groups = load_obj(SRC / "makehuman_base.obj")
    for name, w in SHAPE.items():
        a = np.loadtxt(SRC / f"{name}.target")
        V[a[:, 0].astype(int)] += w * a[:, 1:]
    n = len(V)
    skel = json.load(open(SRC / "makehuman_default.mhskel"))
    wts = json.load(open(SRC / "makehuman_default_weights.mhw"))["weights"]
    bones = skel["bones"]
    head = {b: V[skel["joints"][bones[b]["head"]]].mean(0) for b in bones}

    order = []                                  # parents first
    def visit(b):
        if b in order:
            return
        if bones[b]["parent"]:
            visit(bones[b]["parent"])
        order.append(b)
    for b in bones:
        visit(b)

    hinge = np.cross(head["lowerarm01.L"] - head["upperarm01.L"], head["wrist.L"] - head["lowerarm01.L"])
    pose = {}
    for b, (axis, deg) in POSE_L.items():
        a = hinge if isinstance(axis, str) else np.asarray(axis, np.float64)
        pose[b] = rot(a, deg)
        pose[b[:-2] + ".R"] = rot((a[0], -a[1], -a[2]), deg)           # mirrored in x

    G = {}
    for b in order:
        M = np.eye(4)
        if b in pose:
            M[:3, :3] = pose[b]
            M[:3, 3] = head[b] - pose[b] @ head[b]
        p = bones[b]["parent"]
        G[b] = (G[p] @ M) if p else M

    def part_of(b):
        while b:
            if b in PART_ROOT:
                return PARTS.index(PART_ROOT[b])
            b = bones[b]["parent"]
        return PARTS.index("TORSO")

    P = np.zeros((n, 3))
    wsum = np.zeros(n)
    wpart = np.zeros((n, len(PARTS)))
    Vh = np.c_[V, np.ones(n)]
    for b, lst in wts.items():
        a = np.array(lst)
        idx, w = a[:, 0].astype(int), a[:, 1]
        P[idx] += w[:, None] * (Vh[idx] @ G[b].T)[:, :3]
        wsum[idx] += w
        wpart[idx, part_of(b)] += w
    body = tri(groups["body"])
    eyes = [tri(groups[g]) for g in EYES]
    used = np.unique(np.concatenate([body.ravel()] + [e.ravel() for e in eyes]))
    assert wsum[used].min() > 0.0, "a vertex without weights"
    P = np.where(wsum[:, None] > 0, P / np.maximum(wsum, 1e-9)[:, None], V)
    vpart = wpart.argmax(1).astype(np.int8)

    b0, b1 = P[np.unique(body), 1].min(), P[np.unique(body), 1].max()
    ankle = 0.5 * (head["foot.L"] + head["foot.R"])
    P = (P - np.array([0.0, b0, ankle[2] + 0.2])) * (HEIGHT / (b1 - b0))      # the ankles a little behind the axis
    return P, vpart, outward(P, body), [outward(P, e) for e in eyes]


def vertex_normals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------
# slices, voxels
# ----------------------------------------------------------------------------

def slice_mesh(V, T, vpart, y):
    """Mesh cut by the plane at height y -> (a, b, part): directed segments, chained loop after loop."""
    d = V[:, 1] - y
    d = np.where(d == 0.0, 1e-9, d)
    s = d[T] > 0
    cnt = s.sum(1)
    m = (cnt == 1) | (cnt == 2)
    if not m.any():
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int8)
    t, s = T[m], s[m]
    lone_up = s.sum(1) == 1
    k = np.where(lone_up, s.argmax(1), (~s).argmax(1))           # the vertex alone on its side
    r = np.arange(len(t))
    i0, i1, i2 = t[r, k], t[r, (k + 1) % 3], t[r, (k + 2) % 3]

    def cut(i, j):
        u = d[i] / (d[i] - d[j])
        return V[i] + (V[j] - V[i]) * u[:, None]

    key = lambda i, j: np.minimum(i, j).astype(np.int64) * len(V) + np.maximum(i, j)
    pa, pb, ka, kb = cut(i0, i1), cut(i0, i2), key(i0, i1), key(i0, i2)
    # one direction around the loop whatever the triangle: swap the ones whose lone vertex is below
    sw = ~lone_up
    A = np.where(sw[:, None], pb, pa)
    B = np.where(sw[:, None], pa, pb)
    KA, KB = np.where(sw, kb, ka), np.where(sw, ka, kb)
    part = vpart[i0]
    start = {int(kk): n for n, kk in enumerate(KA)}
    ends = set(int(kk) for kk in KB)
    seen = np.zeros(len(A), bool)
    chain = []
    # open runs first, from their first segment (where the hidden triangles were taken out), then the loops
    firsts = [n for n in range(len(A)) if int(KA[n]) not in ends]
    for n0 in firsts + list(range(len(A))):
        n = n0
        while not seen[n]:
            seen[n] = True
            chain.append(n)
            n = start.get(int(KB[n]), n)
    chain = np.array(chain)
    return A[chain], B[chain], part[chain]


def solid(V, T, lo, shape, step):
    """Voxels inside the closed mesh (V, T): parity of the crossings along z, slice by slice."""
    nx, ny, nz = shape
    xs = lo[0] + (np.arange(nx) + 0.5) * step
    zs = lo[2] + (np.arange(nz) + 0.5) * step
    none = np.zeros(len(V), np.int8)
    out = np.zeros(shape, bool)
    for j in range(ny):
        a, b, _ = slice_mesh(V, T, none, lo[1] + (j + 0.5) * step + 1.3e-7)
        if not len(a):
            continue
        # crossings of every segment with the lines x = xs (half-open, so that a shared end counts once)
        x0, x1 = a[:, 0][:, None], b[:, 0][:, None]
        cross = (x0 <= xs[None]) != (x1 <= xs[None])
        u = (xs[None] - x0) / np.where(x1 == x0, 1.0, x1 - x0)
        zc = np.where(cross, a[:, 2][:, None] + (b[:, 2] - a[:, 2])[:, None] * u, np.inf)
        out[:, j, :] = ((zc[:, :, None] < zs[None, None, :]).sum(0) % 2) == 1
    return out


def grid_for(V, step, y0=None):
    lo = V.min(0) - 3 * step
    if y0 is not None:
        lo[1] = y0
    hi = V.max(0) + 3 * step
    return lo, tuple(int(math.ceil((hi[k] - lo[k]) / step)) for k in range(3))


def label(ins, lo, step, V, vpart):
    """Part of every voxel inside: the part of the nearest vertex."""
    vox = np.full(ins.shape, -1, np.int8)
    for j in range(ins.shape[1]):
        ix, iz = np.nonzero(ins[:, j, :])
        if not len(ix):
            continue
        y = lo[1] + (j + 0.5) * step
        near = np.nonzero(np.abs(V[:, 1] - y) < 0.03)[0]
        q = np.stack([lo[0] + (ix + 0.5) * step, np.full(len(ix), y), lo[2] + (iz + 0.5) * step], 1)
        lab = np.empty(len(q), np.int8)
        for c in range(0, len(q), 2000):
            dd = ((q[c:c + 2000, None, :] - V[near][None]) ** 2).sum(2)
            lab[c:c + 2000] = vpart[near[dd.argmin(1)]]
        vox[ix, j, iz] = lab
    return vox


def hidden_vertices(V, vn, body, eyes, y_min, step=0.002, reach=0.15, rays=24):
    """Vertices above y_min that nothing outside can see (inside of the mouth, eye sockets behind the eyeballs,
    the back of the eyeballs): every direction of the hemisphere around their normal runs into the head."""
    sel = np.nonzero(V[:, 1] > y_min)[0]
    sel = sel[np.isin(sel, np.unique(np.concatenate([body.ravel()] + [e.ravel() for e in eyes])))]
    lo, shape = grid_for(V[sel], step)
    ins = solid(V, body, lo, shape, step)
    for e in eyes:
        ins |= solid(V, e, lo, shape, step)
    rng = np.random.default_rng(5)
    P, N = V[sel], vn[sel]
    t1 = np.cross(N, np.where(np.abs(N[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(N, t1)
    blocked = np.zeros(len(sel))
    for _ in range(rays):
        r, a = math.sqrt(rng.uniform(0, 0.75)), rng.uniform(0, 2 * math.pi)       # up to 60 degrees off the normal
        d = N * math.sqrt(1 - r * r) + t1 * (r * math.cos(a)) + t2 * (r * math.sin(a))
        hit = np.zeros(len(sel), bool)
        for s in np.arange(0.0, reach, step):
            q = np.floor((P + N * 0.003 + d * s - lo[None]) / step).astype(int)
            ok = ((q >= 0) & (q < np.array(shape)[None])).all(1)
            hit[ok] |= ins[q[ok, 0], q[ok, 1], q[ok, 2]]
        blocked += hit
    hid = np.zeros(len(V), bool)
    hid[sel[blocked >= 0.9 * rays]] = True
    return hid


def erode(m):
    e = m.copy()
    e[1:] &= m[:-1]; e[:-1] &= m[1:]
    e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
    e[:, :, 1:] &= m[:, :, :-1]; e[:, :, :-1] &= m[:, :, 1:]
    e[0] = e[-1] = False
    e[:, 0] = e[:, -1] = False
    e[:, :, 0] = e[:, :, -1] = False
    return e


# ----------------------------------------------------------------------------

def bake():
    V, vpart, body, eyes = posed_mesh()
    bv = np.unique(body)
    vol = (V[body[:, 0]] * np.cross(V[body[:, 1]], V[body[:, 2]])).sum() / 6.0
    print(f"body  {len(bv)} vertices, {len(body)} triangles, volume {vol * 1000:.1f} l, "
          f"x {V[bv, 0].min():+.3f} {V[bv, 0].max():+.3f}  z {V[bv, 2].min():+.3f} {V[bv, 2].max():+.3f}")

    # what is seen: body + eyeballs, without what nothing outside can see
    T = np.concatenate([body] + eyes)
    vn = np.zeros_like(V)
    for t in [body] + eyes:
        used = np.unique(t)
        vn[used] = vertex_normals(V, t)[used]
    hid = hidden_vertices(V, vn, body, eyes, y_min=V[bv][vpart[bv] == PARTS.index("NECK"), 1].min())
    keep = ~hid[T].all(1)
    print(f"seen  {int(keep.sum())} of {len(T)} triangles ({int(hid.sum())} hidden vertices in the head)")
    T = T[keep]
    used = np.unique(T)                         # compact: only the vertices that are drawn
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    Vs, Ts, ns, ps = V[used], remap[T], vn[used], vpart[used]

    levels = np.arange(0.5 * SLICE_STEP, HEIGHT, SLICE_STEP)
    ca, cb, clev = [], [], []
    for k, y in enumerate(levels):
        a, b, _ = slice_mesh(Vs, Ts, ps, y + 1.3e-7)
        ca.append(a); cb.append(b); clev.append(np.full(len(a), k, np.int32))
    ca, cb, clev = np.concatenate(ca), np.concatenate(cb), np.concatenate(clev)
    print(f"slices  {len(levels)} levels, {len(ca)} segments")

    olv = np.arange(0.5 * OUTLINE_STEP, HEIGHT, OUTLINE_STEP)
    ox, ol, op = [], [], []
    for k, y in enumerate(olv):
        a, _, p = slice_mesh(V, body, vpart, y + 1.3e-7)
        ox.append(a[:, [0, 2]]); ol.append(np.full(len(a), k, np.int32)); op.append(p)
    ox, ol, op = np.concatenate(ox), np.concatenate(ol), np.concatenate(op)
    order = np.lexsort((op, ol))                # grouped by (level, part)
    ox, ol, op = ox[order], ol[order], op[order]
    gid = ol.astype(np.int64) * len(PARTS) + op
    o_start = np.nonzero(np.r_[True, gid[1:] != gid[:-1]])[0]
    print(f"outline  {len(olv)} levels, {len(ox)} points, {len(o_start)} groups")

    org, shape = grid_for(V[bv], VOX, y0=0.0)
    ins = solid(V, body, org, shape, VOX)
    vox = label(ins, org, VOX, V[bv], vpart[bv])
    print(f"voxels  {vox.shape}, {ins.sum()} inside = {ins.sum() * VOX ** 3 * 1000:.1f} l")
    deep1 = erode(ins)
    deep3 = erode(erode(deep1))
    rng = np.random.default_rng(7)
    core_pts, core_part = [], []
    for pi, name in enumerate(PARTS):
        ys = V[bv][vpart[bv] == pi, 1]
        y0, y1 = ys.min() + 0.25 * (ys.max() - ys.min()), ys.max() - 0.25 * (ys.max() - ys.min())
        for deep in (deep3, deep1, ins):
            idx = np.argwhere(deep & (vox == pi))
            yy = org[1] + (idx[:, 1] + 0.5) * VOX
            idx = idx[(yy >= y0) & (yy <= y1)]
            if len(idx) >= 40:
                break
        idx = idx[rng.permutation(len(idx))[:3000]]
        core_pts.append(org[None] + (idx + 0.5) * VOX)
        core_part.append(np.full(len(idx), pi, np.int8))
        print(f"  {name:<7} {int((ps == pi).sum()):5d} vertices  {int((vox == pi).sum()):7d} voxels  "
              f"{len(idx):5d} core points  y {ys.min():.3f} .. {ys.max():.3f}")
    core_pts, core_part = np.concatenate(core_pts), np.concatenate(core_part)

    # the heart: behind the sternum at 71 % of the height, a little to the left of the figure and forward
    yh = 0.71 * HEIGHT
    j = int((yh - org[1]) / VOX)
    ix, iz = np.nonzero(vox[:, j, :] == PARTS.index("TORSO"))
    heart = np.array([0.04, yh, org[2] + (iz.mean() + 0.5) * VOX + 0.02])
    print(f"heart  {heart.round(3)}  (chest at that height: x {org[0] + ix.min() * VOX:+.3f} .. "
          f"{org[0] + (ix.max() + 1) * VOX:+.3f}  z {org[2] + iz.min() * VOX:+.3f} .. {org[2] + (iz.max() + 1) * VOX:+.3f})")

    np.savez_compressed(
        OUT, parts=np.array(PARTS), height=HEIGHT, heart=heart,
        verts=Vs.astype(np.float32), tris=Ts.astype(np.int32), vnrm=ns.astype(np.float32), vpart=ps,
        levels=levels, ca=ca.astype(np.float32), cb=cb.astype(np.float32), clev=clev,
        o_levels=olv, o_xz=ox.astype(np.float32), o_lev=ol, o_part=op, o_start=o_start.astype(np.int32),
        vox=vox, vox_org=org, vox_step=VOX, core_pts=core_pts.astype(np.float32), core_part=core_part)
    print(f"-> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


def preview(path):
    """Front, side and three-quarter views of the baked figure: rim-lit cloud + slices, as the scenes draw it."""
    from PIL import Image
    d = np.load(OUT)
    V, T, vn = d["verts"], d["tris"], d["vnrm"]
    rng = np.random.default_rng(1)
    area = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    f = rng.choice(len(T), 60000, p=area / area.sum())
    u, v = rng.random(len(f)), rng.random(len(f))
    sw = u + v > 1
    u, v = np.where(sw, 1 - u, u), np.where(sw, 1 - v, v)
    w = np.stack([1 - u - v, u, v], 1)
    P = (V[T[f]] * w[:, :, None]).sum(1)
    N = (vn[T[f]] * w[:, :, None]).sum(1)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    S, W, H = 620.0, 760, 1240
    img = np.zeros((H, W * 3), np.float32)
    for k, yaw in enumerate((0.0, 90.0, 35.0)):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        right, fwd = np.array([c, 0, -s]), np.array([s, 0, c])
        def scr(p):
            return k * W + W / 2 + S * (p @ right), H - 60 - S * p[:, 1]
        x, y = scr(P)
        inten = 0.10 + 0.9 * (1 - np.abs(N @ fwd)) ** 2
        np.add.at(img, (y.astype(int), x.astype(int)), inten)
        for q in np.linspace(0, 1, 12):
            x, y = scr(d["ca"] + (d["cb"] - d["ca"]) * q)
            np.add.at(img, (y.astype(int), x.astype(int)), 0.12)
    Image.fromarray((np.clip(img, 0, 1) ** 0.6 * 255).astype(np.uint8)).save(path)
    print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", default=None, help="PNG of the baked figure (front, side, three-quarter)")
    a = ap.parse_args()
    bake()
    if a.preview:
        preview(a.preview)
