"""YOU - muons pass through you.   Sheet 1.3, 01:12.4 - 01:41.0 (the "hero WOW moment").

About one muon per square centimetre per minute reaches the ground: ~70 go through a spectator
every second (showdata.RATE_YOU), 4 200 a minute, more than fifty thousand during the show (the voice of
scene 5).

A 1.80 m body (a real mesh, see muonbloom/human.py): a rim-lit point cloud with CT-like slice
contours over a dot-lattice floor. Cosmic muons (red) rain through it on the 16th-note grid of
the music (90 BPM here); each crossing is hit-tested: the part lights up, the slice at that
height turns red, the energy left behind is logged.

Linear, cut on the bars of the music (2.667 s):
  01:12.48  "YOU"        the muon tagged in the air shower comes straight down through the HEART;
                         on the drum roll the body assembles, slice by slice, around that hit
  01:14.64  the drop     PERSPECTIVE orbit, the whole instrument switches on, the rain starts
  01:19.97               ORTHO_FRONT / ORTHO_SIDE, a red scan slice sweeping
  01:25.30  "An echo..." THORAX close-up, a second muon through the heart: the one we follow
  01:27.97  "...through your bones, your cells, ancient, indifferent": powers of ten along that
            track, one per bar: BONE (cm) - CELLS (um) - DNA (nm) - WATER (0.1 nm)
  01:38.64  "It is called...": nothing left but the straight red track and what was measured

LAYOUT. The three towers stand in front of the wall for the whole show (dark until 01:44) and
nobody knows yet where: nothing here has a fixed x. `Plan` reads the tower placement from ctx:
  focus bay   the figure, the heart hit, the track we follow and their tags (ctx.focus)
  title col   YOU, the parameters, the hit log         (leftmost free column)
  side col    the second view (ORTHO_SIDE) / the read-outs of the zoom   (widest free column)
  data col    energy left in each part of the body     (what remains)
  bottom      counts on every time scale, hit barcode  (ctx.slots_pre panels)
Only textures (floor lattice, rain, tissue, cells, molecules) run behind the towers. A column
that does not exist with a given placement simply drops its block.

NOTHING FADES OR POPS in the read-outs (the body, the rain, the tissues are the image: they cut).
  before the drop   the ACQUIRING line is decoded, two pens trace its bar, a block opens per 16th note
  the drop (BAR0)   the whole instrument is constructed in about 1.5 s (Frame.build, staggered): title
                    tag pushed out, parameters and hit log decoded, panels drawn, strip written
  every hit         its callout is constructed outwards from the hit (leader, plate, tag, lines) and taken
                    apart the same way when it leaves; its log line is decoded; its tag enters the strip
  every cut         what is new in the view is made from the cut: view tag, view name, scale bar, range
                    rings, scan slice, the ticks of the track, the read-out plates of the zoom, the medium
                    rows of the title column; the scale strip replaces the hit timeline at the first micro
                    view and its cursor runs down the powers of ten at every cut
`callout` (age / life) and `plate` are the two helpers; `_cut(t)` gives the time of the last cut.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import engine as E
from .. import hud
from .. import human
from .. import layout as L
from .. import showdata as sd
from ..engine import Camera, OrthoCamera, hash01, smoothstep, text_w
from ..show import Scene

# -- music grid of this section (found on the kicks of the music stem) --------------------
BAR = 8.0 / 3.0                 # 2.667 s: 4 beats at 90 BPM
STEP = BAR / 16.0
BAR0 = 74.64                    # the drop
T_IN, T_OUT = 72.4, 101.0
CUTS = [BAR0 + 2 * BAR, BAR0 + 4 * BAR, BAR0 + 5 * BAR, BAR0 + 6 * BAR, BAR0 + 7 * BAR, BAR0 + 8 * BAR,
        BAR0 + 9 * BAR]         # ortho, thorax, bone, cells, dna, atoms, track

# -- vertical layout (x comes from the towers, see Plan) -------------------------------------
Y_TOP, Y_BOT = 240.0, 1190.0
Y_MID = 715.0
WALL = (L.FX0 + 4.0, Y_TOP, L.FX1 - 4.0, Y_BOT)

DEDX = 0.2                      # GeV per metre of tissue (~2 MeV/cm)
RATE = sd.RATE_YOU              # muons / s through one spectator

PART_NAMES = human.PART_NAMES   # FOOT_L .. HAND_R (the figure faces +z, its left side is +x)
HEART = human.HEART


def spaced(n):
    """12 345 678 (thin thousands separator, as on the counter cell)."""
    return f"{int(n):,}".replace(",", " ")


def width(c):
    return c[1] - c[0]


# ----------------------------------------------------------------------------
# where things go, for whatever tower placement
# ----------------------------------------------------------------------------

class Plan:
    """Columns of this look, all derived from the tower placement in ctx."""

    def __init__(self, ctx, panels="slots_pre"):
        self.fx, self.fy = float(ctx.focus[0]), Y_MID
        self.bays = [tuple(b) for b in ctx.bays]
        self.fbay = tuple(ctx.focus_bay)
        cols = [tuple(c) for c in ctx.cols]
        inside = [c for c in cols if c[0] - 1 <= self.fx <= c[1] + 1]
        self.fcol = inside[0] if inside else (self.fbay[0] + 28.0, self.fbay[1] - 28.0)
        self.half = min(self.fx - self.fbay[0], self.fbay[1] - self.fx)      # free half-width around the focus
        others = [c for c in cols if c != self.fcol]
        cand = [c for c in others if width(c) >= 255.0]
        self.title = min(cand, key=lambda c: c[0]) if cand else None
        rest = [c for c in others if c != self.title]
        wide = [c for c in rest if width(c) >= 300.0]
        self.side = max(wide, key=lambda c: width(c) - 0.3 * abs(0.5 * (c[0] + c[1]) - self.fx)) if wide else None
        rest = [c for c in rest if c != self.side]
        cand = [c for c in rest if width(c) >= 250.0]
        self.data = max(cand, key=width) if cand else None
        sl = getattr(ctx, panels)
        self.panels = sorted([tuple(p) for p in sl["panels"]], key=width, reverse=True)
        self.py0, self.py1 = sl["y0"], sl["y1"]
        # the field window: focus bay + side bay (and the tower between them). Textures live there only.
        sb = self.bay_of(self.side) if self.side else self.fbay
        self.win = (min(self.fbay[0], sb[0]), max(self.fbay[1], sb[1]))

    def plates(self, f, y0=Y_TOP - 4.0, y1=Y_BOT + 8.0):
        """Black out what is not the field window: the title and data columns are read on black."""
        for a, b in ((L.FX0 + 2.0, self.win[0]), (self.win[1], L.FX1 - 2.0)):
            if b - a > 2.0:
                f.occlude(a, y0, b, y1)
        for col in (self.title, self.data):
            if col is not None and self.win[0] < 0.5 * (col[0] + col[1]) < self.win[1]:
                b = self.bay_of(col)
                f.occlude(b[0], y0, b[1], y1)

    def bay_of(self, col):
        c = 0.5 * (col[0] + col[1])
        for b in self.bays:
            if b[0] <= c <= b[1]:
                return b
        return col


def callout(f, x, y, title, lines=(), col=None, prefer=1, dy=-50.0, red=False, age=9.0, life=None, size=L.T_TAG,
            lsize=L.T_SMALL, elbow=40.0):
    """hud.callout on a black plate, kept inside `col`. It goes on the `prefer` side (outwards, away from
    what it points at); when the block does not fit there it first drops its data lines, then shortens
    its leader, and only then changes side.
    It never fades: it is constructed outwards from its point (age = seconds since it appeared: leader drawn
    by a pen, plate opened, tag pushed out, lines decoded) and, when it has a `life` (seconds it stays), taken
    apart the same way at the end."""
    build = B.io(age, None if life is None else life - age, out=0.3, span=0.7)
    if build < 0:
        return
    lo, hi = col if col else (L.COL_X0, L.COL_X1)

    def block(ls):
        return max([text_w(title, size) + 12] + [text_w(ln, lsize) for ln in ls]) + 10

    def fits(side, ls, el):
        return (x + el + 44 + block(ls) <= hi) if side > 0 else (x - el - 44 - block(ls) >= lo)

    lines = list(lines)
    tries = [(prefer, lines, elbow), (prefer, [], elbow), (prefer, [], 14.0), (-prefer, lines, elbow),
             (-prefer, [], elbow), (-prefer, [], 14.0)]
    side, lines, elbow = next((tr for tr in tries if fits(*tr)), (1 if hi - x >= x - lo else -1, [], 14.0))
    w = block(lines)
    tx = x + side * (elbow + 44)
    y0 = y + dy - size * 0.78
    y1 = y + dy + size * 0.36 + len(lines) * lsize * 1.45 + (16 if lines else 8)
    wp = (w + 8) * float(B.ease((build - 0.1) / 0.25))   # the plate opens away from the leader, with the tag
    if wp > 0.0:
        f.occlude(min(tx - side * 8, tx + side * wp), y0, max(tx - side * 8, tx + side * wp), y1)
    hud.callout(f, x, y, side * elbow, dy, title, lines, red=red, size=size, lsize=lsize, side=side, build=build)


def plate(f, rect, age, dur=0.25, flow="tb"):
    """The black plate under a read-out, opened instead of popped: it grows from its top edge ('tb'), from its
    left ('lr') or from its right ('rl') while the read-out is being made. age = seconds since the read-out
    started to build (None = built, negative = not there: no plate)."""
    if age is not None and age < 0.0:
        return
    g = 1.0 if age is None else float(B.ease(age / dur))
    x0, y0, x1, y1 = rect
    if flow == "tb":
        f.occlude(x0, y0, x1, y0 + (y1 - y0) * g)
    elif flow == "lr":
        f.occlude(x0, y0, x0 + (x1 - x0) * g, y1)
    else:
        f.occlude(x1 - (x1 - x0) * g, y0, x1, y1)


def fit_text(options, room, size):
    """First of `options` (long to short) that fits in `room` px at `size`."""
    for s in options:
        if text_w(s, size) + 14 <= room:
            return s
    return options[-1]


# ----------------------------------------------------------------------------
# the body (shared with the flood scene)
# ----------------------------------------------------------------------------

class Body:
    """Point cloud + slice contours of a standing figure, with an inside test for the muons."""

    def __init__(self, seed=33, density=16000.0, floor=3.2):
        rng = np.random.default_rng(seed)
        self.pts, self.nrm, self.pts_part = human.cloud(rng, density)
        self.rnd = rng.random(len(self.pts)).astype(np.float32)
        self.drift = rng.normal(0, 1, (len(self.pts), 3)).astype(np.float32)
        # CT-like slice contours every 3 cm
        self.levels, self.ca, self.cb, self.clev = human.LEVELS, human.CA, human.CB, human.CLEV
        g = np.arange(-floor, floor + 1e-3, 0.1)
        X, Z = np.meshgrid(g, g)
        self.lattice = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1).astype(np.float32)
        self.lat_major = ((np.abs(np.round(X * 10)) % 5 == 0) & (np.abs(np.round(Z * 10)) % 5 == 0)).ravel()
        self.lat_even = ((np.abs(np.round(X * 10)) % 2 == 0) & (np.abs(np.round(Z * 10)) % 2 == 0)).ravel() & ~self.lat_major
        self.lev_wall = self.clev % 2 == 0                  # the wall rule: one slice in two (6 cm), full level
        self.lev_wall_top = self.clev % 10 == 0             # ... one in ten seen from above (they pile up)

    def inside(self, p):
        """Which part contains each point (N, 3) -> index or -1."""
        return human.inside(p)

    def target_in(self, name, rng):
        if name == "HEART":
            return HEART + rng.normal(0, 0.012, 3)
        return human.core(name, rng)

    def muon(self, rng, t_hit, tgt, hero, zen_sigma=0.38, speed=22.0, zen=None, top=2.9):
        """A straight track through `tgt`; t_hit = time its head reaches the target."""
        th = min(1.0, abs(rng.normal(0, zen_sigma))) if zen is None else zen
        ph = rng.uniform(0, 2 * np.pi)
        d = np.array([math.sin(th) * math.cos(ph), -math.cos(th), math.sin(th) * math.sin(ph)])
        s_top = (top - tgt[1]) / math.cos(th)
        s_bot = tgt[1] / math.cos(th)
        a = tgt - d * s_top
        b = tgt + d * s_bot
        Ln = s_top + s_bot
        u = np.linspace(0, 1, 700)
        pts = a[None] + (b - a)[None] * u[:, None]
        ins = self.inside(pts)
        inside = ins >= 0
        entry = int(np.argmax(inside)) if inside.any() else -1
        part, hit, dE, u_in = None, None, 0.0, None
        if entry >= 0:
            dE = inside.sum() * (Ln / 700) * DEDX
            if hero:        # tag the organ it was sent through, where it crosses it
                hit = tgt
                u_in = s_top / Ln
                k = self.inside(tgt[None])[0]
                part = "HEART" if np.linalg.norm(tgt - HEART) < 0.05 else (PART_NAMES[k] if k >= 0 else "TORSO")
            else:
                hit = pts[entry]
                part = PART_NAMES[ins[entry]]
                if np.linalg.norm(hit - HEART) < 0.085:
                    part = "HEART"
                u_in = u[entry]
        edges = np.diff(np.r_[0, inside.astype(int), 0])
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0] - 1
        runs = list(zip(u[starts], u[ends]))
        dur = Ln / speed
        t0 = t_hit - dur * (u_in if u_in is not None else s_top / Ln)
        return dict(t=t0, a=a.astype(np.float32), b=b.astype(np.float32), L=Ln, dur=dur, part=part, hit=hit,
                    u_in=u_in, dE=dE, E=float(np.exp(rng.normal(1.2, 0.7))), runs=runs, hero=hero,
                    charge=rng.choice(["+", "-"]), t_hit=t_hit if hit is not None else None)

    # ------------------------------------------------------------------ drawing
    def draw_floor(self, f, cam, gain=1.0):
        sx, sy, z, ok = cam.project(self.lattice)
        fog = np.clip(1.4 - z / 9.0, 0.3, 1.0) if not cam.ortho else 1.0
        if E.WALL:              # the wall rule: single pixels do not land - one point in two each way, as dots that
            m = ok & self.lat_even          # get smaller with the distance (full level), the majors keep their cross
            f.dots("w", sx[m], sy[m], 1.4 + 1.0 * np.broadcast_to(fog, sx.shape)[m], min(gain, 1.0))
        else:
            f.pixels("w", sx[ok], sy[ok], (np.where(self.lat_major, 0.9, 0.4) * fog * gain)[ok])
        mj = self.lat_major & ok
        f.crosses("w", sx[mj], sy[mj], 4.5, 0.5 * gain)

    def draw(self, f, cam, hot_pts=(), hot_lev=None, gain=1.0, cloud=1.0, slices=1.0, top=False, reveal=None,
             dissolve=0.0, t=0.0, outline=0.0):
        """hot_pts: [(xyz, strength)] red glow on the cloud; hot_lev: per-level red amount (len(levels));
        reveal: (y_centre, half-height) only the slices inside are there (the body assembling);
        dissolve 0..1: the cloud drifts away and thins out; outline: dotted silhouette left behind."""
        if cloud > 0.01:
            P = self.pts
            keep = np.ones(len(P), bool)
            if dissolve > 0:
                keep = self.rnd > dissolve ** 0.7
                P = P + self.drift * (0.55 * dissolve ** 1.5) + np.array([0.0, 0.25 * dissolve ** 2, 0.0], np.float32)
            px, py, pz, pok = cam.project(P)
            if cam.ortho:
                facing = np.abs(self.nrm @ (-cam.R[2]))
            else:
                v = cam.pos.astype(np.float32)[None] - self.pts
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                facing = np.abs((self.nrm * v).sum(1))
            if E.WALL:          # the wall rule: no haze over the body - the cloud is the rim of the figure, full level
                inten = np.clip(2.2 * (1 - facing) ** 2 - 0.25, 0.0, 1.0) * min(cloud * gain, 1.0) * (1.0 - 0.6 * dissolve)
            else:
                inten = (0.16 + 0.78 * (1 - facing) ** 2) * cloud * gain * (1.0 - 0.6 * dissolve)
            if top:
                inten = inten * (0.0 if E.WALL else 0.14)       # (seen from above the rim piles up into a blob)
            red = np.zeros(len(P), np.float32)
            for hp, strength in hot_pts:
                d = np.linalg.norm(self.pts - np.asarray(hp, np.float32)[None], axis=1)
                red = np.maximum(red, np.exp(-(d / 0.07) ** 2) * strength)
            m = pok & keep
            f.pixels("w", px[m], py[m], (inten * (1 - 0.8 * red))[m])
            rm = m & (red > 0.05)
            if rm.any():
                f.pixels("r", px[rm], py[rm], 2.4 * red[rm] * gain)
        if slices > 0.01:
            ax, ay, az, aok = cam.project(self.ca)
            bx, by, bz, bok = cam.project(self.cb)
            okc = aok & bok
            ci = np.full(len(self.ca), (0.27 if not top else 0.09) * slices * gain, np.float32)
            if E.WALL:          # one slice in two at full level, black between them
                ci = np.where(self.lev_wall_top if top else self.lev_wall, min(slices * gain, 1.0), 0.0).astype(np.float32)
            if reveal is not None:
                yc, hh = reveal
                ylev = self.levels[self.clev]
                edge = np.clip((hh - np.abs(ylev - yc)) / 0.06, 0.0, 1.0)
                ci = ci * edge * (1.0 + 1.6 * np.clip(1.0 - (hh - np.abs(ylev - yc)) / 0.12, 0.0, 1.0))
            if dissolve > 0:
                ci = ci * np.where(hash01(self.clev, np.arange(len(self.clev)) // 5, 3) > dissolve * 1.15, 1.0, 0.0)
            hot = np.zeros(len(self.ca), np.float32) if hot_lev is None else np.asarray(hot_lev, np.float32)[self.clev]
            if E.WALL:
                okc = okc & (ci > 0.0)
            f.segments("w", ax[okc], ay[okc], bx[okc], by[okc], (ci * (1 - hot))[okc], width=E.ww(1.3))
            hm = okc & (hot > 0.02) & (ci > 0.001)
            if hm.any():
                f.segments("r", ax[hm], ay[hm], bx[hm], by[hm], 1.4 * hot[hm] * gain, width=1.8)
        if outline > 0.01 and cam.ortho:
            self._outline(f, cam, outline)

    def _outline(self, f, cam, alpha):
        """Dotted silhouette: for every height the two ends of every run the figure covers (ortho views)."""
        r_, u_ = cam.R[0].astype(np.float64), cam.R[1].astype(np.float64)
        ys, off = human.outline(float(r_[0]), float(r_[2]))
        sx = cam.cx + cam.scale * (off + ys * r_[1] - float(cam.pos @ r_))
        sy = cam.cy - cam.scale * (ys * u_[1] - float(cam.pos @ u_))
        f.dots("w", sx, sy, E.ww(1.5), E.wl(0.75 * alpha))


def draw_muon(f, cam, m, a, col=None, gain=1.0, tag=True, lines=True):
    """One muon track at age `a` (s since it entered the view): head, ionisation runs, hit ring, data tag."""
    prog = min(1.0, a / m["dur"])
    fade = 1.0 if a < m["dur"] else math.exp(-(a - m["dur"]) / 0.3)
    A, B = m["a"], m["b"]
    head = A + (B - A) * prog
    sx, sy, z, ok = cam.project(np.stack([A, head]).astype(np.float32))
    if not ok.all():
        return
    hero = m["hero"]
    f.segments("r", sx[:1], sy[:1], sx[1:], sy[1:], (0.8 if hero or E.WALL else 0.55) * fade * gain,
               width=1.9 if hero else E.ww(1.3))
    for (u0, u1) in m["runs"]:
        if u0 >= prog:
            continue
        u1 = min(u1, prog)
        qx, qy, _, _ = cam.project(np.stack([A + (B - A) * u0, A + (B - A) * u1]).astype(np.float32))
        f.segments("r", qx[:1], qy[:1], qx[1:], qy[1:], 1.3 * fade * gain, width=3.0 if hero else 2.0)
    if prog < 1.0:
        f.dots("r", sx[1:], sy[1:], 3.6, 1.6 * gain)
        f.dots("w", sx[1:], sy[1:], 1.4, 1.0 * gain)
    if m["hit"] is None:
        return
    ah = a - m["dur"] * m["u_in"]
    if 0 <= ah < 0.8:
        hx, hy, _, _ = cam.project(m["hit"][None].astype(np.float32))
        u = ah / 0.8
        f.rings("r", [hx[0]], [hy[0]], [8 + 78 * (1 - (1 - u) ** 3)], 1.0 * (1 - u) ** 1.5 * gain, width=1.8)
        f.dots("w", [hx[0]], [hy[0]], 3.4 * (1 - u) + 1.2, 1.6 * (1 - u) * gain)
    if tag and hero and 0 <= ah < 2.4:
        hx, hy, _, _ = cam.project(m["hit"][None].astype(np.float32))
        x, y = float(hx[0]), float(hy[0])
        callout(f, x, y, m["part"], [f"MU{m['charge']} {m['E']:.3f} GEV", f"DE {m['dE'] * 1000:.1f} MEV"] if lines
                else [], col=col, prefer=1 if x >= cam.cx else -1, red=m["part"] == "HEART", age=ah, life=2.4)


# ----------------------------------------------------------------------------
# small 2D helpers for the micro views
# ----------------------------------------------------------------------------

def _jit(i, j, seed, amp):
    return (hash01(i, j, seed) - 0.5) * 2 * amp, (hash01(i, j, seed + 1) - 0.5) * 2 * amp


def _circ(ax, ay, bx, by, cx, cy):
    """Circumcentres of triangles (vectorised)."""
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy


class You(Scene):
    name = "you"

    def __init__(self, ctx, seed=33):
        super().__init__(ctx)
        self.P = Plan(ctx)
        self.t_you = sd.said("YOU", 72.483)
        self.t_echo = sd.said("An echo", 85.2)
        self.t_called = sd.said("It is called", 97.883)
        self.body = Body(seed)
        rng = np.random.default_rng(seed + 1)
        self._build_muons(rng)
        self._build_micro(np.random.default_rng(seed + 2))

    # ----------------------------------------------------------------- muons
    def _build_muons(self, rng):
        b = self.body
        mus = []
        # the one from the shower: straight down through the heart on the word
        mus.append(b.muon(rng, self.t_you, HEART.copy(), True, zen=0.05, speed=30.0))
        heroes = [(BAR0 + 0.06, "HEAD"), (BAR0 + 1.06, "HAND_L"), (BAR0 + BAR, "TORSO"), (BAR0 + BAR + 1.0, "LEG_R"),
                  (CUTS[0] + 0.1, "HEART"), (CUTS[0] + 1.0, "NECK"), (CUTS[0] + BAR, "HAND_R"),
                  (CUTS[0] + BAR + 1.0, "FOOT_L")]
        for th, name in heroes:
            mus.append(b.muon(rng, th, b.target_in(name, rng), True))
        # the one we follow down the scales: through the heart, on the bar of "An echo, from a distant past"
        self.echo = b.muon(rng, CUTS[1] + 0.12, HEART + np.array([0.0, 0.0, 0.0]), True, zen=0.16, speed=14.0)
        mus.append(self.echo)
        hero_t = np.array([m["t_hit"] for m in mus])
        k = 0
        while BAR0 + k * STEP < T_OUT:
            tk = BAR0 + k * STEP
            k += 1
            dens = 0.58 if tk < CUTS[2] else 0.4
            if rng.random() > dens or np.abs(hero_t - tk).min() < 0.09:
                continue
            if rng.random() < 0.55:
                tgt = b.target_in(PART_NAMES[rng.integers(0, len(PART_NAMES))], rng)
            else:
                a, r = rng.uniform(0, 2 * np.pi), 0.9 * math.sqrt(rng.random())
                tgt = np.array([r * math.cos(a), rng.uniform(0.3, 1.6), r * math.sin(a)])
            mus.append(b.muon(rng, tk, tgt, False))
        self.mus = mus
        self.mu_t = np.array([m["t"] for m in mus])
        # the rest of the rain: it falls on the whole wall, towers included (not hit-tested, just seen)
        n = int(95 * (T_OUT - BAR0))
        tr = np.sort(rng.uniform(BAR0, T_OUT, n))
        rad, az = 0.42 + 2.9 * np.sqrt(rng.random(n)), rng.uniform(0, 2 * np.pi, n)
        G = np.stack([rad * np.cos(az), np.zeros(n), rad * np.sin(az)], 1)
        th, ph = np.minimum(np.abs(rng.normal(0, 0.3, n)), 0.9), rng.uniform(0, 2 * np.pi, n)
        d = np.stack([np.sin(th) * np.cos(ph), -np.cos(th), np.sin(th) * np.sin(ph)], 1)
        Ln = 2.9 / np.cos(th)
        self.rain = dict(t=tr, a=(G - d * Ln[:, None]).astype(np.float32), b=G.astype(np.float32), dur=Ln / 22.0)
        self.hit_t = np.sort([m["t_hit"] for m in mus if m["hit"] is not None])

    # ----------------------------------------------------------------- views
    def _view(self, t):
        """(kind, name, index, progress in the view)."""
        if t < BAR0:
            return "build", "PERSPECTIVE // ACQUIRING", 0, (t - T_IN) / (BAR0 - T_IN)
        names = [("persp", "PERSPECTIVE"), ("ortho", "ORTHO_FRONT / ORTHO_SIDE"), ("thorax", "THORAX / ORTHO_TOP"),
                 ("bone", "BONE // 1E-2 M"), ("cells", "CELLS // 1E-5 M"), ("dna", "DNA // 1E-9 M"),
                 ("atoms", "WATER // 1E-10 M"), ("track", "TRACK // MU")]
        edges = [BAR0] + CUTS + [T_OUT]
        k = int(np.searchsorted(edges, t, side="right")) - 1
        k = min(max(k, 0), len(names) - 1)
        return names[k][0], names[k][1], k + 1, (t - edges[k]) / (edges[k + 1] - edges[k])

    @staticmethod
    def _cut(t):
        """Show time of the last cut (the drop, then every change of view): what is NEW in a view is built from
        there. The furniture that stays (title, columns, bottom band) is built once, at the drop."""
        edges = [BAR0] + CUTS
        return edges[max(int(np.searchsorted(edges, t, side="right")) - 1, 0)]

    def _cams(self, kind, t, u):
        """[(camera, clip rect, column for its tags, label)] - every view centred in a free bay."""
        P = self.P
        if kind in ("build", "persp"):
            yaw = math.radians(-26.0 + 11.0 * (t - T_IN))
            D = 4.95 - 0.2 * smoothstep(BAR0, CUTS[0], t)
            cam = Camera((D * math.sin(yaw), 1.22, D * math.cos(yaw)), (0.0, 0.92, 0.0), fov_deg=34.0,
                         screen_center=(P.fx, Y_MID + 12))
            return [(cam, WALL, P.fcol, None)]
        if kind == "ortho":
            h = Y_MID + 12
            sc = 455.0
            cf = OrthoCamera((0.0, 0.92, 10.0), (0.0, 0.92, 0.0), scale=sc, screen_center=(P.fx, h))
            if P.side is None:          # one bay only: front on the first bar, side on the second
                if (t - CUTS[0]) < BAR:
                    return [(cf, (P.fbay[0] + 4, Y_TOP, P.fbay[1] - 4, Y_BOT), P.fcol, "FRONT")]
                cs = OrthoCamera((10.0, 0.92, 0.0), (0.0, 0.92, 0.0), scale=sc, screen_center=(P.fx, h))
                return [(cs, (P.fbay[0] + 4, Y_TOP, P.fbay[1] - 4, Y_BOT), P.fcol, "SIDE")]
            sb = P.bay_of(P.side)
            cs = OrthoCamera((10.0, 0.92, 0.0), (0.0, 0.92, 0.0), scale=sc,
                             screen_center=(0.5 * (P.side[0] + P.side[1]), h))
            return [(cf, (P.fbay[0] + 4, Y_TOP, P.fbay[1] - 4, Y_BOT), P.fcol, "FRONT"),
                    (cs, (max(sb[0], L.FX0) + 4, Y_TOP, min(sb[1], L.FX1) - 4, Y_BOT), P.side, "SIDE")]
        fit = min(1.0, (2 * P.half - 120.0) / 1000.0)       # the thorax has to fit between two towers
        yaw = math.radians(-22.0 + 20.0 * u)
        D = (1.5 - 0.14 * u) / fit
        cam = Camera((D * math.sin(yaw) + 0.03, 1.36, D * math.cos(yaw)), (0.03, 1.24, 0.0), fov_deg=40.0,
                     screen_center=(P.fx, Y_MID))
        if P.side is None:
            return [(cam, WALL, P.fcol, None)]
        # the neighbouring bay: the same moment seen from above, the slices stacked into a target
        sb = P.bay_of(P.side)
        sc = min(0.5 * width(P.side) - 36.0, 0.5 * (Y_BOT - Y_TOP) - 86.0)          # px per metre
        ct = OrthoCamera((0.0, 10.0, 1e-3), (0.0, 0.0, 0.0), scale=sc, up=(0.0, 0.0, -1.0),
                         screen_center=(0.5 * (P.side[0] + P.side[1]), Y_MID + 16), roll_deg=18.0 * u)
        return [(cam, (P.fbay[0] + 4, Y_TOP, P.fbay[1] - 4, Y_BOT), P.fcol, None),
                (ct, (max(sb[0], L.FX0) + 4, Y_TOP, min(sb[1], L.FX1) - 4, Y_BOT), P.side, "TOP")]

    # ----------------------------------------------------------------- render
    def draw(self, f, t, ctx):
        t = float(np.clip(t, T_IN, T_OUT - 1e-3))
        P = self.P
        kind, vname, vidx, u = self._view(t)
        ages = t - self.mu_t
        opt = {}
        micro = kind in ("bone", "cells", "dna", "atoms", "track")
        if not micro:
            scan_y = None
            if kind == "ortho":
                w = (t - CUTS[0]) / BAR             # down in one bar, up in the next: it slows down to turn round
                scan_y = 0.03 + 1.76 * (0.5 + 0.5 * math.cos(math.pi * w))
            for cam, clip, col, label in self._cams(kind, t, u):
                f.set_clip(*clip)
                self._draw_world(f, cam, t, ages, kind, scan_y, col, clip, label)
                if label:               # the name of the view, decoded when the view comes on
                    since = t - self._cut(t) - (BAR if label == "SIDE" and P.side is None else 0.0)
                    f.text("w", col[1] - 6, Y_BOT - 14, B.resolve(label, since, 40.0, 0.1, key=5, pad=True),
                           size=L.T_SMALL, alpha=0.7, anchor="rs")
            f.set_clip()
        else:
            f.set_clip(*WALL)
            getattr(self, f"_micro_{kind}")(f, t, u)
            f.set_clip()
        if t >= BAR0:
            self._draw_title(f, t, ages, kind)
            self._draw_data(f, t, ages)
            self._draw_view_tag(f, kind, vname, vidx, t)
            if micro:
                self._draw_scale_strip(f, t, kind, u)
            else:
                self._draw_strip(f, t)
            self._draw_bottom(f, t, ages)
        else:
            self._draw_build_hud(f, t)
            opt["edge_alpha"] = 0.6
        # the drop: one short inverted flash
        if BAR0 <= t < BAR0 + 0.07:
            opt["invert"] = True
        return opt

    # ------------------------------------------------------------------ world
    def _hot(self, t, ages):
        """Red glow sources on the cloud + red amount per slice level, from the recent hits."""
        hot_pts = []
        hot_lev = np.zeros(len(self.body.levels), np.float32)
        for m, a in zip(self.mus, ages):
            if m["hit"] is None:
                continue
            ah = a - m["dur"] * m["u_in"]
            if 0 <= ah < 0.9:
                hot_pts.append((m["hit"], math.exp(-ah / 0.3)))
            if 0 <= ah < 0.6:
                k = int(np.argmin(np.abs(self.body.levels - m["hit"][1])))
                hot_lev[max(0, k - 1): k + 2] = np.maximum(hot_lev[max(0, k - 1): k + 2], math.exp(-ah / 0.25))
        return hot_pts, hot_lev

    def _draw_world(self, f, cam, t, ages, kind, scan_y, col, clip, label=None):
        b = self.body
        top = label == "TOP"
        hot_pts, hot_lev = self._hot(t, ages)
        if kind == "build":
            # drum roll: the slices come on around the heart, a surge per 16th note (the figure grows without a
            # halt: it only breathes with the roll)
            k = max(0.0, (t - self.t_you) / STEP)
            hh = 0.05 + 0.1 * (math.floor(k) + 0.5 * (k % 1.0) + 0.5 * float(smoothstep(0.0, 1.0, k % 1.0)))
            b.draw(f, cam, hot_pts=hot_pts, hot_lev=hot_lev, cloud=0.0, slices=1.25 if t >= self.t_you else 0.0,
                   reveal=(HEART[1], hh))
            m0 = self.mus[0]
            a0 = t - m0["t"]
            if a0 >= 0:
                self._draw_persistent(f, cam, m0, a0, t, col)
            return
        b.draw_floor(f, cam)
        self.P.plates(f)            # the floor is the only thing that would run under the title / data columns
        if scan_y is not None:
            k = int(np.argmin(np.abs(b.levels - scan_y)))
            hot_lev[k] = 1.0
        b.draw(f, cam, hot_pts=hot_pts, hot_lev=hot_lev, top=top)
        if top:                     # range rings around you: traced outwards from the centre when the view comes on
            o = cam.project(np.zeros((1, 3), np.float32))
            X, Y = float(o[0][0]), float(o[1][0])
            rr = [r for r in (0.25, 0.5, 0.75, 1.0) if cam.scale * r < 0.5 * width(col) + 10]
            with f.build(t - self._cut(t), clip, flow="out", origin=(X, Y), wave=0.35, marks=False, key=31):
                f.rings("w", [X] * len(rr), [Y] * len(rr), [cam.scale * r for r in rr], E.wl(0.32), width=E.ww(1.0))
                for r in rr:
                    f.text("w", X + cam.scale * r * 0.7071 + 6, Y + cam.scale * r * 0.7071 + 14, f"{r:.2f} M",
                           size=L.T_MICRO, alpha=0.7)
                f.segments("r", [clip[0], X], [Y, clip[1]], [clip[2], X], [Y, clip[3]], 0.6 if E.WALL else 0.3)
        self._draw_rain(f, cam, t)
        for m, a in zip(self.mus, ages):
            if a < 0 or a > m["dur"] + (2.6 if m["hero"] else 0.9) or m is self.echo:
                continue
            # between two close towers only the heart keeps its tag (the others would sit on the figure)
            draw_muon(f, cam, m, a, col=col, tag=not top and (self.P.half >= 300.0 or m["part"] == "HEART"),
                      lines=self.P.half >= 400.0)
        if kind == "thorax":
            a = t - self.echo["t"]
            if a >= 0:
                self._draw_persistent(f, cam, self.echo, a, t, col, lines=not top and self.P.half >= 400.0)
        if scan_y is not None:      # the scan slice: its line is drawn across the bay, its read-out made
            p = cam.project(np.array([[0.0, scan_y, 0.0]], np.float32))
            y = float(p[1][0])
            with f.build(t - CUTS[0], (clip[0], y - 34.0, clip[2], y + 6.0), flow="lr", wave=0.25, marks=False, key=32):
                f.segments("r", [clip[0]], [y], [clip[2]], [y], 0.8, width=1.5)
                f.tag("r", col[0] + 6, y - 9, f"SLICE Y {scan_y:.3f} M", size=L.T_SMALL, pad=4)

    def _draw_rain(self, f, cam, t):
        r = self.rain
        i0, i1 = np.searchsorted(r["t"], t - 0.5), np.searchsorted(r["t"], t)
        if i1 <= i0:
            return
        age = t - r["t"][i0:i1]
        dur = r["dur"][i0:i1]
        prog = np.minimum(1.0, age / dur)
        A, B = r["a"][i0:i1], r["b"][i0:i1]
        Hd = A + (B - A) * prog[:, None].astype(np.float32)
        ax, ay, _, aok = cam.project(A)
        hx, hy, _, hok = cam.project(Hd)
        fade = np.where(age < dur, 1.0, np.exp(-(age - dur) / 0.1))
        ok = aok & hok
        if E.WALL:              # the rain: a tail that still dies away, a body that lands
            f.segments("r", ax[ok], ay[ok], hx[ok], hy[ok], 0.2 * fade[ok], 0.7 * fade[ok], width=1.3)
        else:
            f.segments("r", ax[ok], ay[ok], hx[ok], hy[ok], 0.1 * fade[ok], 0.42 * fade[ok])
        fl = ok & (age < dur)
        f.dots("r", hx[fl], hy[fl], 2.0, 1.1)
        gd = ok & (age >= dur) & (age < dur + 0.3)
        if gd.any():
            f.dots("r", hx[gd], hy[gd], 1.6, 1.0 * (1 - (age[gd] - dur[gd]) / 0.3))

    def _draw_persistent(self, f, cam, m, a, t, col, lines=True):
        """A hero track that stays on screen once it has passed (the one of 'YOU', the one of 'an echo')."""
        prog = min(1.0, a / m["dur"])
        A, B = m["a"], m["b"]
        sx, sy, _, ok = cam.project(np.stack([A, A + (B - A) * prog]).astype(np.float32))
        if not ok.all():
            return
        hold = 0.62 + 0.38 * math.exp(-max(0.0, a - m["dur"]) / 0.5)
        f.segments("r", sx[:1], sy[:1], sx[1:], sy[1:], 0.9 * hold, width=2.2)
        for (u0, u1) in m["runs"]:
            if u0 >= prog:
                continue
            qx, qy, _, _ = cam.project(np.stack([A + (B - A) * u0, A + (B - A) * min(u1, prog)]).astype(np.float32))
            f.segments("r", qx[:1], qy[:1], qx[1:], qy[1:], 1.5 * hold, width=3.4)
        if prog < 1.0:
            f.dots("r", sx[1:], sy[1:], 4.2, 1.7)
            f.dots("w", sx[1:], sy[1:], 1.6, 1.0)
        ah = a - m["dur"] * m["u_in"]
        if ah < 0:
            return
        hx, hy, _, _ = cam.project(m["hit"][None].astype(np.float32))
        x, y = float(hx[0]), float(hy[0])
        if ah < 1.0:
            uu = ah / 1.0
            f.rings("r", [x], [y], [10 + 120 * (1 - (1 - uu) ** 3)], (1 - uu) ** 1.5, width=2.0)
        pulse = 0.55 + 0.45 * math.exp(-self.ctx.cues.since_kick(t) / 0.1)
        f.rings("r", [x], [y], [15.0], 0.9 * pulse, width=1.8)
        f.dots("w", [x], [y], 2.6, 1.4)
        callout(f, x, y, "HEART", [f"MU{m['charge']} {m['E']:.3f} GEV", f"DE {m['dE'] * 1000:.1f} MEV",
                                   "FROM 15.2 KM UP"] if lines else [], col=col, dy=-56.0, red=True, age=ah,
                elbow=46.0)

    # ------------------------------------------------------------------ HUD
    def _draw_build_hud(self, f, t):
        """Before the drop: almost nothing. A line of status under the figure and its bar, constructed on the
        roll: the line is decoded, two pens trace the bar, one block opens on every 16th note."""
        a = t - self.t_you
        if a < 0:
            return
        P = self.P
        y = Y_BOT - 8
        msg = fit_text(["ACQUIRING // SUBJECT 01 // 1.80 M", "ACQUIRING // SUBJECT 01", "ACQUIRING"], width(P.fcol),
                       L.T_LABEL)
        f.text("w", P.fx, y, B.resolve(msg, a, 40.0, key=3, pad=True), size=L.T_LABEL, alpha=0.85, anchor="ms")
        n = 14
        pitch = min(26.0, (width(P.fcol) - 20) / n)
        x0 = P.fx - n * pitch / 2
        xs = x0 + np.arange(n) * pitch
        xa, xb, ya, yb = x0 - 6, x0 + n * pitch - 2, y + 16, y + 40
        ym = 0.5 * (ya + yb)
        pc, pe = B.lin(a, 0.0, 0.08), float(B.ease(B.lin(a, 0.06, 0.42)))      # its left end, then its two long sides
        f.segments("w", [xa], [ym - 12 * pc], [xa], [ym + 12 * pc], E.wl(0.6), width=E.ww(1.0))
        xe = xa + (xb - xa) * pe
        if pe > 0.0:
            f.segments("w", [xa, xa], [ya, yb], [xe, xe], [ya, yb], E.wl(0.6), width=E.ww(1.0))
            if pe < 1.0:
                f.dots("w", [xe, xe], [ya, yb], 3.0, 1.7)
            else:
                f.segments("w", [xb], [ya], [xb], [yb], E.wl(0.6), width=E.ww(1.0))
        g = B.spring((a - np.arange(n) * STEP) / 0.14)       # a block per 16th note: it opens from its middle line
        on = (g > 0.02) & (xs + pitch - 8 <= xe)
        f.rects("w", xs[on], ym - 7 * g[on], xs[on] + pitch - 8, ym + 7 * g[on], 0.95)

    def _draw_view_tag(self, f, kind, name, idx, t):
        P = self.P
        y0 = Y_TOP + 36
        since = t - self._cut(t)                    # the tag of a view is made when the view comes on
        opts = [f"VIEW {idx:02d} // {name}"]
        for sep in (" / ", " // "):
            if sep in name:
                opts.append(f"VIEW {idx:02d} // {name.split(sep)[0]}")
        opts.append(f"VIEW {idx:02d}")
        if kind in ("bone", "cells", "dna", "atoms", "track"):
            # the track leaves the top of the bay on the left of its centre: the tag goes on the right
            room = P.fcol[1] - (P.fx - 50)
            B.tag(f, "w", P.fcol[1] - 6, y0, fit_text(opts, room, L.T_LABEL), since, size=L.T_LABEL, pad=5, anchor="rs",
                  cps=90.0, key=idx)
        else:
            room = width(P.fcol) if kind == "ortho" else (P.fx - 80) - P.fcol[0]
            B.tag(f, "w", P.fcol[0] + 6, y0, fit_text(opts, room, L.T_LABEL), since, size=L.T_LABEL, pad=5, cps=90.0,
                  key=idx)
            if kind == "ortho":
                s = 455.0
                x0 = P.fcol[0] + 6
                if P.fx - 200 - x0 > 0.5 * s + 70:
                    with f.build(since - 0.15, (x0 - 4, y0 + 20, x0 + 0.5 * s + 90, y0 + 50), flow="lr", wave=0.15,
                                 marks=False, key=33):
                        f.segments("w", [x0, x0, x0 + 0.5 * s], [y0 + 34, y0 + 26, y0 + 26],
                                   [x0 + 0.5 * s, x0, x0 + 0.5 * s], [y0 + 34, y0 + 42, y0 + 42], 0.9, width=L.LW)
                        f.text("w", x0 + 0.5 * s + 12, y0 + 40, "0.5 M", size=L.T_SMALL, alpha=0.8)
        # what the neighbouring bay is: the same rain, the same floor (it stays through the three views of the
        # figure: built once, at the drop)
        if P.side is not None and kind in ("persp", "ortho", "thorax"):
            info = fit_text([f"YOU // MUON BLOOM // 1.80 M // EFFECTIVE AREA {sd.AREA_YOU:.2f} M2",
                             f"YOU // 1.80 M // AREA {sd.AREA_YOU:.2f} M2", "YOU // 1.80 M"], width(P.side), L.T_SMALL)
            x1 = P.side[1] - 6
            box = (x1 - text_w(info, L.T_SMALL) - 10, y0 - 22, x1 + 6, y0 + 34)
            plate(f, box, t - BAR0 - 0.1, flow="lr")
            with f.build(t - BAR0 - 0.1, box, flow="lr", wave=0.25, marks=False, key=34):
                f.text("w", x1, y0, info, size=L.T_SMALL, alpha=0.85, anchor="rs")
                f.text("w", x1, y0 + 26, f"{sd.tc(t)}  //  1 MUON /CM2 /MIN", size=L.T_MICRO, alpha=0.6, anchor="rs")

    def _draw_title(self, f, t, ages, kind):
        """YOU, the parameters of the view, the hit log (leftmost free column). Built at the drop: the title
        tag is pushed out, the parameters are decoded (again at every change of medium), every line of the log
        is decoded when its hit happens."""
        P = self.P
        if P.title is None:
            return
        x0, x1 = P.title
        w = x1 - x0
        y0 = Y_TOP + 10
        since = t - BAR0
        ts = min(112.0, (w - 40) / (3 * 0.61))
        B.tag(f, "w", x0 + 10, y0 + 0.93 * ts, "YOU", since, t0=0.05, size=ts, pad=10, bold=True, wipe=0.22, cps=14.0,
              key=1)
        if kind in self.MEDIA:
            rows_, t_rows = self.MEDIA[kind], self._cut(t)
        else:
            rows_ = ["MU FLUX    1 /CM2/MIN", f"THROUGH YOU   ~{RATE:.0f} /S", "HEIGHT       1.800 M",
                     f"AREA_EFF     {sd.AREA_YOU:.2f} M2", "DE/DX     2.0 MEV/CM"]
            t_rows = BAR0 + 0.2
        yr = y0 + ts + 64
        with f.build(t - t_rows, (x0 - 6, yr - 24, x1, yr + 5 * 25.5), flow="tb", wave=0.25, marks=False, key=40):
            hud.rows(f, x0, yr, rows_, size=L.T_SMALL, lead=1.5)
        yl = yr + 5 * 25.5 + 30
        full = w >= 300
        with f.build(since - 0.4, (x0, yl - 20, x1, yl + 8), flow="lr", wave=0.15, marks=False, key=39):
            f.tag("w", x0 + 4, yl, "HIT_LOG", size=L.T_MICRO, pad=3)
            if full:
                f.text("w", x0 + 96, yl, "PART   X     Y    Z     MEV", size=L.T_MICRO, alpha=0.5)
        hits = [(a - m["dur"] * m["u_in"], m) for m, a in zip(self.mus, ages) if m["hit"] is not None]
        n_rows = int((Y_BOT - yl - 40) / 21)
        hits = sorted([h for h in hits if h[0] >= 0], key=lambda h: h[0])[:n_rows]
        for k, (ah, m) in enumerate(hits):
            hp = m["hit"]
            if full:
                line = f"{m['part']:<7}{hp[0]:+.2f} {hp[1]:.2f} {hp[2]:+.2f} {m['dE'] * 1000:5.1f}"
            else:
                line = f"{m['part']:<7} {hp[1]:.2f} M {m['dE'] * 1000:5.1f} MEV"
            # a line is written when its muon hits (and the ones that were there at the drop, with the log)
            line = B.resolve(line, min(ah, since - 0.5 - 0.03 * k), 140.0, key=int(m["t"] * 1000) & 0xFFFF)
            f.text("r" if m["part"] == "HEART" or k == 0 else "w", x0, yl + 32 + k * 21, line, size=L.T_MICRO,
                   alpha=0.95 if k < 3 else 0.65)

    def _energy(self, ages):
        dep = {}
        for m, a in zip(self.mus, ages):
            if m["hit"] is None:
                continue
            ah = a - m["dur"] * m["u_in"]
            if ah >= 0:
                dep[m["part"]] = dep.get(m["part"], 0.0) + m["dE"] * 1000 * math.exp(-ah / 5.0)
        return dep

    PARTS = ["HEAD", "HEART", "TORSO", "ARM_L", "ARM_R", "HAND_L", "HAND_R", "LEG_L", "LEG_R", "FOOT_L"]

    def _draw_data(self, f, t, ages):
        """Energy left in each part of the body + what falls on the wall (the remaining column)."""
        P = self.P
        if P.data is None:
            return
        x0, x1 = P.data
        y = Y_TOP + 44
        since = t - BAR0                            # the two panels of the column are constructed at the drop
        with f.build(since - 0.15, (x0 - 8, y - 24, x1 + 8, y + 30 + len(self.PARTS) * 36 + 2), flow="tb", wave=0.45,
                     key=41):
            hud.panel_header(f, x0, x1, y, "ENERGY LEFT IN YOU // MEV")
            dep = self._energy(ages)
            bw = x1 - x0 - 92 - 74
            for k, p in enumerate(self.PARTS):
                yy = y + 30 + k * 36
                v = dep.get(p, 0.0)
                f.text("r" if p == "HEART" and v > 1 else "w", x0 + 2, yy + 16, f"{p:<7}", size=L.T_SMALL, alpha=0.85)
                f.text("w", x1, yy + 16, f"{v:5.1f}", size=L.T_SMALL, alpha=0.8, anchor="rs")
            yy = y + 30 + np.arange(len(self.PARTS)) * 36.0             # the bars glide as the energy drains
            vv = np.array([dep.get(p, 0.0) for p in self.PARTS])
            heart = np.array([p == "HEART" for p in self.PARTS])
            for lay, m in (("r", heart), ("w", ~heart)):
                hud.bars(f, lay, x0 + 92, yy[m] + 2, x0 + 92 + np.minimum(bw, vv[m] * bw / 260.0), yy[m] + 17, 0.95)
        y2 = y + 30 + len(self.PARTS) * 36 + 56
        s = min(64.0, (x1 - x0 - 10) / (6 * 0.61))
        with f.build(since - 0.5, (x0 - 8, y2 - 24, x1 + 8, y2 + 124 + s), flow="tb", wave=0.3, key=42):
            hud.panel_header(f, x0, x1, y2, "MEANWHILE // EVERY SQUARE METRE")
            f.text("w", x0, y2 + 34 + s, "167 /S", size=s, alpha=0.97)
            f.text("w", x0 + 2, y2 + 68 + s, "MUONS, DAY AND NIGHT", size=L.T_SMALL, alpha=0.8)
            f.text("w", x0 + 2, y2 + 94 + s, "1 /CM2 /MIN AT THE GROUND", size=L.T_MICRO, alpha=0.6)
            f.text("w", x0 + 2, y2 + 116 + s, "ROOFS AND WALLS DO NOT STOP THEM", size=L.T_MICRO, alpha=0.6)
        y3 = y2 + 124 + s + 46
        if Y_BOT - 4 - y3 >= 150.0:                 # room left under it: where the one we follow comes from
            self._draw_source(f, t, x0, x1, y3, Y_BOT - 4.0, since - 0.85)

    def _draw_source(self, f, t, x0, x1, y0, y1, age):
        """ORIGIN: the star of the story, after the user's sketch - a disc, rays leaving it all around, each one
        ending in a dot. It is alive: every ray is sent again in its turn (its dot leaves the disc and draws the
        ray behind it, waits at the end, then the ray is reeled in), the disc rings on the half bars, and one
        ray is red: the one that ends in YOU. It is the star behind the muon this scene follows, not the source
        of every muon through the body, and the title says so."""
        P = self.P
        w = x1 - x0
        left = 0.5 * (x0 + x1) > P.fx               # the figure stands on the left of this column: the red ray points there
        tw = 134.0                                  # room the read-outs take beside the star
        text = w >= tw + 16.0 + 140.0
        R = min(0.5 * (y1 - y0 - 24.0) - 4.0, 0.5 * (w - tw - 16.0) if text else 0.5 * w - 12.0, 96.0)
        cy_ = y0 + 24.0 + 0.5 * (y1 - y0 - 24.0)
        cx_ = 0.5 * (x0 + x1) if not text else (x1 - R - 4.0 if left else x0 + R + 4.0)
        rd = 0.34 * R
        with f.build(age, (x0 - 8, y0 - 24, x1 + 8, y1), flow="tb", wave=0.35, key=47):
            hud.panel_header(f, x0, x1, y0, fit_text(["ORIGIN // THE STAR BEHIND MU- 0001", "ORIGIN // ONE STAR", "ORIGIN"],
                                                     w, L.T_MICRO))
            # the disc: it rings (a shell leaves it on every half bar), its core beats with the kicks
            ring = np.linspace(0, 2 * np.pi, 49)
            kick = math.exp(-self.ctx.cues.since_kick(t) / 0.12)
            f.polyline("w", cx_ + rd * np.cos(ring), cy_ + rd * np.sin(ring), 1.05, width=2.2)
            ab = np.linspace(0, 2 * np.pi, 40, endpoint=False) + 0.25 * t       # its shell: a ring of bars, turning
            f.segments("w", cx_ + 0.52 * rd * np.cos(ab), cy_ + 0.52 * rd * np.sin(ab), cx_ + 0.82 * rd * np.cos(ab),
                       cy_ + 0.82 * rd * np.sin(ab), E.wl(0.5), width=E.ww(1.0))
            f.polyline("w", cx_ + 0.4 * rd * np.cos(ring), cy_ + 0.4 * rd * np.sin(ring), E.wl(0.5), width=E.ww(1.0))
            f.dots("r", [cx_], [cy_], 4.0 + 1.8 * kick, 1.5)
            ph = ((t - BAR0) % (BAR / 2)) / (BAR / 2)
            rs = rd + (0.85 * R - rd) * (1 - (1 - ph) ** 2)
            f.polyline("w", cx_ + rs * np.cos(ring), cy_ + rs * np.sin(ring), (0.95 if E.WALL else 0.42) * (1 - ph) ** 1.5,
                       width=E.ww(1.0))
            # the rays: 28 of them, as drawn. One is red, aimed at the figure
            th, ln = self.src_th.copy(), self.src_len.copy()
            aim = math.pi if left else 0.0
            kr = int(np.argmin(np.abs(np.angle(np.exp(1j * (th - aim))))))
            th[kr], ln[kr] = aim, 1.0
            T = self.src_T.copy()
            tau = (t + self.src_off) % T
            T[kr], tau[kr] = BAR, (t - BAR0) % BAR                # the red one leaves on every bar
            head = 1.0 - (1.0 - np.clip(tau / 0.9, 0.0, 1.0)) ** 3            # its dot runs out and draws the ray
            tail = np.clip((tau - (T - 0.7)) / 0.55, 0.0, 1.0) ** 2             # ... which is reeled in before the next
            dot = np.clip((T - tau) / 0.15, 0.0, 1.0)
            uu = tail[:, None] + (head - tail)[:, None] * np.linspace(0.0, 1.0, 7)[None, :]
            rr = (rd + 3.0) + (ln[:, None] * R - rd - 3.0) * uu
            lat = 2.2 * uu * np.sin(2.6 * math.pi * uu + self.src_wav[:, None] + 0.8 * t)     # the hand of the sketch
            ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
            X, Y = cx_ + rr * ct - lat * st, cy_ + rr * st + lat * ct
            red = np.arange(len(th)) == kr
            for lay, m, gain in (("w", ~red, 0.8), ("r", red, 1.3)):
                f.segments(lay, X[m, :-1].ravel(), Y[m, :-1].ravel(), X[m, 1:].ravel(), Y[m, 1:].ravel(), gain,
                           width=L.LW if lay == "w" else 2.0)
                f.dots(lay, X[m, -1], Y[m, -1], (3.1 if lay == "w" else 4.2) * dot[m], 1.35)
            f.dots("w", X[red, -1], Y[red, -1], 1.6 * dot[red], 1.1)
            if text:
                xt, anc = (cx_ - R - 10.0, "rs") if left else (cx_ + R + 10.0, "ls")
                for yy, lab, val in ((cy_ - 64.0, "COLLAPSED", "4.8E9 YR AGO"), (cy_ + 44.0, "SENT OUT", "P+ 3.2E15 EV")):
                    f.text("w", xt, yy, lab, size=L.T_MICRO, alpha=0.6, anchor=anc)
                    f.text("w", xt, yy + 21, val, size=L.T_SMALL, alpha=0.9, anchor=anc)
                f.text("r", xt, cy_ - 18.0, "ONE RAY ENDS IN", size=L.T_MICRO, alpha=0.9, anchor=anc)
                f.tag("r", xt - (5 if left else -5), cy_ + 9.0, "YOU", size=L.T_TAG, pad=5, anchor=anc, bold=True)

    def _draw_strip(self, f, t):
        ta, tb = t - 5.5, t + 2.0
        age = t - BAR0                              # built at the drop: band and rules, then the hits from the left
        x0, y0, x1, y1, yb = hud.strip_base(f, title="HIT_TIMELINE // WHITE = MUON  RED = ENERGY LEFT IN YOU",
                                            ticks=(ta - BAR0, tb - BAR0, STEP, BAR), age=age)
        X = lambda tt: x0 + (np.asarray(tt) - ta) / (tb - ta) * (x1 - x0)
        placed = []
        with f.build(age - 0.3, L.STRIP, flow="lr", wave=0.5, marks=False, bars="centre", key=43):
            comb = {"w": [], "r": []}               # the bars of the two combs: x, top, bottom, intensity
            for m in self.mus:
                tt = m["t_hit"] if m["t_hit"] is not None else m["t"]
                if not (ta - 0.2 <= tt <= tb + 0.2) or tt < T_IN:
                    continue
                xe = float(X(tt))
                key = int(m["t"] * 1000)
                n = 2 + int(hash01(key, 2) * 6)
                bx = xe + np.arange(n) * 4.0
                ok = (bx > x0) & (bx < x1 - 3)
                hh = (8 + 10 * math.log1p(m["E"])) * (0.3 + 0.7 * hash01(key, np.arange(n)))
                one = np.ones(int(ok.sum()))
                comb["w"].append((bx[ok], (y0 + 1) * one, y0 + 1 + hh[ok], (0.95 if tt <= t else 0.4) * one))
                if m["part"]:
                    hb = (6 + 60 * m["dE"]) * (0.3 + 0.7 * hash01(key, np.arange(n) + 9))
                    comb["r"].append((bx[ok], y1 - hb[ok], (y1 - 1) * one, (0.9 if tt <= t else 0.4) * one))
                    if m["hero"] and x0 + 10 < xe < x1 - 110:
                        row = 1 if any(abs(p - xe) < 120 for p in placed) else 0
                        placed.append(xe)
                        # the tag of a hit is made when it enters the strip on the right, and taken apart
                        # before it leaves on the left (the strip scrolls at `v` px / s)
                        v = (x1 - x0) / (tb - ta)
                        B.tag(f, "r" if tt <= t else "w", xe, yb + 26 + row * 24, m["part"],
                              B.io(min((x1 - 110 - xe) / v, age - 0.5), (xe - x0 - 10) / v, out=0.25, span=0.3),
                              size=L.T_MICRO, pad=4, alpha=1.0 if tt <= t else 0.5, cps=60.0, key=key & 0xFFF)
            for lay, lst in comb.items():           # the strip scrolls: its bars glide (hud.bars), in one go
                if lst:
                    bx, ya_, yb_, ii = (np.concatenate(v) for v in zip(*lst))
                    hud.bars(f, lay, bx, ya_, bx + 2, yb_, ii)
            hud.strip_cursor(f, float(X(t)), y0, y1, f"T {sd.tc(t)}")

    SCALES = {"bone": -2.0, "cells": -4.7, "dna": -8.7, "atoms": -9.5, "track": -15.0}
    MARKS = [(0.26, "YOU 1.8 M"), (-2.0, "RIB 1 CM"), (-4.7, "CELL 20 UM"), (-8.7, "DNA 2 NM"), (-9.5, "H2O 0.3 NM"),
             (-15.0, "NUCLEUS"), (-18.0, "MUON < 1E-18 M")]
    MEDIA = {
        "bone": ["MEDIUM  CORTICAL BONE", "RHO        1.92 G/CM3", "DE/DX     3.4 MEV/CM", "FIELD        ~10 CM",
                 "DEPTH       0.062 M"],
        "cells": ["MEDIUM      MYOCARDIUM", "CELL          ~20 UM", "DE/DX     0.20 KEV/UM", "FIELD         650 UM",
                  "ION PAIRS   ~6.6 /UM"],
        "dna": ["MEDIUM       CHROMATIN", "HELIX         2.0 NM", "TURN          3.4 NM", "FIELD          30 NM",
                "MEAN FREE    ~150 NM"],
        "atoms": ["MEDIUM   WATER  H2O", "O-H        0.096 NM", "IONISATION   12.6 EV", "FIELD          10 NM",
                  "MUON      POINT-LIKE"],
        "track": ["MEDIUM            ---", "CHARGE          -1 E", "SPIN             1/2", "SIZE     < 1E-18 M",
                  "NAME             ___"],
    }

    def _draw_scale_strip(self, f, t, kind, u):
        """The scale strip replaces the hit timeline when we leave the body (first micro view): it is built
        there. At every cut its cursor runs down the powers of ten to the new field (its read-out counts with
        it) and drops the blocks of the comb as it passes."""
        age = t - CUTS[2]
        x0, y0, x1, y1, yb = hud.strip_base(f, title="SCALE // POWERS OF TEN ALONG THE TRACK // METRES",
                                            ticks=(1.0, -19.0, 0.2, 1.0), age=age)
        X = lambda e: x0 + (1.0 - np.asarray(e, np.float64)) / 20.0 * (x1 - x0)
        order = list(self.SCALES)
        k = order.index(kind)
        prev = 0.26 if k == 0 else self.SCALES[order[k - 1]] - 0.1          # where the cursor was before this cut
        run = float(B.ease(B.lin(t - self._cut(t), 0.35 if k == 0 else 0.0, 0.85 if k == 0 else 0.5)))
        cur = prev + (self.SCALES[kind] - 0.1 * u - prev) * run
        with f.build(age - 0.3, L.STRIP, flow="lr", wave=0.5, marks=False, bars="down", key=44):
            for e in range(0, -19, -3):
                f.text("w", float(X(e)) + 5, y0 + 32, f"1E{e:+03d}", size=L.T_MICRO, alpha=0.75)
            for e, word in self.MARKS:
                xv = float(X(e))
                last = e <= -17.5
                red = last or abs(e - self.SCALES[kind]) < 0.3
                f.segments("w", [xv], [yb - 12], [xv], [yb + 12], 0.9)
                f.tag("r" if red else "w", xv - (text_w(word, L.T_MICRO) + 8 if last else -4),
                      yb + (48 if e in (-9.5, -15.0) else 26), word, size=L.T_MICRO, pad=3,
                      alpha=1.0 if e >= cur - 0.3 or last else 0.45)
        # every decade we went through leaves a block on the top comb (dropped by the cursor as it passes)
        dec = np.arange(0.0, cur, -0.2)
        xd = X(dec)
        hd = (13 + 16 * hash01(np.arange(len(dec)), 4)) * B.spring((dec - cur) / 0.5 + 0.25)
        f.rects("w", xd, y0 + 1, xd + 3, y0 + 1 + hd, 0.9)
        xc = float(X(cur))
        with f.build(age - 0.3, (xc - 4.0, y0 - 4.0, xc + 220.0, y1 + 4.0), flow="tb", wave=0.1, marks=False, key=38):
            hud.strip_cursor(f, xc, y0, y1, f"FIELD 1E{cur:+06.2f} M")

    def _draw_bottom(self, f, t, ages):
        """Counts on every time scale, hit barcode (+ energy if there was no column for it): the free
        panels of the bottom band, widest first."""
        P = self.P
        y0, y1 = P.py0, P.py1
        panels = sorted([p for p in P.panels if width(p) >= 200.0], key=lambda p: (-round(width(p)), p[0]))
        since = t - BAR0                            # the panels are constructed at the drop, one after the other
        block = lambda k: f.build(since - 0.2 - 0.15 * k, (panels[k][0] - 8, y0 - 24, panels[k][1] + 8, y1 + 8), wave=0.4,
                                  key=45 + k)
        if panels:
            cx0, cx1 = panels[0]
            with block(0):
                hud.panel_header(f, cx0, cx1, y0, "MUONS THROUGH YOU")
                cols = [("PER SECOND", sd.through_you_in(1)), ("PER MINUTE", sd.through_you_in(60)),
                        ("PER DAY", sd.through_you_in(86400, short=True)), ("IN A LIFE", sd.through_you_in(80 * sd.YEAR))]
                n = int(min(4, max(1, (cx1 - cx0) // 180)))
                cols = {4: cols, 3: [cols[0], cols[1], cols[3]], 2: [cols[0], cols[3]], 1: [cols[0]]}[n]
                cw = (cx1 - cx0) / n
                fs = min(52.0, (cw - 12) / (6 * 0.61))
                for k, (lab, val) in enumerate(cols):
                    xx = cx0 + k * cw
                    f.text("w", xx + 4, y0 + 36, lab, size=L.T_MICRO, alpha=0.75)
                    f.text("r" if lab == "IN A LIFE" else "w", xx + 2, y0 + 100, val, size=fs, alpha=0.97)
        if len(panels) > 1:
            bx0, bx1 = panels[1]
            with block(1):
                hud.panel_header(f, bx0, bx1, y0, "HIT_BARCODE")
                n = int(np.clip((bx1 - bx0) / 3.9, 40, 200))
                kk, frac, dt = hud.barcode_keys(t, 3.0, n)
                tt = kk * dt
                lo = np.searchsorted(self.hit_t, tt)
                hi = np.searchsorted(self.hit_t, tt + dt * 3)
                dens = np.where(tt >= self.t_you - 0.1, 0.05 + 0.9 * np.tanh((hi - lo) / 1.5), 0.0)
                hud.barcode_lanes(f, bx0, bx1, y0 + 12, y1, dens, kk, lanes=3, seed=7, frac=frac)
        if len(panels) > 2 and P.data is not None:
            with block(2):
                self._draw_sequence(f, panels[2], t)
        if len(panels) > 2 and P.data is None and width(panels[2]) >= 330:
            ex0, ex1 = panels[2]
            with block(2):
                hud.panel_header(f, ex0, ex1, y0, "ENERGY LEFT IN YOU // MEV")
                dep = self._energy(ages)
                ncol = 2 if ex1 - ex0 >= 620 else 1
                cw = (ex1 - ex0) / ncol
                for k, p in enumerate(self.PARTS[: 5 * ncol]):
                    c, r = k // 5, k % 5
                    xx, yy = ex0 + c * cw, y0 + 20 + r * 22
                    v = dep.get(p, 0.0)
                    f.text("r" if p == "HEART" and v > 1 else "w", xx + 2, yy + 14, f"{p:<7}", size=L.T_MICRO, alpha=0.85)
                    f.rects("r" if p == "HEART" else "w", xx + 78, yy + 3, xx + 78 + min(cw - 150, v * 1.2), yy + 14, 0.95)
                    f.text("w", xx + cw - 14, yy + 14, f"{v:5.1f}", size=L.T_MICRO, alpha=0.8, anchor="rs")

    SEQ = ["PERSP", "ORTHO", "THRX", "BONE", "CELL", "DNA", "H2O", "TRACK"]

    def _draw_sequence(self, f, panel, t):
        """Where we are in the zoom: one box per view, the current one red."""
        x0, x1 = panel
        hud.panel_header(f, x0, x1, self.P.py0, "SEQUENCE // 1.8 M -> 1E-18 M")
        _, _, idx, u = self._view(t)
        n = len(self.SEQ)
        w = (x1 - x0) / n
        y = self.P.py0 + 24
        for k, name in enumerate(self.SEQ):
            xx = x0 + k * w
            f.rect("w", xx + 2, y, xx + w - 5, y + 40, 0.7)
            if k + 1 < idx:
                f.rects("w", xx + 6, y + 4, xx + w - 9, y + 36, 0.9)
            elif k + 1 == idx:
                hud.bars(f, "r", xx + 6, y + 4, xx + 6 + (w - 15) * max(0.08, u), y + 36, 0.95)
            if w >= 46:
                f.text("r" if k + 1 == idx else "w", xx + 3, y + 64, name, size=L.T_MICRO,
                       alpha=0.95 if k + 1 <= idx else 0.5)
        f.text("w", x0 + 3, y + 98, f"VIEW {idx:02d} / {n:02d}", size=L.T_SMALL, alpha=0.8)

    # ------------------------------------------------------------------------------
    # micro views: the same straight track, the matter around it at smaller and smaller scales.
    # The field fills the wall (it runs behind the towers); the track, its ticks and its tags stay in the
    # focus bay; the read-outs sit on plates in the side column.
    # ------------------------------------------------------------------------------
    TILT = math.radians(11.0)

    def _build_micro(self, rng):
        self.fib_ph = rng.uniform(0, 2 * np.pi, (110, 3))
        self.fib_y = np.sort(rng.uniform(-1.02, 0.95, 110))
        self.skin = rng.uniform(0, 1, (3200, 2))
        self.marrow = rng.uniform(-1, 1, (1500, 2))
        self.water = rng.uniform(0, 1, (1400, 2))
        self.delta = np.cumsum(rng.normal(0, 1, (26, 2)) * np.array([1.0, 0.6]) + np.array([0.55, -0.3]), 0)
        self._build_bone()
        self._build_cells()
        self._build_source()

    def _build_bone(self):
        """What moves in the BONE view, and does not depend on t: the speed of the ripple of every fibre, the
        pulses that run along some of them, the nodes of the trabecular network, the ion pairs of the track."""
        n = len(self.fib_y)
        k = np.arange(n)
        self.fib_w = 0.9 + 0.9 * hash01(k, 61)                       # rad / s
        self.fib_i = 0.2 + 0.16 * hash01(k, 3)
        nd = 18                                                      # pulses travelling along a fibre
        d = np.arange(nd)
        self.dash_k = (hash01(d, 62) * n).astype(int)
        self.dash_v = 0.45 + 0.5 * hash01(d, 63)                     # cm / s
        self.dash_o = hash01(d, 64)
        # trabecular network: a jittered lattice inside the rib, linked to its neighbours
        i, j = np.meshgrid(np.arange(-9, 10), np.arange(-6, 7))
        jx, jy = _jit(i, j, 21, 0.03)
        self.tr_x, self.tr_y = (i * 0.085 + jx).ravel(), (j * 0.085 + jy).ravel()      # cm from the centre of the rib
        self.tr_ph = 2 * np.pi * np.stack([hash01(i, j, 22), hash01(i, j, 23)]).reshape(2, -1)
        ok = (self.tr_x / (0.98 * 0.76)) ** 2 + (self.tr_y / (0.6 * 0.76)) ** 2 < 1.0
        self.tr_ok = ok
        idx = np.arange(i.size).reshape(i.shape)
        ok2 = ok.reshape(i.shape)
        la, lb = [], []
        for di, dj, sd_ in ((1, 0, 31), (0, 1, 32), (1, 1, 33)):
            a_ = (slice(None, -dj or None), slice(None, -di or None))
            b_ = (slice(dj, None), slice(di, None))
            link = ok2[a_] & ok2[b_] & (hash01(i[a_], j[a_], sd_) < (0.72 if sd_ < 33 else 0.3))
            la.append(idx[a_][link])
            lb.append(idx[b_][link])
        self.tr_a, self.tr_b = np.concatenate(la), np.concatenate(lb)
        self.marrow_ph = 2 * np.pi * hash01(np.arange(len(self.marrow)), 24)
        # ion pairs along the track (cm from the centre of the view): about twice as dense in the bone
        s, x = [], -2.6
        q = 0
        while x < 2.6:
            bone = self.RIB[0] <= x <= self.RIB[1]
            x += (0.03 if bone else 0.062) * (0.6 + 0.8 * float(hash01(q, 25)))
            s.append(x)
            q += 1
        self.ion_s = np.array(s)
        q = np.arange(len(s))
        self.ion_bone = (self.ion_s >= self.RIB[0]) & (self.ion_s <= self.RIB[1])
        self.ion_d = (7.0 + 19.0 * hash01(q, 26)) * np.where(hash01(q, 27) < 0.5, -1.0, 1.0)     # px, to one side

    def _build_cells(self):
        """The cells of the CELLS view: seeds on a jittered triangular lattice (um, from the centre of the view),
        the way each one wanders, and the cells the track goes through, in the order it meets them."""
        P = self.P
        a = self.CELL
        ppu = self.CELL_PPU
        half_w = max(P.fx - WALL[0], WALL[2] - P.fx)
        nj = int((WALL[3] - WALL[1]) / ppu / (a * 0.866) / 2) + 3
        ni = int(half_w / ppu / a) + nj // 2 + 4
        i, j = np.meshgrid(np.arange(-ni, ni + 1), np.arange(-nj, nj + 1))
        jx, jy = _jit(i, j, 41, 0.2 * a)
        self.c_shape = i.shape
        self.c_x, self.c_y = (i + 0.5 * j) * a + jx, j * a * 0.866 + jy          # where each seed rests
        self.c_ph = 2 * np.pi * np.stack([hash01(i, j, 51), hash01(i, j, 52), hash01(i, j, 56), hash01(i, j, 57)])
        self.c_om = 1.3 + 1.3 * np.stack([hash01(i, j, 53), hash01(i, j, 54)])   # rad / s
        self.c_amp = 0.03 * a * (0.3 + 0.7 * hash01(i, j, 55))
        self.c_nr = 2.6 + 1.2 * hash01(i, j, 43)                                 # nucleus: radius, place in its cell
        self.c_nx, self.c_ny = _jit(i, j, 47, 2.2)
        # the cells on the track = nearest seed of its points (at rest, over the field of the first frame of
        # the view): the list does not change while the seeds move, only the length of the path in each cell
        S = 560.0 / ppu
        self.c_S = S
        s = np.arange(-S, S, 4.0 / ppu)
        tx, ty = math.sin(self.TILT) * s, math.cos(self.TILT) * s
        X, Y = self.c_x.ravel(), self.c_y.ravel()
        near_ok = np.nonzero(np.abs(X) < 420.0 / ppu)[0]
        near = near_ok[np.argmin((tx[:, None] - X[None, near_ok]) ** 2 + (ty[:, None] - Y[None, near_ok]) ** 2, 1)]
        order = []
        for n_ in near:
            if n_ not in order:
                order.append(int(n_))
        self.c_order = np.array(order)
        ys = Y_MID + Y[self.c_order] * ppu
        self.c_tags = [(r, int(n_)) for r, n_ in enumerate(order) if r in (1, 5, 9) and Y_TOP + 130 < ys[r] < Y_BOT - 150]

    def _build_source(self):
        """The rays of the SOURCE block (the star of the story, after the user's sketch: a disc, rays all
        around it, each one ending in a dot): direction, length, and the rhythm on which each ray is sent again."""
        n = 28
        k = np.arange(n)
        self.src_th = 2 * np.pi * (k + 0.55 * (hash01(k, 91) - 0.5)) / n
        self.src_len = 0.68 + 0.3 * hash01(k, 92)
        self.src_T = 5.5 + 3.5 * hash01(k, 93)                       # s between two departures
        self.src_off = hash01(k, 94) * self.src_T
        self.src_wav = 2 * np.pi * hash01(k, 95)

    def _track_pt(self, s):
        """Screen point at distance s (px) along the track from the centre of the view."""
        return self.P.fx + math.sin(self.TILT) * s, Y_MID + math.cos(self.TILT) * s

    def _track(self, f, t, tick=None, gain=1.0, width_=2.2, layer="r"):
        """The red line across the view, its ticks, and a pulse running down it on every beat. The line is the
        muon (the image); its ticks are a scale: they are thrown out from the centre when the view comes on."""
        x0, y0 = self._track_pt(-560.0)
        x1, y1 = self._track_pt(560.0)
        f.segments(layer, [x0], [y0], [x1], [y1], 1.0 * gain, width=width_)
        if tick:
            s = np.arange(-552.0, 552.0, tick)
            px, py = self._track_pt(s)
            nx, ny = math.cos(self.TILT), -math.sin(self.TILT)
            ln = np.where(np.arange(len(s)) % 5 == 0, 11.0, 6.0)
            with f.build(t - self._cut(t), (self.P.fx - 300.0, Y_MID - 560.0, self.P.fx + 300.0, Y_MID + 560.0), flow="out",
                         wave=0.4, marks=False, key=56):
                f.segments(layer, px - nx * ln, py - ny * ln, px + nx * ln, py + ny * ln, 0.85 * gain)
        beat = BAR / 4
        ph = ((t - BAR0) % beat) / beat
        px, py = self._track_pt(-560.0 + 1120.0 * ph)
        f.dots("r", [px], [py], 5.0, 1.8 * gain)
        f.dots("w", [px], [py], 1.8, 1.2 * gain)

    def _scale_bar(self, f, ppu, length, label, since):
        """Bottom left of the focus bay (the track leaves it on the right). Drawn when the view comes on."""
        x0, y = self.P.fcol[0] + 12 + text_w(label, L.T_SMALL) + 14, Y_BOT - 40
        x1 = x0 + length * ppu
        box = (self.P.fcol[0] + 4, y - 22, x1 + 14, y + 22)
        plate(f, box, since - 0.2, flow="lr")
        with f.build(since - 0.2, box, flow="lr", wave=0.15, marks=False, key=55):
            f.segments("w", [x0, x0, x1], [y, y - 8, y - 8], [x1, x0, x1], [y, y + 8, y + 8], 0.95, width=L.LW)
            f.text("w", x0 - 12, y + 6, label, size=L.T_SMALL, alpha=0.9, anchor="rs")

    def _lattice(self, f):
        """Faint measuring lattice over every micro view. Called once the field is drawn: it also lays the
        black plates outside the field window."""
        gx = np.arange(WALL[0] + 56, WALL[2] - 20, 120.0)
        gy = np.arange(WALL[1] + 55, WALL[3] - 20, 120.0)
        X, Y = np.meshgrid(gx, gy)
        f.crosses("w", X.ravel(), Y.ravel(), 5.0, 0.22)
        self.P.plates(f)

    def _readouts(self, f, entries, y=None):
        """Read-out list in the side column, on ONE outlined plate: entries = [(title, lines, red, age)].
        Constructed: the plate is traced when the first entry comes, then every entry is made at its own age
        (tag pushed out, lines decoded)."""
        P = self.P
        if P.side is None:
            return
        x0, x1 = P.side
        size, lsize = L.T_TAG, L.T_SMALL
        y = y if y is not None else Y_TOP + 84
        cpl = int((x1 - x0 - 34) / (lsize * 0.61))
        hs = [size * 1.5 + len(ls) * lsize * 1.5 + 28 for _, ls, _, _ in entries]
        w = min(x1 - x0, max(text_w(ln[:cpl], lsize) for _, ls, _, _ in entries for ln in ls) + 44)
        box = (x0, y - 46, x0 + w, y - 46 + sum(hs) + 14)
        first = max(a for _, _, _, a in entries)
        plate(f, box, first)
        with f.build(first, box, flow="tb", wave=0.25, key=53):
            f.rect("w", *box, E.wl(0.4), width=E.ww(1.0))
        for n, ((title, lines, red, age), h) in enumerate(zip(entries, hs)):
            with f.build(age, (x0 + 8, y - size - 6, x0 + w, y + h - size - 12), flow="tb", wave=0.15, marks=False,
                         key=60 + n):
                f.tag("r" if red else "w", x0 + 18, y, title, size=size, pad=5)
                for k, ln in enumerate(lines):
                    f.text("w", x0 + 14, y + size * 0.5 + (k + 1) * lsize * 1.5, ln[:cpl], size=lsize, alpha=0.9)
            y += h

    # -- bone ------------------------------------------------------------------------
    RIB = (-0.63, 0.52)                       # cm along the track, from the centre of the view: its path in the rib

    def _micro_bone(self, f, t, u):
        """The rib in section, alive: the fibres ripple and carry pulses, the skin and the pleura breathe, the
        hatching of the cortical ring travels, the trabecular network works, the marrow glitters, and the muon
        leaves its ion pairs behind it on every pass (the read-outs count them)."""
        P = self.P
        fit = min(1.0, (P.half - 30.0) / 361.0)              # the rib has to fit between two towers
        ppu0 = 300.0 * fit
        ppu = ppu0 * (1.0 + 0.16 * u)                        # px per cm
        cx, cy = P.fx, Y_MID
        X = lambda wx: cx + np.asarray(wx) * ppu
        Y = lambda wy: cy + np.asarray(wy) * ppu
        wx = np.linspace((WALL[0] - cx) / ppu, (WALL[2] - cx) / ppu, 320)
        br = 2 * math.pi * (t - CUTS[2]) / 3.4               # one breath in 3.4 s

        def wavy(y, k=0, w=wx):
            return y + 0.028 * np.sin(1.3 * w + k + 0.45 * t) + 0.012 * np.sin(3.7 * w + 2.1 * k - 0.8 * t)

        ex, ey, ea, eb = 0.0, -0.06, 0.98, 0.6    # the rib
        inside = lambda x, y, s=1.0: ((x - ex) / (ea * s)) ** 2 + ((y - ey) / (eb * s)) ** 2 < 1.0
        # skin
        f.polyline("w", X(wx), Y(wavy(-1.5, 0)), 0.95, width=L.LW_BOLD)
        f.polyline("w", X(wx), Y(wavy(-1.34, 1)), E.wl(0.6), width=L.LW)
        sx_ = wx[0] + self.skin[:, 0] * (wx[-1] - wx[0])
        if E.WALL:              # the grain of the skin: one point in six, as dots
            f.dots("w", X(sx_[::6]), Y(-1.49 + 0.14 * self.skin[::6, 1]), 1.6, 1.0)
        else:
            f.pixels("w", X(sx_), Y(-1.49 + 0.14 * self.skin[:, 1]), 0.55)
        # fat: lobules, each one swelling a little on its own
        ni = int((wx[-1] - wx[0]) / 0.125 / 2) + 2
        i, j = np.meshgrid(np.arange(-ni, ni + 1), np.arange(0, 3))
        jx, jy = _jit(i, j, 5, 0.035)
        fx_, fy_ = i * 0.125 + (j % 2) * 0.06 + jx, -1.27 + j * 0.105 + jy
        fph = 6.2832 * hash01(i, j, 6)
        fx_, fy_ = fx_ + 0.007 * np.sin(1.3 * t + fph), fy_ + 0.007 * np.sin(1.05 * t + 1.7 * fph)
        f.rings("w", X(fx_.ravel()), Y(fy_.ravel()), (0.05 * ppu * (1.0 + 0.07 * np.sin(1.6 * t + 2.3 * fph))).ravel(), E.wl(0.3), width=E.ww(1.0))
        f.polyline("w", X(wx), Y(wavy(-1.02, 2)), E.wl(0.6), width=L.LW)
        # muscle fibres, parting around the bone: a ripple runs along each of them, at its own speed
        def fibre(k, w):
            """y of fibres k at the abscissas w (broadcast), and whether it is outside the bone."""
            y0, ph = self.fib_y[k], self.fib_ph[k]
            yy = y0 + 0.028 * np.sin(2.2 * w + ph[..., 0] - self.fib_w[k] * t) + 0.013 * np.sin(
                7.0 * w + ph[..., 1] - 2.3 * self.fib_w[k] * t)
            push = np.exp(-((w - ex) / (ea * 1.25)) ** 2)
            yy = yy + np.sign(y0 - ey + 1e-6) * push * np.clip(eb * 1.12 - np.abs(y0 - ey), 0, None)
            return yy, ~inside(w, yy, 1.08)

        kf = np.arange(len(self.fib_y))[:, None]
        yy, m = fibre(kf, wx[None, :])
        seg = m[:, :-1] & m[:, 1:]
        xs, ys = np.broadcast_to(X(wx)[None, :], yy.shape), Y(yy)
        if E.WALL:              # one fibre in three, at full level
            seg = seg & (np.arange(len(self.fib_y)) % 3 == 0)[:, None]
            f.segments("w", xs[:, :-1][seg], ys[:, :-1][seg], xs[:, 1:][seg], ys[:, 1:][seg], 0.9, width=E.WALL_LINE)
        else:
            f.segments("w", xs[:, :-1][seg], ys[:, :-1][seg], xs[:, 1:][seg], ys[:, 1:][seg],
                       np.broadcast_to(self.fib_i[:, None], seg.shape)[seg])
        # ... and pulses travel along some of them (a short bright stretch of the fibre, its head leading)
        w0, w1 = (WALL[0] - cx) / ppu0, (WALL[2] - cx) / ppu0
        head = w0 + ((self.dash_v * (t - T_IN) + self.dash_o * (w1 - w0)) % (w1 - w0))
        dw = head[:, None] - 0.2 * np.linspace(1.0, 0.0, 9)[None, :]                   # 0.2 cm long
        dy, dm = fibre(self.dash_k[:, None], dw)
        dseg = dm[:, :-1] & dm[:, 1:]
        di = np.broadcast_to(np.linspace(0.0, 1.0, 9)[None, :], dw.shape)
        f.segments("w", X(dw)[:, :-1][dseg], Y(dy)[:, :-1][dseg], X(dw)[:, 1:][dseg], Y(dy)[:, 1:][dseg],
                   E.wl(0.75) * di[:, :-1][dseg], E.wl(0.75) * di[:, 1:][dseg], width=2.4 if E.WALL else 1.5)
        hm = dm[:, -1]
        f.dots("w", X(dw)[:, -1][hm], Y(dy)[:, -1][hm], 1.7, 1.2)
        # bone: cortical shell (its hatching travels round the ring), trabecular sponge, marrow
        a = np.linspace(0, 2 * np.pi, 200)
        f.polyline("w", X(ex + ea * np.cos(a)), Y(ey + eb * np.sin(a)), 1.0, width=L.LW_BOLD + 0.6)
        f.polyline("w", X(ex + ea * 0.8 * np.cos(a)), Y(ey + eb * 0.74 * np.sin(a)), E.wl(0.75), width=L.LW)
        ah = np.linspace(0, 2 * np.pi, 50 if E.WALL else 150, endpoint=False) + 0.2 * t       # (the wall: one hatch in three)
        f.segments("w", X(ex + ea * 0.97 * np.cos(ah)), Y(ey + eb * 0.97 * np.sin(ah)), X(ex + ea * 0.83 * np.cos(ah)),
                   Y(ey + eb * 0.77 * np.sin(ah)), E.wl(0.5), width=E.ww(1.0))
        swell = 1.0 + 0.012 * math.sin(br)
        nx_ = ex + self.tr_x * swell + 0.012 * np.sin(1.9 * t + self.tr_ph[0])
        ny_ = ey + self.tr_y * swell + 0.012 * np.sin(1.5 * t + self.tr_ph[1])
        f.segments("w", X(nx_[self.tr_a]), Y(ny_[self.tr_a]), X(nx_[self.tr_b]), Y(ny_[self.tr_b]), E.wl(0.6), width=E.ww(1.3))
        ok = self.tr_ok
        f.dots("w", X(nx_[ok]), Y(ny_[ok]), 1.8 + 0.5 * np.sin(2.6 * t + self.tr_ph[0][ok]), E.wl(0.8))
        mk = inside(ex + self.marrow[:, 0] * ea, ey + self.marrow[:, 1] * eb, 0.74)
        if E.WALL:              # the marrow: one point in four, as dots that twinkle in size
            mk = mk & (np.arange(len(mk)) % 4 == 0)
            f.dots("w", X(ex + self.marrow[mk, 0] * ea), Y(ey + self.marrow[mk, 1] * eb),
                   1.6 * (0.55 + 0.45 * np.sin(2.4 * t + self.marrow_ph[mk])), 1.0)
        else:
            f.pixels("w", X(ex + self.marrow[mk, 0] * ea), Y(ey + self.marrow[mk, 1] * eb),
                     0.45 * (0.55 + 0.45 * np.sin(2.4 * t + self.marrow_ph[mk])))
        # pleura + lung: the alveoli fill and empty, a wave of it running along the lung
        lift = -0.012 * math.sin(br)
        f.polyline("w", X(wx), Y(wavy(0.95, 3) + lift), 0.8, width=L.LW)
        f.polyline("w", X(wx), Y(wavy(1.0, 3) + lift), E.wl(0.55), width=E.ww(1.0))
        ni = int((wx[-1] - wx[0]) / 0.09 / 2) + 2
        i, j = np.meshgrid(np.arange(-ni, ni + 1), np.arange(0, 7))
        jx, jy = _jit(i, j, 9, 0.02)
        lx, ly = i * 0.09 + (j % 2) * 0.045 + jx, 1.07 + j * 0.08 + jy
        vis = (Y(ly) < Y_BOT + 20).ravel()
        lr = 0.036 * ppu * (1.0 + 0.11 * np.sin(br - 1.1 * lx + 0.6 * j))
        if E.WALL:              # the alveoli: one in two, at full level
            vis = vis & ((i + j) % 2 == 0).ravel()
        f.rings("w", X(lx.ravel()[vis]), Y(ly.ravel()[vis] + lift), lr.ravel()[vis], E.wl(0.26), width=E.ww(1.0))
        # the track
        self._lattice(f)
        self._track(f, t, tick=15.0)
        s0, s1 = self.RIB[0] * ppu, self.RIB[1] * ppu
        x0_, y0_ = self._track_pt(s0)
        x1_, y1_ = self._track_pt(s1)
        f.rings("r", [x0_, x1_], [y0_, y1_], [11.0, 11.0], 1.0, width=1.8)
        f.segments("r", [x0_], [y0_], [x1_], [y1_], 1.2, width=4.0)
        # the ion pairs it leaves: each one is set free when the pulse of the track passes (once a beat), its
        # electron thrown to one side; they are about twice as many in the bone
        beat = BAR / 4
        sp = -560.0 + 1120.0 * (((t - BAR0) % beat) / beat)          # where the pulse is, px along the track
        si = self.ion_s * ppu
        age = ((sp - si) % 1120.0) / 1120.0 * beat
        g = np.exp(-age / 0.25) * np.where(self.ion_bone, 1.0, 0.7)
        on = (np.abs(si) < 554.0) & (g > 0.03)
        bx, by = self._track_pt(si[on])
        off = self.ion_d[on] * (1.0 - np.exp(-age[on] / 0.05))
        qx, qy = bx + math.cos(self.TILT) * off, by - math.sin(self.TILT) * off
        f.segments("r", bx, by, qx, qy, 0.7 * g[on])
        f.dots("r", qx, qy, 1.3 + 1.5 * g[on], 1.7 * g[on])
        f.dots("w", bx, by, 1.5, 1.3 * np.exp(-age[on] / 0.07))
        # the read-outs of the rib count while the pulse crosses it, and hold until it comes back
        q = float(np.clip((sp - s0) / (s1 - s0), 0.0, 1.0)) if sp >= s0 else 1.0
        age = t - CUTS[2]
        xe, ye = self._track_pt(0.82 * ppu)
        callout(f, xe, ye, "RIB_3", [f"PATH {1.15 * q:4.2f} CM", f"DE {3.9 * q:3.1f} MEV",
                                    f"~{int(130.0 * q) * 1000:07,d} IONS".replace(",", " ")], col=P.fcol, dy=44.0,
                red=True, age=age)
        # tissues on the track + their energy loss: rows at the height of each tissue, in the side column. The
        # bar of the tissue the pulse is in lights up
        if P.side is not None:
            sx0, sx1 = P.side
            bw = max(60.0, sx1 - sx0 - 330.0)
            ct = math.cos(self.TILT)
            spw = sp / ppu                      # cm along the track
            with f.build(age - 0.15, (sx0 - 10, Y_TOP + 16, sx1, Y_BOT), flow="tb", wave=0.45, marks=False, key=50) as blk:
                f.tag("w", sx0 + 6, Y_TOP + 36, "ON THE TRACK // RHO G/CM3 // DE/DX MEV/CM", size=L.T_MICRO, pad=3)
                for y_, name, rho, de, spans in ((-1.18, "FAT", "0.95", 1.8, ((-1.34 / ct, -1.02 / ct),)),
                                                 (-0.5, "MUSCLE", "1.05", 2.1, ((-1.02 / ct, self.RIB[0]),
                                                                                (self.RIB[1], 0.95 / ct))),
                                                 (ey, "BONE", "1.92", 3.4, (self.RIB,)),
                                                 (1.3, "LUNG", "0.26", 0.5, ((1.0 / ct, 560.0 / ppu),))):
                    yy = float(Y(y_))
                    if not (Y_TOP + 70 < yy < Y_BOT - 30):
                        continue
                    # seconds since the pulse was in this tissue (0 while it is there)
                    since = min(0.0 if a_ <= spw <= b_ else ((spw - b_) % (1120.0 / ppu)) / (1120.0 / ppu) * beat
                                for a_, b_ in spans)
                    lit = math.exp(-since / 0.18)
                    # its plate opens from the tick on the left when the wave gets to this row
                    plate(f, (sx0 - 10, yy - 24, sx0 + 266 + de / 3.4 * bw, yy + 24), float(blk.la(sx0, yy)), 0.2, "lr")
                    f.segments("w", [sx0 - 8], [yy], [sx0 + 10], [yy], 0.9 + 0.8 * lit, width=L.LW)
                    f.tag("r" if name == "BONE" else "w", sx0 + 20, yy + 8, name, size=L.T_LABEL, pad=4)
                    f.text("w", sx0 + 130, yy + 7, rho, size=L.T_SMALL, alpha=0.8)
                    f.rects("r" if name == "BONE" else "w", sx0 + 200, yy - 7, sx0 + 200 + de / 3.4 * bw, yy + 7,
                            0.8 + 0.75 * lit)
                    f.text("w", sx0 + 210 + de / 3.4 * bw, yy + 7, f"{de:.1f}", size=L.T_SMALL, alpha=0.85)
        self._scale_bar(f, ppu, 1.0, "1 CM", t - self._cut(t))

    # -- cells -----------------------------------------------------------------------
    CELL = 20.0                               # um between two seeds
    CELL_PPU = 4.6                            # px per um at the cut
    # the tissue moves as one: slow waves several cells long. (amplitude / CELL, kx, ky in rad / um, rad / s,
    # phase, direction of the displacement)
    CELL_WAVES = ((0.05, 0.0335, 0.0251, 2.0, 0.0, 0.6, -0.8), (0.04, -0.0286, 0.0495, 1.5, 1.9, 0.866, 0.5),
                  (0.03, 0.0286, 0.0, 2.6, 4.1, 1.0, 0.0))

    def _cell_seeds(self, t):
        """Where the seeds are at t (um): waves run through the tissue and every seed wanders a little on its
        own. All of it stays small against the pitch of the lattice: neighbours stay neighbours, so the walls
        (which are computed from the seeds at every frame) lean and stretch without a jump."""
        x0, y0 = self.c_x, self.c_y
        dx = self.c_amp * np.sin(self.c_om[0] * t + self.c_ph[0])
        dy = self.c_amp * np.sin(self.c_om[1] * t + self.c_ph[1])
        for amp, kx, ky, w, ph, ux, uy in self.CELL_WAVES:
            s = amp * self.CELL * np.sin(kx * x0 + ky * y0 - w * t + ph)
            dx, dy = dx + ux * s, dy + uy * s
        return x0 + dx, y0 + dy

    def _micro_cells(self, f, t, u):
        P = self.P
        ppu = self.CELL_PPU * (1.0 + 0.16 * u)             # px per um
        cx, cy = P.fx, Y_MID
        px, py = self._cell_seeds(t)
        # Voronoi from the (jittered, moving) triangular lattice: circumcentres of its two triangle families
        A = (slice(None, -1), slice(None, -1))
        Bx, By = px[:-1, 1:], py[:-1, 1:]           # P(i+1, j)
        Cx, Cy = px[1:, :-1], py[1:, :-1]           # P(i, j+1)
        Dx, Dy = px[1:, 1:], py[1:, 1:]             # P(i+1, j+1)
        c1x, c1y = _circ(px[A], py[A], Bx, By, Cx, Cy)
        c2x, c2y = _circ(Bx, By, Dx, Dy, Cx, Cy)
        S = lambda wx, wy: (cx + wx * ppu, cy + wy * ppu)
        for x0, y0, x1, y1 in ((c1x, c1y, c2x, c2y), (c2x[:, :-1], c2y[:, :-1], c1x[:, 1:], c1y[:, 1:]),
                               (c2x[:-1, :], c2y[:-1, :], c1x[1:, :], c1y[1:, :])):
            X0, Y0 = S(x0.ravel(), y0.ravel())
            X1, Y1 = S(x1.ravel(), y1.ravel())
            f.segments("w", X0, Y0, X1, Y1, E.wl(0.5), width=E.ww(1.3))
        # the nuclei: the points the walls follow (each one sits a little off the seed of its cell)
        NX, NY = S(px.ravel(), py.ravel())
        UX, UY = NX + self.c_nx.ravel() * ppu, NY + self.c_ny.ravel() * ppu
        nr = self.c_nr.ravel() * ppu
        vis = (NX > WALL[0] - 30) & (NX < WALL[2] + 30) & (NY > WALL[1] - 30) & (NY < WALL[3] + 30)
        f.rings("w", UX[vis], UY[vis], nr[vis], E.wl(0.42), width=E.ww(1.0))
        f.dots("w", UX[vis], UY[vis], 1.8, E.wl(0.7))
        # the path of the track in each cell it meets, from wall to wall: the wall between two cells is where
        # the track is as far from one seed as from the other, so the lengths follow the seeds without a step
        o = self.c_order
        ox_, oy_ = px.ravel()[o], py.ravel()[o]
        dx_, dy_ = math.sin(self.TILT), math.cos(self.TILT)
        den = 2.0 * (dx_ * (ox_[1:] - ox_[:-1]) + dy_ * (oy_[1:] - oy_[:-1]))
        b = ((ox_[1:] ** 2 + oy_[1:] ** 2) - (ox_[:-1] ** 2 + oy_[:-1] ** 2)) / np.where(np.abs(den) < 1e-6, 1e-6, den)
        edges = np.r_[-self.c_S, np.maximum.accumulate(np.clip(b, -self.c_S, self.c_S)), self.c_S]
        path = edges[1:] - edges[:-1]                              # um, in the order the muon meets the cells
        self._lattice(f)
        # the cells on the track: each one lights up when the pulse of the track goes through it
        beat = BAR / 4
        lead = -560.0 + 1120.0 * (((t - BAR0) % beat) / beat)
        since = ((lead - 0.5 * (edges[1:] + edges[:-1]) * ppu) % 1120.0) / 1120.0 * beat
        glow = np.maximum(0.25, np.exp(-since / 0.16))
        nc = self.c_shape[1]
        r, c = o // nc, o % nc
        ok = (r >= 1) & (r < c1x.shape[0]) & (c >= 1) & (c < c1x.shape[1])
        r, c = r[ok], c[ok]
        vx = np.stack([c1x[r, c], c2x[r, c - 1], c1x[r, c - 1], c2x[r - 1, c - 1], c1x[r - 1, c], c2x[r - 1, c]], 1)
        vy = np.stack([c1y[r, c], c2y[r, c - 1], c1y[r, c - 1], c2y[r - 1, c - 1], c1y[r - 1, c], c2y[r - 1, c]], 1)
        VX, VY = S(vx, vy)
        gl = np.broadcast_to(glow[ok][:, None], VX.shape)
        f.segments("r", VX.ravel(), VY.ravel(), np.roll(VX, -1, 1).ravel(), np.roll(VY, -1, 1).ravel(),
                   (0.5 + 0.9 * gl).ravel(), width=2.4)
        f.rings("r", UX[o[ok]], UY[o[ok]], nr[o[ok]], 0.5 + 0.6 * glow[ok])
        self._track(f, t, width_=2.6)
        for q, (rank, n_) in enumerate(self.c_tags):
            side = 1 if q % 2 == 0 else -1
            callout(f, float(NX[n_]) + side * 34, float(NY[n_]), f"CELL {rank + 1:02d}",
                    [f"PATH {path[rank]:4.1f} UM", f"DE {path[rank] * 0.2:.2f} KEV", f"{int(path[rank] * 6.6):03d} ION PAIRS"],
                    col=P.fcol, prefer=side, dy=-46.0 * side, red=True, age=t - CUTS[3] - 0.15 * q, elbow=36.0)
        # ion pairs per crossed cell, in the order the muon met them (side column)
        if P.side is not None:
            sx0, sx1 = P.side
            yt = Y_TOP + 96
            n_show = min(len(o), 22)
            plate(f, (sx0, yt - 58, sx1, yt + n_show * 17 + 136), t - CUTS[3] - 0.15, 0.4)
            with f.build(t - CUTS[3] - 0.15, (sx0, yt - 58, sx1, yt + n_show * 17 + 136), flow="tb", wave=0.5, key=51):
                f.rect("w", sx0, yt - 58, sx1, yt + n_show * 17 + 136, E.wl(0.4), width=E.ww(1.0))
                sx0, sx1 = sx0 + 16, sx1 - 16
                f.tag("w", sx0 + 6, yt - 26, fit_text(["ION PAIRS PER CELL // IN THE ORDER IT MET THEM",
                                                       "ION PAIRS PER CELL"], sx1 - sx0, L.T_MICRO), size=L.T_MICRO, pad=3)
                bw = sx1 - sx0 - 60
                rk = np.arange(n_show)
                hud.bars(f, "r", sx0 + 44, yt + rk * 17, sx0 + 44 + np.minimum(bw, path[:n_show] * 6.6 * bw / 190.0),
                         yt + rk * 17 + 10, 0.75 + 0.6 * (glow[:n_show] - 0.25))
                for rank in range(n_show):
                    f.text("w", sx0 + 34, yt + rank * 17 + 11, f"{rank + 1:02d}", size=L.T_MICRO, alpha=0.7, anchor="rs")
                yn = yt + n_show * 17 + 44
                for k, ln in enumerate([f"{len(o)} CELLS ON THE TRACK", f"{path.sum() * 6.6:.0f} ION PAIRS",
                                        "NO CELL NOTICED"]):
                    f.text("r" if k == 2 else "w", sx0 + 6, yn + k * 32, ln, size=L.T_TAG, alpha=0.92)
        self._scale_bar(f, ppu, 50.0, "50 UM", t - self._cut(t))

    # -- dna ---------------------------------------------------------------------------
    DNA_PPU = 100.0                           # px per nm at the cut
    DNA_AXIS = 1.3                            # nm: the axis of the helix runs under the centre of the view
    DNA_PITCH, DNA_RISE = 3.4, 0.34           # nm per turn, nm per base pair: 10 pairs a turn
    DNA_MINOR = 1.2                           # nm between the two strands along the axis (minor groove; major: 2.2)
    DNA_SEQ = "TTAGGG"                        # the repeat at the end of every human chromosome (telomere)
    DNA_PAIR = {"A": "T", "T": "A", "G": "C", "C": "G"}
    DNA_TURN = 0.85                           # rad / s: it turns about its axis

    def _delta(self, n):
        """The first n (fractional) steps of the path of the electron that was set free: it grows smoothly."""
        n = float(np.clip(n, 1.0, len(self.delta)))
        k = int(n)
        d = self.delta[:k]
        if k < len(self.delta) and n > k:
            d = np.vstack([d, self.delta[k - 1] + (self.delta[k] - self.delta[k - 1]) * (n - k)])
        return d

    def _micro_dna(self, f, t, u):
        """DNA as a diagram: ONE double helix across the field, flat. Two strands (the one in front heavy, the
        one behind light) between two rules; a rung per base pair, purine heavy and pyrimidine light, with its
        two letters; the hydrogen bonds as a lane of bars; the grooves named as they pass; dimension lines for
        the figures of the panel. It turns slowly about its axis, a read head runs along the sequence, and the
        muon crosses between two pairs."""
        P = self.P
        ppu = self.DNA_PPU * (1.0 + 0.16 * u)             # px per nm
        cx, cy = P.fx, Y_MID
        age = t - CUTS[4]
        ya, R = cy + self.DNA_AXIS * ppu, 1.0 * ppu
        top, bot = ya - R, ya + R
        w0, w1 = P.win[0] + 8.0, P.win[1] - 8.0           # the helix crosses the whole field window
        k = 2 * math.pi / self.DNA_PITCH
        dg = k * self.DNA_MINOR
        phi = self.DNA_TURN * age + 0.6
        # the two strands
        s = np.arange((w0 - cx) / ppu, (w1 - cx) / ppu, 0.04)          # nm along the axis
        xs = cx + s * ppu
        for d in (0.0, dg):
            z = 0.5 + 0.5 * np.sin(k * s + phi + d)                    # 1 = in front
            y = ya - R * np.cos(k * s + phi + d)
            f.segments("w", xs[:-1], y[:-1], xs[1:], y[1:], E.wl(0.4 + 0.85 * z[:-1]), E.wl(0.4 + 0.85 * z[1:]),
                       width=1.4 + 2.6 * z[:-1])
        f.segments("w", [w0, w0], [top, bot], [w1, w1], [top, bot], E.wl(0.3), width=E.ww(1.0))
        # a rung per base pair: the purine (A, G) is the heavy, longer half, the pyrimidine (T, C) the light one
        sp = self.DNA_RISE * ppu
        n = np.arange(int(math.ceil((w0 + 6 - cx) / sp - 0.5)), int(math.floor((w1 - 6 - cx) / sp - 0.5)) + 1)
        xn = cx + (n + 0.5) * sp
        a1 = k * (n + 0.5) * self.DNA_RISE + phi
        y1, y2 = ya - R * np.cos(a1), ya - R * np.cos(a1 + dg)
        zz = 0.5 + 0.25 * (np.sin(a1) + np.sin(a1 + dg))
        b1 = np.array(list(self.DNA_SEQ))[n % len(self.DNA_SEQ)]
        b2 = np.array([self.DNA_PAIR[c] for c in b1])
        pur = (b1 == "A") | (b1 == "G")
        ym = y1 + (y2 - y1) * np.where(pur, 0.58, 0.42)
        gap = np.clip(0.5 * (y2 - y1), -3.0, 3.0)
        # the read head runs along the sequence from the left of the focus bay; the pair it is on is red
        n_first = int(math.ceil((P.win[0] + 14.0 - cx) / (self.DNA_RISE * self.DNA_PPU) - 0.5))      # pair 0000
        h = (P.fcol[0] + 30.0 - cx) / (self.DNA_RISE * self.DNA_PPU) + 10.5 * max(0.0, age - 0.5)
        cur = int(math.floor(h))
        on = (n == cur) & (age >= 0.5)
        for lay, m, gain in (("w", ~on, 1.0), ("r", on, 1.5)):
            f.segments(lay, xn[m], y1[m], xn[m], (ym - gap)[m], gain * (0.45 + 0.55 * zz[m]), width=np.where(pur, 5.0, 2.0)[m])
            f.segments(lay, xn[m], (ym + gap)[m], xn[m], y2[m], gain * (0.45 + 0.55 * zz[m]), width=np.where(pur, 2.0, 5.0)[m])
        self._lattice(f)
        # what is written on it is made at the cut, from the left: letters, bonds, dimensions
        fs = 24.0
        with f.build(age - 0.1, (w0, top - 70.0, w1, bot + 124.0), flow="lr", wave=0.55, marks=False, key=57):
            xa, xb = P.fcol[0] + 4, P.fcol[1] - 4
            for xx, c1, c2, o in zip(xn, b1, b2, on):
                if xa - 12 < xx < xa + 32 or xb - 32 < xx < xb + 12:           # the ends of the rows carry their marks
                    continue
                f.text("r" if o else "w", xx, top - 14, str(c1), size=fs, alpha=1.0 if o else 0.82, anchor="ms", bold=True)
                f.text("r" if o else "w", xx, bot + 14 + fs, str(c2), size=fs, alpha=1.0 if o else 0.82, anchor="ms",
                       bold=True)
            # hydrogen bonds, one line each: two between A and T, three between G and C
            gc = (b1 == "G") | (b1 == "C")
            yb = bot + 22 + fs
            for row in range(3):
                for lay, m in (("w", ~on & (gc | (row < 2))), ("r", on & (gc | (row < 2)))):
                    hud.bars(f, lay, xn[m] - 0.3 * sp, yb + 5 * row, xn[m] + 0.3 * sp, yb + 5 * row + 3, 0.95)
            # the two strands run opposite ways
            for yy, left, right in ((top - 14, "5'", "3'"), (bot + 14 + fs, "3'", "5'")):
                f.tag("w", xa, yy, left, size=L.T_MICRO, pad=3)
                f.tag("w", xb, yy, right, size=L.T_MICRO, pad=3, anchor="rs")
            f.tag("w", xa, yb + 12, "H-BONDS", size=L.T_MICRO, pad=3)
            # 2.0 nm wide
            xd = xa + 82.0
            f.segments("w", [xd, xd - 7, xd - 7], [top, top, bot], [xd, xd + 7, xd + 7], [bot, top, bot], 0.95, width=L.LW)
            f.tag("w", xd + 8, ya + 6, "2.0 NM", size=L.T_MICRO, pad=3)
            # one turn = 10 pairs = 3.4 nm, and the rise from a pair to the next: measured on the rungs
            yd = yb + 40
            sp1 = self.DNA_RISE * self.DNA_PPU * 1.16                # the pitch of the rungs at the end of the view
            if cx - P.fcol[0] >= 9.5 * sp1 + 6.0 and P.fcol[1] - cx >= 5.5 * sp1 + 160.0:
                x0_, x1_ = cx - 9.5 * sp, cx + 0.5 * sp
                f.segments("w", [x0_, x0_, x1_], [yd, yd - 16, yd - 16], [x1_, x0_, x1_], [yd, yd + 7, yd + 7], 0.95, width=L.LW)
                f.text("w", 0.5 * (x0_ + x1_), yd + 24, fit_text(["ONE TURN // 3.4 NM // 10 PAIRS", "3.4 NM // 10 PAIRS", "3.4 NM"],
                                                                 x1_ - x0_, L.T_MICRO), size=L.T_MICRO, alpha=0.9, anchor="ms")
                x0_, x1_ = cx + 3.5 * sp, cx + 4.5 * sp
                f.segments("w", [x0_ - 26, x0_, x1_], [yd, yd - 16, yd - 16], [x1_ + 26, x0_, x1_], [yd, yd + 7, yd + 7], 0.95,
                           width=L.LW)
                f.text("w", x1_ + 34, yd + 5, "0.34 NM A PAIR", size=L.T_MICRO, alpha=0.9)
        # the grooves, named as they go by: between the two strands 1.2 nm one way (minor), 2.2 nm the other
        # (major). The marks ride on the top rule with the turning helix; each is made as it comes in on the
        # right of the window and taken apart before it leaves on the left
        v = self.DNA_TURN / k * ppu                        # px / s
        yg = top - 18 - fs
        m0 = math.floor((k * (w0 - cx) / ppu + phi) / (2 * math.pi))
        for m in range(m0, m0 + int((w1 - w0) / (self.DNA_PITCH * ppu)) + 3):
            x1_ = cx + (2 * math.pi * m - phi) / k * ppu           # the first strand touches the top rule
            for xa_, xb_, word in ((x1_ - self.DNA_MINOR * ppu, x1_, "MINOR 1.2 NM"),
                                   (x1_, x1_ + (self.DNA_PITCH - self.DNA_MINOR) * ppu, "MAJOR 2.2 NM")):
                if xa_ < w0 + 4 or xb_ > w1 - 4:
                    continue
                with f.build(B.io(min((w1 - 4 - xb_) / v, age - 0.3), (xa_ - w0 - 4) / v, out=0.3, span=0.5),
                             (xa_, yg - 26, xb_, yg + 8), flow="lr", wave=0.12, marks=False, key=58):
                    f.segments("w", [xa_ + 3, xa_ + 3, xb_ - 3], [yg, yg - 5, yg - 5], [xb_ - 3, xa_ + 3, xb_ - 3],
                               [yg, yg + 6, yg + 6], 0.7)
                    f.text("w", 0.5 * (xa_ + xb_), yg - 8, word if xb_ - xa_ > 112 else word[:5], size=L.T_MICRO,
                           alpha=0.85, anchor="ms")
        # the read head
        xh = cx + h * sp
        if age >= 0.5 and w0 + 10 < xh < w1 - 10:
            c1 = self.DNA_SEQ[cur % len(self.DNA_SEQ)]
            f.segments("r", [xh], [yg - 54], [xh], [yb + 16], 0.9)
            f.tag("r", xh + 5, yg - 40, f"PAIR {cur - n_first:04d} {c1}-{self.DNA_PAIR[c1]}", size=L.T_MICRO, pad=3)
        self._track(f, t, width_=2.0)
        # one ionisation in the field (in the water over the helix): on average they are 150 nm apart
        px, py = self._track_pt(-1.9 * ppu)
        a = age % (BAR / 2)
        uu = min(1.0, a / 0.9)
        ring = np.linspace(0, 2 * np.pi, 49)
        rr = 9 + 60 * (1 - (1 - uu) ** 3)
        f.polyline("r", px + rr * np.cos(ring), py + rr * np.sin(ring), (1 - uu) ** 1.4, width=1.8)
        f.dots("r", [px], [py], 5.0, 1.6)
        dl = self._delta(3 + a * 40)
        dx_, dy_ = px + dl[:, 0] * 9.0, py + dl[:, 1] * 9.0
        f.polyline("r", np.r_[px, dx_], np.r_[py, dy_], 0.9, width=1.6)
        f.dots("r", dx_[-1:], dy_[-1:], 3.0, 1.5)
        solo = P.side is None           # no list beside the view: the tags carry the data themselves
        callout(f, px, py, "ION PAIR", ["~33 EV", "NEXT +148 NM"] if solo else [], col=P.fcol, prefer=-1, dy=56.0,
                red=True, age=age - 0.3)
        qx, qy = self._track_pt(-330.0)
        callout(f, qx, qy, "MU-", ["BETWEEN THE TURNS", "MFP ~150 NM"] if solo else [], col=P.fcol, prefer=1, dy=-40.0,
                red=True, age=age - 0.6)
        self._readouts(f, [
            ("DNA", ["DOUBLE HELIX // 2.0 NM WIDE", "3.4 NM PER TURN // 10 BASE PAIRS", "2 M OF IT IN EVERY CELL"],
             False, age),
            ("ION PAIR", ["ONE ELECTRON SET FREE // ~33 EV", "THE NEXT ONE: 148 NM FURTHER"], True, age - 0.3),
            ("MU-", ["PASSES BETWEEN THE TURNS", "MEAN FREE PATH ~150 NM", "THE HELIX DOES NOT NOTICE"], True,
             age - 0.6)])
        self._scale_bar(f, ppu, 5.0, "5 NM", t - self._cut(t))

    # -- atoms -------------------------------------------------------------------------
    def _micro_atoms(self, f, t, u):
        P = self.P
        ppu = 290.0 * (1.0 + 0.16 * u)            # px per nm
        cx, cy = P.fx, Y_MID
        a = 0.31
        half_w = max(cx - WALL[0], WALL[2] - cx)
        ni, nj = int(half_w / ppu / a) + 2, int((WALL[3] - WALL[1]) / ppu / a / 2) + 2
        i, j = np.meshgrid(np.arange(-ni, ni + 1), np.arange(-nj, nj + 1))
        jx, jy = _jit(i, j, 71, 0.085)
        wob = 0.012 * np.sin(t * 9.0 + 6.28 * hash01(i, j, 73))
        ox_, oy_ = (i + 0.5 * (j % 2)) * a + jx + wob, j * a * 0.9 + jy + wob[::-1]
        ang = 2 * np.pi * hash01(i, j, 75) + 0.5 * np.sin(t * 2.0 + 6.28 * hash01(i, j, 77))
        OX, OY = cx + ox_.ravel() * ppu, cy + oy_.ravel() * ppu
        ang = ang.ravel()
        # the molecule on the track gets ionised: the one that rests nearest to a point beside the track (chosen
        # where the molecules rest, in nm, so that it stays the same one while they tremble and the view closes in)
        tpx, tpy = self._track_pt(110.0)
        rx, ry = (i + 0.5 * (j % 2)) * a + jx, j * a * 0.9 + jy
        hit = int(np.argmin((rx - (tpx + 34 - cx) / 290.0) ** 2 + (ry - (tpy - cy) / 290.0) ** 2))
        for sgn in (-1.0, 1.0):
            hx_ = OX + 0.096 * ppu * np.cos(ang + sgn * 0.912)
            hy_ = OY + 0.096 * ppu * np.sin(ang + sgn * 0.912)
            f.segments("w", OX, OY, hx_, hy_, E.wl(0.55), width=E.ww(1.3))
            f.dots("w", hx_, hy_, 3.2, E.wl(0.85))
        f.dots("w", OX, OY, 6.0, E.wl(0.9))
        if not E.WALL:          # (the faint ring of every molecule: it cannot land, and at full level it is a wallpaper)
            f.rings("w", OX, OY, 0.14 * ppu, 0.13)
        self._lattice(f)
        self._track(f, t, width_=1.8)
        tage = t - CUTS[5]
        age = tage % BAR
        uu = min(1.0, age / 1.4)
        f.rings("r", [OX[hit]], [OY[hit]], [0.14 * ppu], 1.0, width=2.0)
        ring = np.linspace(0, 2 * np.pi, 73)
        rr = 0.14 * ppu + 150 * (1 - (1 - uu) ** 3)
        f.polyline("r", OX[hit] + rr * np.cos(ring), OY[hit] + rr * np.sin(ring), (1 - uu) ** 1.5, width=1.6)
        f.dots("r", [OX[hit]], [OY[hit]], 6.5, 1.5)
        dl = self._delta(2 + age * 22)
        room = P.fcol[1] - OX[hit] - 210
        sc = min(22.0, room / max(1.0, float(np.abs(self.delta[:, 0]).max())))
        ex_ = OX[hit] + dl[:, 0] * sc
        ey_ = OY[hit] - np.abs(dl[:, 1]) * 17.0
        f.polyline("r", np.r_[OX[hit], ex_], np.r_[OY[hit], ey_], 0.9, width=1.6)
        f.dots("r", ex_[-1:], ey_[-1:], 4.0, 1.7)
        solo = P.side is None
        callout(f, float(ex_[-1]), float(ey_[-1]), "E-", ["KNOCKED OUT", "12.6 EV"] if solo else [], col=P.fcol,
                prefer=1, dy=-50.0, red=True, age=age - 0.2, elbow=30.0)
        callout(f, float(OX[hit]), float(OY[hit]) + 0.14 * ppu, "H2O+", ["1 MOLECULE", "IN ~500"] if solo else [],
                col=P.fcol, prefer=-1, dy=110.0, red=True, age=age - 0.1)
        qx, qy = self._track_pt(-300.0)
        callout(f, qx, qy, "MU-", ["NO SIZE MEASURED", "< 1E-18 M"] if solo else [], col=P.fcol, prefer=1, dy=-40.0,
                red=True, age=tage - 0.5)
        self._readouts(f, [
            ("WATER", ["H2O // O-H 0.096 NM // 104.5 DEG", "70 % OF YOU"], False, tage),
            ("H2O+ / E-", ["ONE MOLECULE IN ~500 ON THE TRACK", "ITS ELECTRON KNOCKED OUT // 12.6 EV"], True,
             tage - 0.3),
            ("MU-", ["NO SIZE EVER MEASURED", "< 1E-18 M // POINT-LIKE", "SMALLER THAN ANYTHING IT MEETS"], True,
             tage - 0.6)])
        self._scale_bar(f, ppu, 1.0, "1 NM", t - self._cut(t))

    # -- the track alone -----------------------------------------------------------------
    def _micro_track(self, f, t, u):
        P = self.P
        hud.cross_grid(f, WALL, step=96.0, inten=0.24, origin=(P.fx, Y_MID))
        P.plates(f)
        self._track(f, t, tick=28.0, width_=2.6)
        age = t - CUTS[6]
        x, y = self._track_pt(-150.0)
        lines = ["MOMENTUM   4.02 GEV/C", "SPEED      0.99965 C", "GAMMA      38.1", "DE/DX      2.0 MEV/CM",
                 "CHARGE     -1 E", "MASS       105.658 MEV/C2", "BORN       15.2 KM UP", "AGE        1.3 US  ITS OWN CLOCK"]     # 15.2 km at 0.99965 c = 50.7 us, / gamma
        col = P.side if P.side is not None else (P.fx + 150.0, P.fcol[1])
        cw = col[1] - col[0]
        if cw < 345.0:              # no room for the list: the tag carries the essential, the name stays open
            callout(f, x, y, "TRACK 0001", ["4.02 GEV/C", "0.99965 C", "NAME ___"], col=P.fcol, prefer=1, dy=-40.0,
                    red=True, age=age)
            return
        callout(f, x, y, "TRACK 0001", [], col=P.fcol, prefer=1, dy=-40.0, red=True, age=age)
        fs = float(np.clip((cw - 50) / (32 * 0.61), 15.0, 28.0))
        xl, yl = col[0] + 22, Y_TOP + 150
        box = (col[0], yl - 2.6 * fs, col[1], yl + (len(lines) + 5.4) * fs * 1.45)
        # the list of what was measured: its plate opens and is traced, its lines are decoded from the top
        # (figures spinning); the name stays an open red block
        plate(f, box, age - 0.1, 0.45)
        with f.build(age - 0.1, box, flow="tb", wave=0.6, key=52):
            f.rect("w", *box, E.wl(0.4), width=E.ww(1.0))
            f.tag("r", xl, yl - fs * 1.2, "MEASURED ON THE TRACK", size=L.T_MICRO, pad=3)
            for k, ln in enumerate(lines):
                f.text("w", xl, yl + 8 + k * fs * 1.45, ln, size=fs, alpha=0.92)
            yn = yl + 8 + len(lines) * fs * 1.45 + 0.9 * fs
            f.text("w", xl, yn, "NAME", size=fs, alpha=0.92)
            blink = int(t * 4) % 2 == 0
            f.rects("r", xl + 11 * fs * 0.61, yn - fs * 1.2, xl + 11 * fs * 0.61 + fs * (11 if blink else 10.2),
                    yn + fs * 0.3, 1.0)
            note = ["DID NOT STOP //", "CONTINUES ~8 M INTO THE", "GROUND BELOW YOU"] if cw < 660 else \
                   ["DID NOT STOP //", "CONTINUES ~8 M INTO THE GROUND BELOW YOU"]
            for k, ln in enumerate(note):
                f.text("r", xl, yn + 2.6 * fs + k * 30, ln, size=L.T_LABEL, alpha=0.9)
