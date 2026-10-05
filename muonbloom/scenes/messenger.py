"""MESSENGER - the journey of the one that will reach us.   Sheet 1.1 VO + 1.2, 00:44 - 01:07.

The TouchDesigner sequence (rays streaking past, then a single white dot with its long straight trail,
the 'Year' counter racing to zero, red rings on the drums) kept, with one change of direction: the star
is "out there, up above", so the trail leaves towards the top of the wall and the messenger comes down
to us - where the atmosphere waits at 01:07.

  00:44  BLOOMED     the camera leaves with one of the rays; the others thin out, one by one
  00:48  MESSENGER   one dot, one trail. It is the PRIMARY cosmic ray (a proton): the muon is only born
                     in the atmosphere, in the next scene. Year counter (bottom right), journey strip with
                     what happened on Earth meanwhile, the two clocks (Earth / on board)
  00:56  DRUMS       red rings pulse around it on every accent ("travelling close to the speed of light")
  01:02  GALAXIES    the last galaxies fall behind (M87, M101, Andromeda ...), then darkness, then the limb
                     of the atmosphere rises from the bottom of the wall. A galaxy that is named keeps its
                     name ON ONE SIDE for its whole life: its lane, the side and the moments its name is made
                     and taken apart are planned once from the towers (_plan_tags), not tested frame by frame

THE TOWERS stand in front of the wall for the whole show, nobody knows yet where: all the geometry comes
from origin.Lay (ctx.focus, ctx.cols, ctx.slots_pre, ctx.cell_r). The messenger, its trail and its red
rings live in the focus bay; stars, rays, galaxies and the limb pass behind the towers; every label is
placed only where the wall is free.

NO FADE for what shows data: the journey strip, the two cards, the notes, the panels, the galaxy tags, the
marks on the trail and the labels of the limb are constructed when they appear (Frame.build, the helpers
of build.py) and taken apart when they leave (B.io). Only the image (stars, rays, galaxies, limb) fades.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import Camera, smoothstep, text_w
from ..show import Scene
from .origin import (CARD_Y, FOCAL, FOV, FRAME_CLIP, T_END, T_HERE, YEAR0, Lay, Nova, Ripples, _note_col, card, cell_right,
                     header_gap, hero_note, note, swell, title_fit)

T0, T1 = 44.0, 67.0
T_ALONE = 47.97                      # the triplet pulse starts: only the messenger is left
T_INVISIBLE = sd.said("Invisible messengers", 48.37)
T_SPEED = sd.said("Travelling close to the speed of light", 56.65)
T_SINCE = sd.said("Travelling ever since", 59.37)
T_GALAX = sd.said("Through galaxies", 62.97)
ACCENTS = np.array([56.0, 58.64, 61.30, 63.97, 66.63])      # the drum accents of 1.2
GAMMA = 3.4e6                        # a 3.2E15 eV proton
GAL_LIFE = 2.3                       # seconds a galaxy stays in the picture
TAG_AGES = (0.2, GAL_LIFE - 0.5)     # a galaxy may wear its name between these ages
TAG_MIN = 0.8                        # ... if it can keep it that long on one side; else it goes by unnamed
TAG_GOOD = 1.0                       # a lane where it keeps it this long (and is seen) is taken as it is
TAG_MAX = 3                          # names on the wall at a time
MARKS = (1, 2)                       # the distance marks of the trail that may carry a figure (150 px apart)
LIMB = ((0.0, "100 KM // KARMAN LINE"), (46.0, "50 KM // STRATOPAUSE"), (84.0, "15 KM // TROPOPAUSE"),
        (112.0, "CINCINNATI 0.147 KM"))      # the lines of the limb (px under its top) and their names
LANES = (0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12)      # how far a galaxy may be moved (in wall position u)

# years before now on a log scale: slow among the billions, a plunge at the end (TD: one decade every 5 s)
_YT = [44.0, 48.4, 56.6, 62.3, 63.5, 64.1, 65.4, 66.3, 66.8, 67.0]
_YL = [math.log10(YEAR0), 9.30, 8.30, 7.60, 6.40, 5.20, 4.00, 2.40, 1.00, 0.0]
# what happened here meanwhile (years before now)
EARTH = [(4.54e9, "EARTH"), (3.7e9, "LIFE"), (5.4e8, "ANIMALS"), (6.6e7, "DINOSAURS END"), (3.0e5, "HUMANS"),
         (238.0, "CINCINNATI")]


def years(t):
    return 10.0 ** float(np.interp(t, _YT, _YL))


def when(y):
    """Show time at which `y` years (= light-years) are left."""
    return float(np.interp(math.log10(y), _YL[::-1], _YT[::-1]))


# galaxies on the way: (name, light-years from here, kind, where it crosses the wall 0 = left .. 1 = right, size)
_GAL = [("NGC 1275 // PERSEUS A", 2.4e8, "ell", 0.62, 0.9), ("M87 // VIRGO A", 5.4e7, "ell", 0.22, 1.0),
        ("M77", 4.7e7, "spiral", 0.80, 0.8), ("M104 // SOMBRERO", 3.1e7, "edge", 0.93, 0.9),
        ("M51 // WHIRLPOOL", 2.9e7, "spiral", 0.08, 1.1), ("M101 // PINWHEEL", 2.1e7, "spiral", 0.70, 1.0),
        ("NGC 5128 // CEN A", 1.2e7, "ell", 0.97, 0.9), ("M81", 1.18e7, "spiral", 0.30, 1.0),
        ("M33 // TRIANGULUM", 2.7e6, "spiral", 0.86, 0.9), ("M31 // ANDROMEDA", 2.5e6, "spiral", 0.52, 1.7)]
# (time it falls behind, name, distance, kind, wall position, size)
PASSED = [(when(d), n, f"{d:.1E} LY".replace("E+0", "E"), k, u, sz) for n, d, k, u, sz in _GAL]


def _fmt_int(v):
    return f"{int(v):,}".replace(",", " ")


class Messenger(Scene):
    name = "messenger"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.lay = Lay(ctx)
        self.C = self.lay.C
        self.nova = Nova(self.lay)
        self.rip = Ripples(T_HERE, list(ctx.cues.onset_t), self.lay.far)
        self.cam0 = self.nova.camera(T_END)
        self.v_far = (self.C[0] - 50.0, -2500.0)                      # where the star ends up: far above the wall
        rng = np.random.default_rng(77)
        n = 2600
        self.st_t = rng.uniform(-60.0, 66.0, n)                       # when each star falls behind
        self.st_c = np.exp(rng.uniform(math.log(9e3), math.log(4e5), n))
        self.st_a = np.radians(rng.uniform(35.0, 145.0, n))
        self.st_b = rng.random(n)
        self.v48 = self._star_screen(T_ALONE)
        # galaxy glyphs in the unit disc: two-arm spiral with a bulge / elliptical / irregular
        g = np.random.default_rng(5)
        n = 620
        th = 0.4 + 4.4 * g.random(n) ** 0.8
        arm = g.integers(0, 2, n) * np.pi
        r = 0.1 + 0.9 * (th - 0.4) / 4.4
        r = r + g.normal(0, 0.035 + 0.05 * r, n)
        sp = np.stack([r * np.cos(th + arm), r * np.sin(th + arm)], 1)
        self.g_spiral = np.vstack([sp, g.normal(0, 0.11, (260, 2))])
        e = g.normal(0, 0.3, (700, 2)) * np.array([1.0, 0.72])
        self.g_ell = e[np.hypot(e[:, 0], e[:, 1]) < 1.0]
        c = np.array([[-0.35, 0.1], [0.25, -0.15], [0.05, 0.3]])
        self.g_irr = c[g.integers(0, 3, 500)] + g.normal(0, 0.2, (500, 2))
        self.gal_x = [self._gal_x(p[4]) for p in PASSED]              # where each galaxy comes up (its lane)
        self.tag_plan = self._plan_tags()
        self.mark_plan = {j: self._spans(lambda t, j=j: self._mark_free(j, t), T_ALONE + 0.09 * (j + 1) + 0.1, T1, 2.0)
                          for j in MARKS}                             # when a mark of the trail can be read
        self.limb_in = self._plan_limb()                              # when each label of the limb is written

    # ------------------------------------------------------------------ geometry
    def _travel(self, t):
        return 16000.0 * max(0.0, (t - T0) / (T_ALONE - T0)) ** 2.2

    def _shift(self, t):
        u = float(smoothstep(T0, T_ALONE + 0.4, t)) ** 1.15
        return (-350.0 * u, -1150.0 * u)

    def _camera(self, t):
        """44 - 48 s: the camera of the collapse leaves along the hero ray, looking back; the lens shifts so
        the star slides out through the top of the focus bay."""
        c0 = self.cam0
        R = c0.R.astype(np.float64)
        pos = c0.pos + self.nova.hero_dir * self._travel(t)
        dx, dy = self._shift(t)
        return Camera(pos, pos + R[2] * 1000.0, fov_deg=FOV, up=tuple(R[1]),
                      screen_center=(self.C[0] + dx, self.C[1] + dy))

    def _star_screen(self, t):
        cam = self._camera(min(t, T_ALONE))
        x, y, _, _ = cam.project(np.zeros((1, 3), np.float32))
        return float(x[0]), float(y[0])

    def _vanish(self, t):
        """Screen position of the star (vanishing point of the trail)."""
        if t <= T_ALONE:
            return self._star_screen(t)
        u = float(smoothstep(T_ALONE, 58.5, t))
        return (self.v48[0] + (self.v_far[0] - self.v48[0]) * u, self.v48[1] + (self.v_far[1] - self.v48[1]) * u)

    def _tip(self, t):
        """Screen position of the messenger: from where the collapse left it to the centre of the focus bay
        (never behind a tower)."""
        C = self.C
        hx, hy = self.nova.hero_at
        ts = [44.0, 48.0, 56.0, 62.0, 67.0]
        xs = [hx, C[0] + 0.6 * (hx - C[0]), C[0] + 0.2 * (hx - C[0]), C[0], C[0]]
        ys = [hy, C[1] - 60.0, C[1] - 140.0, C[1] - 120.0, C[1]]
        k = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
        u = float(smoothstep(ts[k], ts[k + 1], t))
        return xs[k] + (xs[k + 1] - xs[k]) * u, ys[k] + (ys[k + 1] - ys[k]) * u

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        lay = self.lay
        lay.avoid = []
        V = self._vanish(t)
        T = self._tip(t)
        # 1 - the image
        f.set_clip(*FRAME_CLIP)
        solo = float(smoothstep(T_ALONE - 0.5, T_ALONE + 0.6, t))
        self._stars(f, t, V, gain=0.25 + 0.75 * solo)
        if t < T_ALONE + 0.3:
            cam = self._camera(t)
            g = 1.0 - float(smoothstep(T_ALONE - 1.6, T_ALONE + 0.2, t))
            self.nova.draw_plane(f, cam, self.rip, t, gain=g)
            near = FOCAL * 1.5 / (FOCAL * 1.5 + self._travel(t))
            c = ctx.cues                        # as in origin._nova: the music swells the glare, it does not make it jump
            beat = 1.6 * swell(c.onset_t, c.onset_a, t, 0.1, 0.12) + swell(c.kick_t, c.kick_a, t, 0.1, 0.12)
            self.nova.draw_core(f, V[0], V[1], t, beat=beat * g, scale=near, gain=g)      # the star left behind
            self.nova.draw_rays(f, cam, t, gain=1.0 + 0.35 * g)
        gal = self._galaxies(f, t, V)
        self._limb(f, t, T)
        self._messenger(f, t, ctx, T, V)
        f.set_clip()
        # 2 - the furniture
        self._strip(f, t, ctx)
        self._card(f, t)
        # 3 - labels, only where the wall is free
        lay.take((T[0] - 255.0, T[1] - 255.0, T[0] + 255.0, T[1] + 255.0))
        if t < T_ALONE:
            hero_note(f, lay, T, 9.0, left=T_ALONE - 0.2 - t)
        self._notes(f, t)
        self._galaxy_tags(f, gal)
        self._limb_labels(f, t, T)
        self._marks(f, t, T, V)
        # 4 - bottom band: only the counters (the four panels were taken out)
        y = years(t)
        now = t >= 66.85
        cell_right(f, lay, "YEAR // BEFORE NOW", "NOW" if now else "-" + _fmt_int(y), red=True,
                   sub=f"{y:.1E} LY TO GO".replace("E+0", "E").replace("E+", "E") if not now else "ARRIVAL",
                   value_short="NOW" if now else f"-{y:.2E}".replace("E+0", "E"))
        return {"cell_age": t - T0 - 0.1}          # the counter of the show is built here, at the cut

    def _notes(self, f, t):
        """What the voice says about it, in the notes column (one block at a time; it leaves for the galaxies)."""
        lay, y = self.lay, self.C[1] - 250.0
        if T_INVISIBLE + 0.3 <= t < T_SPEED - 0.2:
            note(f, lay, y, "INVISIBLE", ["NO LIGHT // NO TRAIL // NO SOUND", "A BARE PROTON // 1.7E-15 M ACROSS",
                                          "ONE OF 1E53 // THIS ONE IS OURS"], size=30,
                 build=B.io(t - T_INVISIBLE - 0.3, T_SPEED - 0.3 - t, out=0.4))
        elif T_SPEED <= t < T_GALAX:
            a = B.io(t - T_SPEED, T_GALAX - 0.4 - t, out=0.4)
            lines = ["FRACTION OF THE SPEED OF LIGHT", "SLOWER THAN LIGHT BY 13 MICROMETRES / S",
                     ("STRAIGHT LINE: 1.8 H BEHIND ITS OWN LIGHT", min(a, t - T_SINCE)),        # "travelling ever since"
                     ("AFTER 4.8E9 YEARS", min(a, t - T_SINCE - 0.25))]
            note(f, lay, y, "SPEED", lines, size=30, red=True, big="0.999 999 999 999 96", big_red=True, build=a)

    def _stars(self, f, t, V, gain=1.0):
        """Rear window: everything we pass falls back towards the star and shrinks."""
        a = t - self.st_t
        m = a > 0.4
        r = self.st_c[m] / a[m]
        x = V[0] + r * np.cos(self.st_a[m])
        y = V[1] + r * np.sin(self.st_a[m])
        ok = (x > L.FX0) & (x < L.FX1) & (y > L.FY0) & (y < L.FY1)
        b = self.st_b[m][ok]
        dark = 1.0 - 0.75 * float(smoothstep(64.4, 65.4, t)) * (1.0 - float(smoothstep(66.2, 66.9, t)))
        f.dots("w", x[ok], y[ok], 1.0 + 1.5 * b ** 5, gain * dark * (0.3 + 0.7 * b ** 2))
        sp = self.st_c[m][ok] / a[m][ok] ** 2                # the fastest ones leave a short streak
        fast = sp > 60.0
        if fast.any():
            ln = np.minimum(sp[fast] * 0.09, 60.0)
            ca, sa = np.cos(self.st_a[m][ok][fast]), np.sin(self.st_a[m][ok][fast])
            f.segments("w", x[ok][fast], y[ok][fast], x[ok][fast] + ca * ln, y[ok][fast] + sa * ln,
                       gain * dark * 0.5, 0.0, width=1.3)

    def _gal_x(self, u):
        """Where a galaxy comes up through the bottom of the wall (u = 0 .. 1, left to right): anywhere but
        under the messenger - its way back to the star would run straight over it."""
        a0, a1 = L.FX0 + 100.0, self.C[0] - 460.0
        b0, b1 = self.C[0] + 460.0, L.FX1 - 100.0
        la, lb = max(a1 - a0, 0.0), max(b1 - b0, 0.0)
        if la + lb <= 0.0:
            return a0
        d = u * (la + lb)
        return a0 + d if d < la else b0 + (d - la)

    def _gal_at(self, k, a, xt=None):
        """Where galaxy k is `a` seconds after it came up under the wall (at xt: its lane), and its radius: it
        falls back up towards the star and shrinks."""
        tp, size = PASSED[k][0], PASSED[k][5]
        V = self._vanish(tp + a)
        xt, yt = self.gal_x[k] if xt is None else xt, L.FY1 + 160.0
        s = 1.0 / (1.0 + a / 4.0)
        return V[0] + (xt - V[0]) * s, V[1] + (yt - V[1]) * s, size * 230.0 / (1.0 + a / 1.1)

    # -- the names of the galaxies ---------------------------------------------------------
    # A galaxy crosses the wall in two seconds, and what stands in its way (a tower, the card, a note, the
    # messenger, the header) is known in advance: so which galaxy is named, ON WHICH SIDE and from when to when
    # is decided once, here, not at every frame. A name keeps its side for its whole life and travels with its
    # galaxy; it is constructed when its way is clear and taken apart before it would run into something
    # (asking "is the right free? else the left" at every frame made the names hop from one side to the other).
    # The lane of a galaxy is chosen with it: where it would rise behind a tower, or where its name would find
    # no room, it comes up a little further along the wall.
    @staticmethod
    def _tag_text(k):
        name, dist = PASSED[k][1], PASSED[k][2]
        lines = [dist + " FROM HERE", "FALLING BEHIND"]
        return name, lines, max(text_w(name, L.T_TAG) + 12.0, max(text_w(s, L.T_SMALL) for s in lines))

    @staticmethod
    def _tag_geom(x, y, R, side, w):
        """Anchor of the name on its galaxy, and the rectangle the name takes on that side."""
        ax, ay = x + side * R * 0.45, y - R * 0.2
        tx = ax + side * 88.0
        return ax, ay, (min(tx, tx + side * w) - 8.0, ay - 38.0 - 24.0, max(tx, tx + side * w) + 8.0, ay - 38.0 + 70.0)

    def _taken(self, t):
        """What stands on the wall at t, where a name must not go: the messenger, the card, the note."""
        lay = self.lay
        T = self._tip(t)
        out = [(T[0] - 255.0, T[1] - 255.0, T[0] + 255.0, T[1] + 255.0)]
        if lay.card is not None:
            out.append((lay.card[0] + 2.0, CARD_Y - 46.0, lay.card[1] + 10.0, CARD_Y + 36.0 + 8 * 26.0))
        col, _ = _note_col(lay)                             # (a note is gone when it has been taken apart: _notes)
        if col is not None and (T_INVISIBLE + 0.3 <= t < T_SPEED - 0.3 or T_SPEED <= t < T_GALAX - 0.4):
            out.append((col[0], self.C[1] - 290.0, col[1], self.C[1] - 62.0))
        return out

    @staticmethod
    def _spans(ok, t0, t1, min_len, dt=1.0 / 60.0):
        """[(from, to)]: the stretches of [t0, t1] of at least min_len seconds during which ok(t) holds. What a
        label that travels needs to know once, so that it is made when it has room and taken apart before it
        loses it - instead of being switched on and off by a test at every frame."""
        out, a = [], None
        n = int(round((t1 - t0) / dt))
        for i in range(n + 2):
            t = t0 + i * dt
            good = i <= n and ok(t)
            if good and a is None:
                a = t
            elif not good and a is not None:
                if t - dt - a >= min_len:
                    out.append((a, t - dt))
                a = None
        return out

    def _tag_rects(self, t):
        """The rectangles of the galaxy names on the wall at t."""
        out = []
        for j, (sj, a0, a1) in self.tag_plan.items():
            aj = t - PASSED[j][0]
            if a0 <= aj <= a1:
                out.append(self._tag_geom(*self._gal_at(j, aj), sj, self._tag_text(j)[2])[2])
        return out

    def _room(self, rect, t, pad=10.0):
        """True if a text rect is clear, at t, of the towers and the bands, of the messenger, the card, the note
        and of the galaxy names."""
        self.lay.avoid = []
        if not self.lay.free(*rect, pad=pad):
            return False
        x0, y0, x1, y1 = rect
        return not any(x1 > r[0] - 8.0 and x0 < r[2] + 8.0 and y1 > r[1] - 8.0 and y0 < r[3] + 8.0
                       for r in self._taken(t) + self._tag_rects(t))

    def _seen(self, k, xt):
        """Share of its life a galaxy coming up at xt spends in the open: not behind a tower, the card or a note."""
        n = ok = 0
        for a in np.arange(0.3, 1.91, 0.1):
            x, y, _ = self._gal_at(k, float(a), xt)
            if not (L.FX0 < x < L.FX1 and L.HEAD_Y < y < L.VIEW[3]):
                continue
            n += 1
            hid = any(a_ < x < b_ and top < y < bot for a_, top, b_, bot in self.lay.towers) or any(
                r[0] < x < r[2] and r[1] < y < r[3] for r in self._taken(PASSED[k][0] + float(a))[1:])
            ok += not hid
        return ok / max(n, 1)

    def _tag_free(self, k, side, w, a, plan, xt=None):
        """True if the name of galaxy k (coming up at xt) can stand on that side at age a."""
        lay = self.lay
        t = PASSED[k][0] + a
        x, y, R = self._gal_at(k, a, xt)
        ax, ay, (x0, y0, x1, y1) = self._tag_geom(x, y, R, side, w)
        if R <= 40.0 or x0 < L.COL_X0 or x1 > L.COL_X1 or y0 < L.HEAD_Y + 10.0 or y1 > L.VIEW[3]:
            return False
        if not (L.FX0 + 8.0 < ax < L.FX1 - 8.0 and L.HEAD_Y + 14.0 < ay < L.VIEW[3] - 4.0):
            return False
        for a_, top, b_, bot in lay.towers:                 # neither the name nor its anchor behind a tower
            if (x1 > a_ - 16.0 and x0 < b_ + 16.0 and y1 > top - 16.0 and y0 < bot + 16.0) or (
                    a_ - 6.0 < ax < b_ + 6.0 and top - 6.0 < ay < bot + 6.0):
                return False
        hit = lambda r, pad: x1 > r[0] - pad and x0 < r[2] + pad and y1 > r[1] - pad and y0 < r[3] + pad
        if any(hit(r, 8.0) for r in self._taken(t)):
            return False
        there = 0
        for j, (sj, a0, a1) in plan.items():                # the names already given: not on them, not too many
            aj = t - PASSED[j][0]
            if a0 <= aj <= a1:
                there += 1
                if there >= TAG_MAX or hit(self._tag_geom(*self._gal_at(j, aj), sj, self._tag_text(j)[2])[2], 12.0):
                    return False
        return True

    def _plan_tags(self):
        """Sets the lane of every galaxy (self.gal_x) and returns {galaxy: (side, age at which its name is made,
        age at which it is gone)} for those that can keep a name at least TAG_MIN seconds on one side.
        The larger galaxy has the priority (Andromeda first), then the earlier one. A galaxy keeps the lane of
        the list if it is seen there and named for TAG_GOOD seconds; else the nearest lane that does better.
        The right side is preferred unless the left one stays free clearly longer."""
        da = 1.0 / 60.0
        ages = np.arange(TAG_AGES[0], TAG_AGES[1] + 1e-9, da)
        plan = {}
        for k in sorted(range(len(PASSED)), key=lambda j: (-PASSED[j][5], PASSED[j][0])):
            w = self._tag_text(k)[2]
            best = None
            for du in LANES:
                uu = PASSED[k][4] + du
                if not 0.0 <= uu <= 1.0:
                    continue
                xt = self._gal_x(uu)
                seen = self._seen(k, xt)
                tag = None
                for side in (1.0, -1.0):
                    ok = [self._tag_free(k, side, w, float(a), plan, xt) for a in ages]
                    i0 = n = i = 0
                    while i < len(ok):                      # the longest stretch during which that side is free
                        j = i
                        while j < len(ok) and ok[j]:
                            j += 1
                        if j - i > n:
                            i0, n = i, j - i
                        i = j + 1
                    if tag is None or n > tag[2] + 15:
                        tag = (side, i0, n)
                good = tag[2] * da >= TAG_GOOD and seen >= 0.85
                score = min(tag[2] * da, TAG_GOOD) + 0.6 * seen - 0.5 * abs(du)
                if good or best is None or score > best[0] + 1e-9:
                    best = (score, xt, tag)
                if good:
                    break
            self.gal_x[k] = best[1]
            side, i0, n = best[2]
            if (n - 2) * da >= TAG_MIN:                     # (one frame of margin at each end)
                plan[k] = (side, float(ages[i0 + 1]), float(ages[i0 + n - 2]))
        return plan

    def _galaxies(self, f, t, V):
        """Galaxies falling behind: each enters under the wall when the distance left equals its distance from
        here, and falls back up towards the star. Returns what may be tagged."""
        out = []
        for k, (tp, name, dist, kind, u, size) in enumerate(PASSED):
            a = t - tp
            if a < 0.0 or a > GAL_LIFE:
                continue
            x, y, R = self._gal_at(k, a)
            if not (L.FX0 - R < x < L.FX1 + R and L.FY0 - R < y < L.FY1 + R):
                continue
            P = {"spiral": self.g_spiral, "ell": self.g_ell, "irr": self.g_irr, "edge": self.g_ell}[kind]
            tilt, squash = 0.5 + 0.9 * k, {"spiral": 0.45, "edge": 0.16, "ell": 0.8, "irr": 0.8}[kind]
            rot = tilt + (0.12 * t if kind == "spiral" else 0.0)
            c, sn = math.cos(rot), math.sin(rot)
            px = P[:, 0] * c - P[:, 1] * sn
            py = (P[:, 0] * sn + P[:, 1] * c) * squash
            c2, s2 = math.cos(tilt), math.sin(tilt)
            gx = x + R * (px * c2 - py * s2)
            gy = y + R * (px * s2 + py * c2)
            al = float(smoothstep(0.0, 0.3, a)) * (1.0 - float(smoothstep(GAL_LIFE - 0.8, GAL_LIFE, a)))
            core = np.exp(-np.hypot(P[:, 0], P[:, 1]) * 2.6)
            f.dots("w", gx, gy, np.clip(R / 80.0, 1.0, 2.6), al * (0.28 + 0.9 * core))
            f.dots("w", [x], [y], max(2.5, R * 0.05), 1.3 * al)
            if k in self.tag_plan:
                out.append((k, x, y, R, a))
        return out

    def _galaxy_tags(self, f, gal):
        """The names, as planned (_plan_tags): each on its one side, made when its way is clear, travelling
        with its galaxy, taken apart before it would meet a tower, the card, the messenger or the header."""
        for k, x, y, R, a in gal:
            side, a0, a1 = self.tag_plan[k]
            if not (a0 <= a < a1):
                continue
            name, lines, w = self._tag_text(k)
            ax, ay, rect = self._tag_geom(x, y, R, side, w)
            self.lay.take(rect)
            # (a quick build: the name has about a second on the wall - leader, tag and lines in half of one)
            with f.build(B.io(a - a0, a1 - a, out=0.25, span=0.5), (min(ax, rect[0]) - 6.0, rect[1] - 6.0,
                                                                   max(ax, rect[2]) + 6.0, max(ay, rect[3]) + 6.0),
                         flow="out", origin=(ax, ay), wave=0.14, line=0.1, cps=200.0, marks=False, key=31 + k):
                hud.callout(f, ax, ay, side * 44.0, -38.0, name, lines, side=int(side))

    def _limb_geom(self, t):
        u = float(smoothstep(65.2, T1, t)) ** 0.8
        return u, 5600.0, L.FY1 + 80.0 - 560.0 * u

    def _limb(self, f, t, T):
        """The atmosphere above this city comes up from the bottom of the wall (its apex under the messenger)."""
        u, R, top = self._limb_geom(t)
        if u <= 0:
            return
        C = self.C
        xs = np.linspace(L.FX0, L.FX1, 260)
        for j, (dy, it) in enumerate(((0.0, 1.0), (46.0, 0.6), (84.0, 0.6), (112.0, 1.0))):
            ys = top + dy + R - np.sqrt(R * R - (xs - C[0]) ** 2)
            if j in (0, 3):
                f.polyline("w", xs, ys, 0.9 * it * u, width=L.LW_BOLD if j == 0 else L.LW)
            else:
                f.dots("w", xs[::2], ys[::2], 1.5, 0.9 * it * u)
        yh = top + 84.0                             # where it will hit: straight below the messenger
        p = float(B.ease(B.lin(t, 65.2, 65.8)))     # the aim line is drawn downwards; the cross rises with the limb
        f.segments("r", [T[0]], [T[1] + 40.0], [T[0]], [T[1] + 40.0 + (yh - T[1] - 40.0) * p], 0.0, 0.8 * p, width=1.3)
        f.crosses("r", [T[0]], [yh], 14.0, 1.2, width=L.LW)

    def _limb_label_at(self, j, t):
        """Baseline of label j of the limb at t, and its rectangle (None without a column for it)."""
        col = self.lay.note or self.lay.card
        if col is None:
            return None
        u, R, top = self._limb_geom(t)
        xl = col[0] + 24.0
        yl = top + LIMB[j][0] + R - math.sqrt(max(R * R - (xl - self.C[0]) ** 2, 1.0))
        return xl, yl, (xl - 4.0, yl - 30.0, xl + text_w(LIMB[j][1], L.T_SMALL) + 6.0, yl - 6.0)

    def _plan_limb(self):
        """When each label of the limb, and the tag of the first interaction, start to be written: when their
        line has come up into the view (the limb rises from under the bottom band). They used to be decoded
        from a fixed time, out of sight, and to come up already written. None = never."""
        dt = 1.0 / 60.0
        ts = [65.2 + i * dt for i in range(int(round((T1 - 65.2) / dt)) + 1)]
        out = {}
        for j in range(len(LIMB)):
            ok = [self._limb_label_at(j, t) is not None and self._room(self._limb_label_at(j, t)[2], t) for t in ts]
            last_no = max([i for i, v in enumerate(ok) if not v], default=-1)       # free from there to the end
            out[j] = max(ts[last_no + 1], 65.5 + 0.12 * j) if last_no + 1 < len(ts) else None
        ok = [self._limb_geom(t)[0] > 0.4 and self._limb_geom(t)[2] + 84.0 + 12.0 < L.VIEW[3] for t in ts]
        last_no = max([i for i, v in enumerate(ok) if not v], default=-1)
        out["hit"] = max(ts[last_no + 1], 65.85) if last_no + 1 < len(ts) else None
        return out

    def _limb_labels(self, f, t, T):
        u, R, top = self._limb_geom(t)
        if u <= 0:
            return
        lay = self.lay
        for j, (dy, lab) in enumerate(LIMB):                # each is decoded on its line once the line is in the view
            t_in = self.limb_in[j]
            if t_in is None or t < t_in:
                continue
            xl, yl, rect = self._limb_label_at(j, t)
            txt = B.decode(lab, t - t_in, cps=90.0, key=j)   # (no spinning figures: the scene ends in a second)
            if txt.strip():
                f.occlude(rect[0], rect[1], xl + text_w(txt, L.T_SMALL) + 6.0, rect[3])
                f.text("w", xl, yl - 10.0, txt, size=L.T_SMALL, alpha=0.9)
        yh = top + 84.0
        if self.limb_in["hit"] is not None and t >= self.limb_in["hit"]:
            for s in ("FIRST INTERACTION // T-" + f"{max(T1 + 0.55 - t, 0):.2f} S", "T-" + f"{max(T1 + 0.55 - t, 0):.2f} S"):
                w = text_w(s, L.T_SMALL) + 12.0
                if T[0] + 22.0 + w < lay.focus_col[1]:
                    B.tag(f, "r", T[0] + 22.0, yh + 6.0, s, t - self.limb_in["hit"], size=L.T_SMALL, pad=4, bold=True,
                          cps=70.0, key=6)
                    break

    def _messenger(self, f, t, ctx, T, V):
        """The dot, its trail to the star, the red rings of the drums."""
        x, y = T
        dx, dy = V[0] - x, V[1] - y
        d = math.hypot(dx, dy)
        ux, uy = dx / d, dy / d
        g = float(smoothstep(T0, T0 + 0.6, t))
        f.segments("w", [x], [y], [x + ux * min(d, 4200.0)], [y + uy * min(d, 4200.0)], 1.25, 0.55, width=3.2)
        f.segments("r", [x], [y], [x + ux * 520.0], [y + uy * 520.0], 1.1, 0.0, width=1.8)
        if t > T_ALONE:                             # distance marks along the trail: thrown out one after the other
            s = 150.0 * np.arange(1, 9)
            ak = t - T_ALONE - 0.09 * np.arange(8)
            on = ak >= 0.0
            hl = 11.0 * (1.0 + 1.6 * np.exp(-np.maximum(ak, 0.0) / 0.07))
            mx, my = x + ux * s, y + uy * s
            f.segments("w", (mx - uy * hl)[on], (my + ux * hl)[on], (mx + uy * hl)[on], (my - ux * hl)[on], 0.85, width=1.3)
        # red rings: one family per drum accent, a small one per kick
        if t >= ACCENTS[0] - 0.02:
            for ta in ACCENTS[ACCENTS <= t]:
                a = t - ta
                if a > 3.2:
                    continue
                rr = np.array([34.0, 52.0, 60.0, 84.0, 96.0, 128.0, 150.0, 186.0, 240.0]) * (1.0 - math.exp(-a / 0.12)) * (1 + 0.12 * a)
                ii = 1.7 * math.exp(-a / 1.5) * np.array([1.0, 1.0, 0.7, 0.95, 0.7, 0.85, 0.6, 0.6, 0.4])
                f.rings("r", np.full(9, x), np.full(9, y), rr, ii, width=2.2)
            kt, ka = ctx.cues.kicks(t - 0.5, t + 1e-6)
            for tk, ak in zip(kt, ka):
                a = t - tk
                f.rings("r", [x], [y], [24.0 + 70.0 * a / 0.5], 0.7 * min(ak, 1.2) * (1 - a / 0.5), width=1.3)
        pulse = min(1.0, ctx.cues.kick(t, 0.1))
        f.dots("r", [x], [y], 13.0, 0.9 * (1.0 - g))
        f.dots("w", [x], [y], 7.5 + 5.5 * g + 3.0 * pulse * g, 1.7)
        f.rings("r", [x], [y], [19.0 + 3.0 * pulse], 1.2 * g, width=2.0)
        an = B.io(t - T_ALONE - 0.3, 66.3 - t, out=0.25, span=0.5)
        if an > 0.0:                                # its name, hugging it (the data are on the card): made, then taken apart
            sd_ = 1.0 if self.lay.focus_col[1] - x > 130.0 else -1.0
            B.pen(f, "r", x + sd_ * 16.0, y + 16.0, x + sd_ * 36.0, y + 36.0, B.lin(an, 0.0, 0.14), 0.9, width=L.LW, head=2.6)
            B.tag(f, "r", x + sd_ * 42.0, y + 54.0, "P+", an, t0=0.12, size=L.T_TAG, pad=5, bold=True,
                  anchor="ls" if sd_ > 0 else "rs", key=3)

    def _mark_at(self, j, t):
        """Where mark j of the trail is at t, and the rectangle of its figure."""
        x, y = self._tip(t)
        V = self._vanish(t)
        d = math.hypot(V[0] - x, V[1] - y)
        mx, my = x + (V[0] - x) / d * 150.0 * (j + 1), y + (V[1] - y) / d * 150.0 * (j + 1)
        return mx, my, (mx + 14.0, my - 12.0, mx + 26.0 + text_w("0.00E9 LY BEHIND", L.T_MICRO), my + 12.0)

    def _mark_free(self, j, t):
        return self._room(self._mark_at(j, t)[2], t)

    def _marks(self, f, t, T, V):
        """Light-years behind, written along the trail while the wall is free there (mark_plan): the figure is
        decoded when it has room for two seconds at least and taken apart before the trail carries it under
        the header or behind something - it does not come on written, nor go out in one frame."""
        if t <= T_ALONE:
            return
        done = YEAR0 - years(t)
        for j in MARKS:
            for t0, t1 in self.mark_plan[j]:
                if t0 <= t < t1:
                    mx, my, _ = self._mark_at(j, t)
                    s = f"{done * (1 - 0.11 * (j + 1)):.2E} LY BEHIND".replace("E+0", "E")
                    f.text("w", mx + 20.0, my + 6.0, B.resolve(s, B.io(t - t0, t1 - t, out=0.25, span=0.5), 80.0, key=j),
                           size=L.T_MICRO, alpha=0.75)

    # ------------------------------------------------------------------ HUD
    def _strip(self, f, t, ctx):
        """The journey: time runs left to right, the years before now are written on it (billions take most of
        the strip, the last hundred thousand years a sliver), with what happened on Earth meanwhile."""
        x0, y0, x1, y1 = L.STRIP
        f.occlude(x0, y0, x1, y1)
        ix0, iy0, ix1, iy1, yb = hud.strip_base(f, title="JOURNEY // YEARS BEFORE NOW // WHAT HAPPENED HERE MEANWHILE",
                                                age=t - T0)
        with f.build(t - T0 - 0.3, L.STRIP, flow="lr", wave=0.5, marks=False, bars="down", key=12):
            self._strip_body(f, t, ctx, ix0, iy0, ix1, iy1, yb)
        header_gap(f, ctx, t)

    def _strip_body(self, f, t, ctx, ix0, iy0, ix1, iy1, yb):
        X = lambda tt: ix0 + (np.asarray(tt, np.float64) - T0) / (T1 - T0) * (ix1 - ix0)
        n = int((ix1 - ix0) / 5)
        lv = ctx.cues.loud_curve(T0, T1, n)
        tb = T0 + (np.arange(n) + 0.5) / n * (T1 - T0)
        xb = X(tb)
        f.rects("w", xb, iy0 + 1, xb + 2, iy0 + 2 + 30 * lv ** 1.5, np.where(tb <= t, 0.9, 0.28))
        for d in range(9, -1, -1):                                  # decades of years
            xd = float(X(when(10.0 ** d)))
            f.segments("w", [xd], [yb - 14], [xd], [yb + 14], 0.95, width=L.LW)
            if d >= 7:
                f.text("w", xd + 5, yb - 9, f"1E{d} YR", size=L.T_MICRO, alpha=0.8)
            for m in (2, 5):
                if d < 9 or m * 10.0 ** d < YEAR0:
                    xm = float(X(when(m * 10.0 ** d)))
                    f.segments("w", [xm], [yb - 7], [xm], [yb + 7], 0.6)
        y = years(t)
        for k, (ye, name) in enumerate(EARTH):
            te = when(ye)
            xe = float(X(te))
            passed = t >= te
            late = xe > ix1 - 330
            f.segments("w", [xe], [yb + 6], [xe], [yb + 20], 0.9 if passed else 0.4)
            f.tag("r" if passed and t - te < 2.5 else "w", xe + (-4 if late else 4), yb + 30, name,
                  size=L.T_MICRO, pad=3, alpha=1.0 if passed else 0.4, anchor="rs" if late else "ls")
        xc = float(X(t))
        hud.strip_cursor(f, xc, iy0, iy1, None)
        lab = "NOW" if t >= 66.85 else f"-{y:.2E} YR".replace("E+0", "E")
        f.tag("r", xc + 8 if xc < ix1 - 230 else xc - 8, iy1 - 4, lab, size=L.T_SMALL, pad=4, bold=True,
              anchor="ls" if xc < ix1 - 230 else "rs")

    def _card(self, f, t):
        yr = years(t)
        if t < T_ALONE:
            n = int((self.nova.fade_t > t).sum()) + 1
            card(f, self.lay, "BLOOM", [("RELEASED   1E53 NUCLEI", "1E53 NUCLEI"), ("P+ 89 %   HE 10 %   Z>2 1 %", "P+ 89 %  HE 10 %"),
                                        ("ACCELERATED IN THE SHOCK", None), ("UP TO 1E15 EV AND MORE", "UP TO 1E15 EV"),
                                        ("DIRECTIONS  ALL", None), (f"STILL IN SIGHT  {n:03d}", f"IN SIGHT  {n:03d}"),
                                        ("FOLLOWING   ONE", "FOLLOWING ONE")], t - T0, red_title=True, red_rows=(6,), cps=80.0,
                 build=True)
            return
        board = (YEAR0 - yr) / GAMMA
        card(f, self.lay, "MESSENGER", [("PRIMARY    P+  PROTON", "P+  PROTON"), ("ENERGY     3.2E15 EV", "E  3.2E15 EV"),
                                        (f"GAMMA      {GAMMA:.1E}".replace("E+0", "E"), "GAMMA 3.4E6"),
                                        ("SPEED      0.999 999 999 999 96 C", "V  0.999 999 C"),
                                        (f"TO GO      {yr:.3E} LY".replace("E+0", "E"), f"TO GO {yr:.1E} LY".replace("E+0", "E")),
                                        (f"ON BOARD   {board:07.1f} YR", f"BOARD {board:06.1f} YR"), ("CHARGE     +1 E", None),
                                        ("THE MUON IS NOT BORN YET", "NO MUON YET")], t - T_ALONE, red_rows=(4, 5),
             dim_rows=(7,), cps=80.0, build=True)

    # bottom band ------------------------------------------------------------------
    def _panels(self, f, t, ctx):
        y0, y1 = L.BOT[1], L.BOT[3]
        fns = [self._p_clocks, self._p_left, self._p_spectrum, self._p_energy]
        for k, ((x0, x1), fn) in enumerate(zip(self.lay.slots, fns)):       # each panel constructs itself at the cut
            with f.build(t - T0 - 0.2 - 0.12 * k, (x0 - 8.0, y0 - 24.0, x1 + 8.0, y1 + 8.0), key=50 + k):
                f.occlude(x0 - 10.0, y0 - 26.0, x1 + 10.0, L.FY1 - 3.0)
                fn(f, t, ctx, x0, x1, y0, y1)

    def _p_clocks(self, f, t, ctx, x0, x1, y0, y1):
        w = x1 - x0
        hud.panel_header(f, x0, x1, y0, title_fit(["TWO CLOCKS // TIME DILATION X 3.4E6", "TWO CLOCKS // X 3.4E6", "TWO CLOCKS"], w))
        done = YEAR0 - years(t)
        size = int(np.clip((w - 122.0) / (16 * 0.61), 18, 30))
        f.text("w", x0 + 4, y0 + 40, "EARTH", size=L.T_SMALL, alpha=0.8)
        f.text("w", x0 + 112, y0 + 46, _fmt_int(done) + " YR", size=size)
        f.text("r", x0 + 4, y0 + 96, "ON BOARD", size=L.T_SMALL, alpha=0.9)
        f.text("r", x0 + 112, y0 + 102, f"{done / GAMMA:,.1f} YR".replace(",", " "), size=size)
        f.rects("w", x0 + 4, y1 - 12, x0 + 4 + (w - 8) * done / YEAR0, y1 - 6, 0.9)

    def _p_left(self, f, t, ctx, x0, x1, y0, y1):
        w = x1 - x0
        hud.panel_header(f, x0, x1, y0, "LEFT BEHIND")
        ev = [(tp, name, dist) for tp, name, dist, _, _, _ in PASSED]
        ev += [(when(ye), name, f"{ye:.1E} YR AGO".replace("E+0", "E")) for ye, name in EARTH]
        ev = sorted([e for e in ev if e[0] <= t], reverse=True)[:4]
        for k, (te, name, val) in enumerate(ev):
            if text_w(name, L.T_SMALL) + text_w(val, L.T_SMALL) + 24.0 > w:
                name = name.split(" // ")[-1]
            lay_ = "r" if k == 0 and t - te < 1.5 else "w"
            al = 0.95 if k == 0 else 0.65
            f.text(lay_, x0 + 4, y0 + 40 + k * 25, B.resolve(name, t - te, 90.0, key=k), size=L.T_SMALL, alpha=al)
            f.text(lay_, x1 - 4, y0 + 40 + k * 25, B.resolve(val, t - te, 90.0, 0.15, key=10 + k, pad=True), size=L.T_SMALL,
                   alpha=al, anchor="rs")

    def _p_spectrum(self, f, t, ctx, x0, x1, y0, y1):
        w = x1 - x0
        hud.panel_header(f, x0, x1, y0, title_fit(["COSMIC-RAY SPECTRUM // FLUX VS ENERGY", "COSMIC-RAY SPECTRUM", "SPECTRUM"], w))
        gx0, gx1, gy0, gy1 = x0 + 10, x1 - 10, y0 + 22, y1 - 20
        le = np.linspace(9.0, 20.0, 120)
        lf = np.where(le < 15.5, -2.7 * (le - 9.0), -2.7 * 6.5 - 3.1 * (le - 15.5))
        xs = gx0 + (le - 9.0) / 11.0 * (gx1 - gx0)
        ys = gy0 + (-lf) / 32.0 * (gy1 - gy0)
        f.polyline("w", xs, ys, 0.95, width=L.LW)
        f.segments("w", [gx0, gx0], [gy0, gy1], [gx0, gx1], [gy1, gy1], 0.6)
        for e in (9, 12, 15, 18):
            xe = gx0 + (e - 9.0) / 11.0 * (gx1 - gx0)
            f.segments("w", [xe], [gy1], [xe], [gy1 + 6], 0.8)
            f.text("w", xe + 3, gy1 + 18, f"1E{e}", size=L.T_MICRO, alpha=0.6)
        xk = gx0 + (15.5 - 9.0) / 11.0 * (gx1 - gx0)
        yk = gy0 + (2.7 * 6.5) / 32.0 * (gy1 - gy0)
        f.dots("r", [xk], [yk], 5.0, 1.6)
        f.rings("r", [xk], [yk], [10.0 + 6.0 * min(1.0, ctx.cues.kick(t, 0.12))], 1.0, width=1.3)
        s = title_fit(["THE KNEE // 3E15 EV", "THE KNEE"], x1 - xk - 34.0, L.T_SMALL)
        if s:
            f.tag("r", xk + 18, yk - 8, s, size=L.T_SMALL, pad=4, bold=True)
        f.text("w", x1, y0 + 30, title_fit(["1 PER M2 PER YEAR", ""], w - 20), size=L.T_MICRO, alpha=0.75, anchor="rs")

    def _p_energy(self, f, t, ctx, x0, x1, y0, y1):
        hud.panel_header(f, x0, x1, y0, title_fit(["ONE PROTON // ITS ENERGY", "ENERGY"], x1 - x0))
        rows = [("ENERGY", "3.2E15 EV"), ("", "0.51 MILLIJOULE"), ("LHC BEAM", "6.8E12 EV"), ("RATIO", "X 470")]
        for k, (a, b) in enumerate(rows):
            f.text("w", x0 + 4, y0 + 40 + k * 25, a, size=L.T_SMALL, alpha=0.7)
            f.text("r" if k == 3 else "w", x1 - 4, y0 + 40 + k * 25, b, size=L.T_SMALL, alpha=0.95, anchor="rs")
