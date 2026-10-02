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
                     of the atmosphere rises from the bottom of the wall

THE TOWERS stand in front of the wall for the whole show, nobody knows yet where: all the geometry comes
from origin.Lay (ctx.focus, ctx.cols, ctx.slots_pre, ctx.cell_r). The messenger, its trail and its red
rings live in the focus bay; stars, rays, galaxies and the limb pass behind the towers; every label is
placed only where the wall is free.
"""
from __future__ import annotations

import math

import numpy as np

from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import Camera, smoothstep, text_w
from ..show import Scene
from .origin import (FOCAL, FOV, FRAME_CLIP, T_END, T_HERE, YEAR0, Lay, Nova, Ripples, card, cell_right, header_gap,
                     hero_note, note, title_fit)

T0, T1 = 44.0, 67.0
T_ALONE = 47.97                      # the triplet pulse starts: only the messenger is left
T_INVISIBLE = sd.said("Invisible messengers", 48.37)
T_SPEED = sd.said("Travelling close to the speed of light", 56.65)
T_SINCE = sd.said("Travelling ever since", 59.37)
T_GALAX = sd.said("Through galaxies", 62.97)
ACCENTS = np.array([56.0, 58.64, 61.30, 63.97, 66.63])      # the drum accents of 1.2
GAMMA = 3.4e6                        # a 3.2E15 eV proton
GAL_LIFE = 2.3                       # seconds a galaxy stays in the picture

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
            f.dots("w", [V[0]], [V[1]], 36.0 * near + 5.0, 1.6)
            f.dots("w", [V[0]], [V[1]], 120.0 * near, 0.10 * g)
            self.nova.draw_rays(f, cam, t, gain=1.0)
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
            hero_note(f, lay, T, 9.0, alpha=1.0 - float(smoothstep(T_ALONE - 1.0, T_ALONE - 0.2, t)))
        self._notes(f, t)
        self._galaxy_tags(f, gal)
        self._limb_labels(f, t, T)
        self._marks(f, t, T, V)
        # 4 - bottom band
        self._panels(f, t, ctx)
        y = years(t)
        now = t >= 66.85
        cell_right(f, lay, "YEAR // BEFORE NOW", "NOW" if now else "-" + _fmt_int(y), red=True,
                   sub=f"{y:.1E} LY TO GO".replace("E+0", "E").replace("E+", "E") if not now else "ARRIVAL",
                   value_short="NOW" if now else f"-{y:.2E}".replace("E+0", "E"))
        return {}

    def _notes(self, f, t):
        """What the voice says about it, in the notes column (one block at a time; it leaves for the galaxies)."""
        lay, y = self.lay, self.C[1] - 250.0
        if T_INVISIBLE + 0.3 <= t < T_SPEED - 0.2:
            al = 1.0 - float(smoothstep(T_SPEED - 1.0, T_SPEED - 0.3, t))
            note(f, lay, y, "INVISIBLE", ["NO LIGHT // NO TRAIL // NO SOUND", "A BARE PROTON // 1.7E-15 M ACROSS",
                                          "ONE OF 1E53 // THIS ONE IS OURS"], age=t - T_INVISIBLE - 0.3, alpha=al, size=30)
        elif T_SPEED <= t < T_GALAX:
            al = 1.0 - float(smoothstep(T_GALAX - 1.0, T_GALAX - 0.4, t))
            a = t - T_SPEED
            lines = ["FRACTION OF THE SPEED OF LIGHT", "SLOWER THAN LIGHT BY 13 MICROMETRES / S"]
            if t >= T_SINCE:
                lines += ["STRAIGHT LINE: 1.8 H BEHIND ITS OWN LIGHT", "AFTER 4.8E9 YEARS"]
            note(f, lay, y, "SPEED", lines, age=a, alpha=al, size=30, red=True, big="0.999 999 999 999 96", big_red=True)

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

    def _galaxies(self, f, t, V):
        """Galaxies falling behind: each enters under the wall when the distance left equals its distance from
        here, and falls back up towards the star. Returns what may be tagged."""
        out = []
        for k, (tp, name, dist, kind, u, size) in enumerate(PASSED):
            a = t - tp
            if a < 0.0 or a > GAL_LIFE:
                continue
            xt = self._gal_x(u)
            yt = L.FY1 + 160.0
            s = 1.0 / (1.0 + a / 4.0)
            x, y = V[0] + (xt - V[0]) * s, V[1] + (yt - V[1]) * s
            R = size * 230.0 / (1.0 + a / 1.1)
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
            if R > 40 and 0.2 < a < GAL_LIFE - 0.5:
                out.append((x, y, R, name, dist, a, al))
        return out

    def _galaxy_tags(self, f, gal):
        lay = self.lay
        shown = 0
        for x, y, R, name, dist, a, al in gal:
            lines = [dist + " FROM HERE", "FALLING BEHIND"]
            w = max(text_w(name, L.T_TAG) + 12.0, max(text_w(s, L.T_SMALL) for s in lines))
            for side in (1.0, -1.0):
                ax, ay = x + side * R * 0.45, y - R * 0.2
                tx = ax + side * 88.0
                rect = (min(tx, tx + side * w) - 8.0, ay - 38.0 - 24.0, max(tx, tx + side * w) + 8.0, ay - 38.0 + 70.0)
                if lay.free(*rect):
                    lay.take(rect)
                    hud.callout(f, ax, ay, side * 44.0, -38.0, name, lines, age=a - 0.3, alpha=al, side=int(side))
                    shown += 1
                    break
            if shown >= 3:
                break

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
        f.segments("r", [T[0]], [T[1] + 40.0], [T[0]], [yh], 0.0, 0.8 * u, width=1.3)
        f.crosses("r", [T[0]], [yh], 14.0, 1.2 * u, width=L.LW)

    def _limb_labels(self, f, t, T):
        u, R, top = self._limb_geom(t)
        if u <= 0:
            return
        lay, C = self.lay, self.C
        col = lay.note or lay.card
        if col is not None:
            xl = col[0] + 24.0
            for dy, lab in ((0.0, "100 KM // KARMAN LINE"), (46.0, "50 KM // STRATOPAUSE"), (84.0, "15 KM // TROPOPAUSE"),
                            (112.0, "CINCINNATI 0.147 KM")):
                yl = top + dy + R - math.sqrt(max(R * R - (xl - C[0]) ** 2, 1.0))
                rect = (xl - 4.0, yl - 30.0, xl + text_w(lab, L.T_SMALL) + 6.0, yl - 6.0)
                if lay.free(*rect, pad=10.0):
                    f.occlude(*rect)
                    f.text("w", xl, yl - 10.0, lab, size=L.T_SMALL, alpha=0.9 * u)
        yh = top + 84.0
        if u > 0.4:
            al = min(1.0, (u - 0.4) * 3)
            for s in ("FIRST INTERACTION // T-" + f"{max(T1 + 0.55 - t, 0):.2f} S", "T-" + f"{max(T1 + 0.55 - t, 0):.2f} S"):
                w = text_w(s, L.T_SMALL) + 12.0
                if T[0] + 22.0 + w < lay.focus_col[1] and yh + 12.0 < L.VIEW[3]:
                    f.tag("r", T[0] + 22.0, yh + 6.0, s, size=L.T_SMALL, pad=4, alpha=al, bold=True)
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
        solo = float(smoothstep(T_ALONE, T_ALONE + 1.0, t))
        if solo > 0:                                # distance marks along the trail
            s = 150.0 * np.arange(1, 9)
            mx, my = x + ux * s, y + uy * s
            f.segments("w", mx - uy * 11, my + ux * 11, mx + uy * 11, my - ux * 11, 0.85 * solo, width=1.3)
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
        if T_ALONE + 0.3 < t < 66.3:                # its name, hugging it (the data are on the card)
            sd_ = 1.0 if self.lay.focus_col[1] - x > 130.0 else -1.0
            al = float(smoothstep(T_ALONE + 0.3, T_ALONE + 0.8, t))
            f.segments("r", [x + sd_ * 16.0], [y + 16.0], [x + sd_ * 36.0], [y + 36.0], 0.9 * al, width=L.LW)
            f.tag("r", x + sd_ * 42.0, y + 54.0, "P+", size=L.T_TAG, pad=5, alpha=al, bold=True,
                  anchor="ls" if sd_ > 0 else "rs")

    def _marks(self, f, t, T, V):
        """Light-years behind, written along the trail where the wall is free."""
        solo = float(smoothstep(T_ALONE, T_ALONE + 1.0, t))
        if solo <= 0:
            return
        x, y = T
        d = math.hypot(V[0] - x, V[1] - y)
        ux, uy = (V[0] - x) / d, (V[1] - y) / d
        done = YEAR0 - years(t)
        for j in (2, 4):
            mx, my = x + ux * 150.0 * (j + 1), y + uy * 150.0 * (j + 1)
            s = f"{done * (1 - 0.11 * (j + 1)):.2E} LY BEHIND".replace("E+0", "E")
            rect = (mx + 14.0, my - 12.0, mx + 26.0 + text_w(s, L.T_MICRO), my + 12.0)
            if self.lay.free(*rect, pad=10.0):
                f.text("w", mx + 20.0, my + 6.0, s, size=L.T_MICRO, alpha=0.75 * solo)

    # ------------------------------------------------------------------ HUD
    def _strip(self, f, t, ctx):
        """The journey: time runs left to right, the years before now are written on it (billions take most of
        the strip, the last hundred thousand years a sliver), with what happened on Earth meanwhile."""
        x0, y0, x1, y1 = L.STRIP
        f.occlude(x0, y0, x1, y1)
        ix0, iy0, ix1, iy1, yb = hud.strip_base(f, title="JOURNEY // YEARS BEFORE NOW // WHAT HAPPENED HERE MEANWHILE")
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
        header_gap(f, ctx, t)

    def _card(self, f, t):
        yr = years(t)
        if t < T_ALONE:
            n = int((self.nova.fade_t > t).sum()) + 1
            card(f, self.lay, "BLOOM", [("RELEASED   1E53 NUCLEI", "1E53 NUCLEI"), ("P+ 89 %   HE 10 %   Z>2 1 %", "P+ 89 %  HE 10 %"),
                                        ("ACCELERATED IN THE SHOCK", None), ("UP TO 1E15 EV AND MORE", "UP TO 1E15 EV"),
                                        ("DIRECTIONS  ALL", None), (f"STILL IN SIGHT  {n:03d}", f"IN SIGHT  {n:03d}"),
                                        ("FOLLOWING   ONE", "FOLLOWING ONE")], t - T0, red_title=True, red_rows=(6,), cps=80.0)
            return
        board = (YEAR0 - yr) / GAMMA
        card(f, self.lay, "MESSENGER", [("PRIMARY    P+  PROTON", "P+  PROTON"), ("ENERGY     3.2E15 EV", "E  3.2E15 EV"),
                                        (f"GAMMA      {GAMMA:.1E}".replace("E+0", "E"), "GAMMA 3.4E6"),
                                        ("SPEED      0.999 999 999 999 96 C", "V  0.999 999 C"),
                                        (f"TO GO      {yr:.3E} LY".replace("E+0", "E"), f"TO GO {yr:.1E} LY".replace("E+0", "E")),
                                        (f"ON BOARD   {board:07.1f} YR", f"BOARD {board:06.1f} YR"), ("CHARGE     +1 E", None),
                                        ("THE MUON IS NOT BORN YET", "NO MUON YET")], t - T_ALONE, red_rows=(4, 5),
             dim_rows=(7,), cps=80.0)

    # bottom band ------------------------------------------------------------------
    def _panels(self, f, t, ctx):
        y0, y1 = L.BOT[1], L.BOT[3]
        fns = [self._p_clocks, self._p_left, self._p_spectrum, self._p_energy]
        for (x0, x1), fn in zip(self.lay.slots, fns):
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
            f.text(lay_, x0 + 4, y0 + 40 + k * 25, hud.typed(name, t - te, 90), size=L.T_SMALL, alpha=al)
            f.text(lay_, x1 - 4, y0 + 40 + k * 25, hud.typed(val, t - te, 90, 0.15), size=L.T_SMALL, alpha=al, anchor="rs")

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
