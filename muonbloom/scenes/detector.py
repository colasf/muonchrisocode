"""DETECTOR - the data explainer.   Sheet scene 2, 01:44 - 02:13.

The TouchDesigner idea is kept: the detector as a wireframe object (a scintillator plate with its grid
of marks and two read-out modules) turning in space, a new view on every line of the voice. Added:
the three towers come out of the dark with their detectors bracketed and WAITING, simulated muons
cross the plate, and the reply is spelled out as a chain of four read-outs:
ENERGY (the pulse) -> CODE (ADC word + the OSC float) -> LIGHT (the bloom) -> SOUND (the wave).

  01:44  These detectors. Live. Scientific apparatus...   VIEW 01 perspective, towers revealed
  01:49  Waiting                                          VIEW 02 edge-on, everything flat
  01:51  Every muon that finds them, becomes a conductor  VIEW 03 a muon crosses the plate
  01:54  The more powerful the muon, brighter, louder...  VIEW 04 top: three muons, three energies
  02:00  Energy becomes code becomes light becomes sound  VIEW 05 the chain lights up word by word
  02:05  A visitor. A messenger. A muon...                 the plate flies back into the centre tower,
                                                           one muon comes down to it -> 02:13 A BLOOM

NOTHING THAT SHOWS DATA FADES IN OR POPS IN (muonbloom/build.py). At 01:44 the furniture of the scene
constructs itself, block after block (strip, view tag and part labels, card, reply chain, bottom panels;
the three scopes are built by towers.scopes). Then, on every cut of the voice, only what is new is made:
the name of the view and the leaders to the parts. Each stage of the chain re-plots its read-out when a
muon reaches it; the callouts grow out of their point; the strip and the view tag are taken apart when the
plate flies back to its tower. The image (plate, muons, light, blooms) keeps its own life.
"""
from __future__ import annotations

import math

import numpy as np

from .. import build as B
from .. import hud, towers
from .. import layout as L
from .. import showdata as sd
from ..engine import Camera, OrthoCamera, hash01, smoothstep
from ..show import Scene

T0, T1 = 104.0, 133.0
PLATE = ((-10.0, 0.0, -10.0), (10.0, 1.0, 10.0))
MODS = [((-8.4, 1.0, -3.2), (-4.4, 2.8, 1.6)), ((4.6, 1.0, -6.6), (8.6, 2.8, 2.4))]
STAGES = ("ENERGY", "CODE", "LIGHT", "SOUND")


def _box_edges(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
                  [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], np.float32)
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return c[[a for a, _ in e]], c[[b for _, b in e]]


class Detector(Scene):
    name = "detector"
    towers = "auto"

    def __init__(self, ctx):
        super().__init__(ctx)
        rng = np.random.default_rng(12)
        g = np.linspace(-9.0, 9.0, 15)
        X, Z = np.meshgrid(g, g)
        gx, gz = X.ravel(), Z.ravel()
        h = 0.3
        y = np.full_like(gx, 1.0)
        self.cross_a = np.concatenate([np.stack([gx - h, y, gz], 1), np.stack([gx, y, gz - h], 1)]).astype(np.float32)
        self.cross_b = np.concatenate([np.stack([gx + h, y, gz], 1), np.stack([gx, y, gz + h], 1)]).astype(np.float32)
        self.pa, self.pb = _box_edges(*PLATE)
        self.ma = np.concatenate([_box_edges(*m)[0] for m in MODS])
        self.mb = np.concatenate([_box_edges(*m)[1] for m in MODS])
        # voice cues
        self.c_wait = sd.said("Waiting", 109.0)
        self.c_every = sd.said("Every muon that finds them", 111.4)
        self.c_more = sd.said("The more powerful the muon", 114.7)
        self.c_energy = sd.said("Energy becomes code", 120.0)
        self.c_visitor = sd.said("A visitor", 125.02)
        self.c_messenger = sd.said("A messenger", 127.32)
        self.c_muon = sd.said("A muon...", 130.27, nth=1)        # nth=1: "...A muon" at 01:41 is the first match
        # simulated muons crossing the plate: (time, energy 0..1, x, z, stage delay)
        self.demo = [(self.c_every + 0.7, 0.50, -1.5, 2.0, 0.12), (self.c_more + 0.6, 0.25, 2.8, -5.0, 0.12),
                     (self.c_more + 2.2, 0.55, -4.2, 5.4, 0.12), (self.c_more + 3.8, 0.95, 0.6, 0.4, 0.12),
                     (self.c_energy + 0.15, 0.76, 1.6, 3.4, 1.02)]
        self.dirs = []
        for k in range(len(self.demo)):
            th, ph = rng.uniform(0.05, 0.3), rng.uniform(0, 2 * np.pi)
            self.dirs.append(np.array([math.sin(th) * math.cos(ph), -math.cos(th), math.sin(th) * math.sin(ph)]))
        self.ph_ang = rng.uniform(0, 2 * np.pi, (len(self.demo), 46))
        self.ph_len = rng.uniform(0.3, 1.0, (len(self.demo), 46))
        self.roles = self._roles(ctx)
        # the views cut on these; the view tag and the part labels leave when the plate starts to fly back
        self.cuts = [T0, self.c_wait, self.c_every, self.c_more, self.c_energy, self.c_visitor]
        ts = np.linspace(self.c_visitor, self.c_messenger + 1.6, 2000)
        u = np.asarray(smoothstep(self.c_visitor + 0.4, self.c_messenger + 1.6, ts))
        self.t_fly = float(ts[int(np.argmax(1.0 + 7.5 * u >= 1.5))])

    @staticmethod
    def _roles(ctx):
        """Which column does what, for any tower placement: the object view in the focus bay, the reply chain
        in the widest of the others, then the card (leftmost) and the three replies in what is left."""
        cols = list(ctx.cols)
        fb = ctx.focus_bay
        obj = max(cols, key=lambda c: min(c[1], fb[1]) - max(c[0], fb[0]))
        rest = [c for c in cols if c != obj]
        chain = max(rest, key=lambda c: c[1] - c[0]) if rest else None
        rest = sorted(c for c in rest if c != chain)
        card = next((c for c in rest if c[1] - c[0] >= 250), None)
        rest = [c for c in rest if c != card]
        reps = next((c for c in rest if c[1] - c[0] >= 330), None)
        return dict(obj=obj, chain=chain, card=card, reps=reps, k=min(1.2, (obj[1] - obj[0]) / 767.0))

    # ------------------------------------------------------------------ views
    def _camera(self, t, ctx):
        b0, b1 = self.roles["obj"]
        kz = self.roles["k"]                        # the object is scaled to the bay it gets
        sc = ((b0 + b1) / 2, 668.0)
        if t < self.c_wait:
            yaw, el, name = math.radians(28 + 9 * (t - T0)), math.radians(27), "01 // PERSPECTIVE"
        elif t < self.c_every:
            cam = OrthoCamera((0.0, 0.5, 60.0), (0.0, 0.5, 0.0), scale=31.0 * kz, screen_center=sc)
            return cam, "02 // ORTHO_SIDE", sc, 1.0
        elif t < self.c_more:
            yaw, el, name = math.radians(-42 + 9 * (t - self.c_every)), math.radians(21), "03 // PERSPECTIVE"
        elif t < self.c_energy:
            cam = OrthoCamera((0.0, 60.0, 1e-3), (0.0, 0.0, 0.0), scale=27.0 * kz, up=(0.0, 0.0, -1.0), screen_center=sc,
                              roll_deg=45.0 + 3.0 * (t - self.c_more))
            return cam, "04 // ORTHO_TOP", sc, 1.0
        elif t < self.c_visitor:
            yaw, el, name = math.radians(118 + 10 * (t - self.c_energy)), math.radians(34), "05 // PERSPECTIVE"
        else:
            yaw, el, name = math.radians(168 + 14 * (t - self.c_visitor)), math.radians(30), "06 // DET_C"
        D = 96.0 / kz
        shrink = 1.0
        if t >= self.c_visitor:         # the plate flies back to where it lives: the head of the centre tower
            u = float(smoothstep(self.c_visitor + 0.4, self.c_messenger + 1.6, t))
            tw = ctx.towers["C"]
            tx, ty = tw.cx, tw.top + tw.det_h * 0.5
            sc = (sc[0] + (tx - sc[0]) * u, sc[1] + (ty - sc[1]) * u)
            shrink = 1.0 + 7.5 * u
            D *= shrink
        pos = (D * math.cos(el) * math.sin(yaw), D * math.sin(el), D * math.cos(el) * math.cos(yaw))
        return Camera(pos, (0.0, 0.5, 0.0), fov_deg=30.0, screen_center=sc), name, sc, shrink

    # ------------------------------------------------------------------ draw
    def draw(self, f, t, ctx):
        cam, vname, sc, shrink = self._camera(t, ctx)
        b0, b1 = self.roles["obj"]
        self._draw_object(f, cam, t, shrink, clip=(b0 - 16, L.HEAD_Y + 8, b1 + 16, L.VIEW[3]) if shrink == 1.0 else None,
                          anchor=(b1 - 302, 1052.0))
        if shrink < 1.5:
            self._view_tag(f, t, vname, ctx, cam)
        self._chain(f, t, ctx)
        self._left(f, t, ctx)
        self._right(f, t, ctx)
        self._bottom(f, t, ctx)
        self._visitor(f, t, ctx)
        marks = [(self.c_wait, "WAIT"), (self.c_every, "MUON"), (self.c_more, "REPLY"), (self.c_energy, "CHAIN"),
                 (self.c_visitor, "VISITOR"), (sd.T_BLOOM, "BLOOM")]
        age = B.io(t - T0, self.c_visitor - t, out=0.6)                # taken apart: the sky is clear for the visitor
        if age >= 0.0:
            hud.show_strip(f, t, ctx, "SCENE 02 // DETECTOR / DATA EXPLAINER", T0, 139.0, marks, age=age)
        # the edge meters were off during the static of 'A MUON': they are built again with the reveal
        return {"edge_kw": {"reveal": B.lin(t, T0, T0 + 0.7)}}

    def _draw_object(self, f, cam, t, shrink, clip=None, anchor=None):
        if clip:
            f.set_clip(*clip)
        g = 1.0 if shrink < 2 else 1.0 + 0.3 * (shrink - 1)          # keep it visible when tiny

        def seg(a, b, inten, layer="w", width=L.LW):
            ax, ay, _, oa = cam.project(a)
            bx, by, _, ob = cam.project(b)
            ok = oa & ob
            f.segments(layer, ax[ok], ay[ok], bx[ok], by[ok], inten, width=width)

        seg(self.pa, self.pb, 0.9, width=L.LW_BOLD if shrink < 2 else L.LW)
        seg(self.ma, self.mb, 0.85)
        if shrink < 4:
            seg(self.cross_a, self.cross_b, 0.55 / g, width=L.LW_HAIR)
        # muons through the plate
        for k, (tm, e, x, z, _) in enumerate(self.demo):
            a = t - tm
            if not (0 <= a < 3.2):
                continue
            d = self.dirs[k]
            hit = np.array([x, 0.5, z])
            fade = math.exp(-a / 1.1)
            prog = min(1.0, a / 0.16)
            A = hit - d * 34.0
            P = np.stack([A, A + d * 68.0 * prog]).astype(np.float32)
            px, py, _, ok = cam.project(P)
            if ok.all():
                f.segments("r", px[:1], py[:1], px[1:], py[1:], (0.7 + 0.8 * e) * fade, width=L.LW_BOLD)
                if prog < 1.0:
                    f.dots("r", px[1:], py[1:], 4.0, 1.8)
            if a < 0.08:
                continue
            ah = a - 0.08
            # scintillation: light spreading in the plate from the crossing point, collected by the modules
            n = int(12 + 34 * e)
            ang, ln = self.ph_ang[k, :n], self.ph_len[k, :n]
            r = (2.0 + 9.0 * ln) * (1 - math.exp(-ah / 0.12))
            ex = np.clip(x + np.cos(ang) * r, -10, 10)
            ez = np.clip(z + np.sin(ang) * r, -10, 10)
            Pa = np.repeat(hit[None], n, 0).astype(np.float32)
            Pb = np.stack([ex, np.full(n, 0.5), ez], 1).astype(np.float32)
            ax, ay, _, _ = cam.project(Pa)
            bx, by, _, _ = cam.project(Pb)
            fl = math.exp(-ah / 0.5)
            f.segments("w", ax, ay, bx, by, 0.9 * fl * (0.4 + e), 0.15 * fl)
            f.dots("w", bx, by, 1.8, 1.3 * fl)
            for (lo, hi) in MODS:           # the light that reaches the two read-outs
                c = np.array([(lo[0] + hi[0]) / 2, 1.0, (lo[2] + hi[2]) / 2], np.float32)
                cx, cy, _, _ = cam.project(c[None])
                f.segments("r", ax[:1], ay[:1], cx, cy, 0.9 * e * fl, width=L.LW)
                f.dots("r", cx, cy, 5.0, 1.6 * e * fl)
            hx, hy, _, _ = cam.project(hit[None].astype(np.float32))
            u = min(1.0, ah / 1.2)
            f.rings("r", hx, hy, [(10 + 120 * e * (1 - (1 - u) ** 3)) / shrink], (1 - u) ** 1.5 * (0.6 + e), width=L.LW)
            f.dots("w", hx, hy, 3.0, 1.6 * fl)
            if shrink < 1.5 and a < 2.6 and anchor is not None:
                hud.callout(f, float(hx[0]), float(hy[0]), anchor[0] - float(hx[0]), anchor[1] - float(hy[0]),
                            f"MU  E {e:.2f}",
                            [f"DE {1.6 + 3.4 * e:.2f} MEV", f"{int(9000 * (1.6 + 3.4 * e)):,} PHOTONS".replace(",", " ")],
                            red=True, side=1, build=B.io(ah, 2.6 - a, out=0.4, span=0.9))
        if clip:
            f.set_clip()

    def _view_tag(self, f, t, vname, ctx, cam):
        """Name of the view + part labels. Made at the cut to the scene; on every cut of the voice only what
        is new is made again (the name of the view, the leaders to the parts: a pen from each part to its
        label, which stays); all of it is taken apart when the plate starts to fly back to its tower."""
        b0, b1 = self.roles["obj"]
        x, y = b0 + 2, L.HEAD_Y + 52
        left = self.t_fly - t
        a_all = B.io(t - T0 - 0.1, left, out=0.35, span=0.7)                        # what stays from view to view
        a_cut = B.io(t - max([c for c in self.cuts if c <= t], default=T0), left, out=0.35, span=0.7)   # new in this view
        B.tag(f, "w", x, y, f"VIEW {vname}", a_cut, size=L.T_LABEL, pad=4, cps=70.0, key=3)
        f.text("w", x, y + 32, B.resolve("DET_C // 1 OF 3 IDENTICAL // SIMULATED MUONS", a_all - 0.2, 120.0, key=5),
               size=L.T_MICRO, alpha=0.7)
        # part labels: fixed positions around the view, leaders to the parts
        parts = [("SCINTILLATOR", np.array([[-9.5, 1.0, 9.5]], np.float32), (b0 + 32, 1090.0), 1),
                 ("SIPM A", np.array([[-6.4, 2.8, -0.8]], np.float32), (b0 + 32, 372.0), 1),
                 ("SIPM B", np.array([[6.6, 2.8, -2.0]], np.float32), (b1 - 32, 372.0), -1)]
        for k, (name, p, (ax, ay), side) in enumerate(parts):
            px, py, _, ok = cam.project(p)
            if not ok[0]:
                continue
            x, y = float(px[0]), float(py[0])
            pl = float(B.ease((a_cut - 0.08 * k) / 0.28))
            if pl > 0.0:
                f.dots("w", [x], [y], 2.4, 1.2)
                B.pen(f, "w", x, y, ax + side * 110, ay, pl, 0.6, width=L.LW_HAIR, head=2.6)
                if pl >= 1.0:
                    f.segments("w", [ax + side * 110], [ay], [ax + side * 96], [ay], 0.6, width=L.LW_HAIR)
            B.tag(f, "w", ax, ay + 7, name, a_all - 0.3 - 0.1 * k, size=L.T_SMALL, pad=4,
                  anchor="ls" if side > 0 else "rs", cps=70.0, key=10 + k)

    # ------------------------------------------------------------------ the chain
    def _current(self, t):
        """The demo muon whose reply is on the read-outs: (index, age) or (None, 0)."""
        k = None
        for i, m in enumerate(self.demo):
            if m[0] <= t:
                k = i
        return (k, t - self.demo[k][0]) if k is not None else (None, 0.0)

    def _chain(self, f, t, ctx):
        if self.roles["chain"] is None:
            return
        b0, b1 = self.roles["chain"]
        x0, x1 = b0 + 16, b1 - 12
        y_top, y_bot = L.HEAD_Y + 34, L.VIEW[3] - 6
        rh = (y_bot - y_top) / 4
        k, age = self._current(t)
        e = self.demo[k][1] if k is not None else 0.0
        delay = self.demo[k][4] if k is not None else 0.0
        for s, name in enumerate(STAGES):
            y0 = y_top + s * rh
            y1 = y0 + rh - 22
            a = age - s * delay if k is not None else -1.0
            live = a >= 0
            hot = live and a < 1.3
            px0, px1, py0, py1 = x0 + 190, x1 - 6, y0 + 22, y1
            # the four stages are constructed one after the other at the start of the scene ...
            with f.build(t - (T0 + 0.6) - 0.2 * s, (x0 - 8, y0 - 10, x1 + 8, y1 + 8), key=30 + s, wave=0.35):
                f.rects("w", x0, y0, x1, y0 + 4, 0.9)
                f.tag("r" if hot else "w", x0, y0 + 36, f"{s + 1:02d} {name}", size=L.T_TAG, pad=5)
                if s < 3:           # 'becomes' arrow to the next stage
                    xa = x0 + 60
                    f.segments("w", [xa, xa - 7, xa + 7], [y0 + 70, y1 + 6, y1 + 6], [xa, xa, xa],
                               [y1 + 16, y1 + 16, y1 + 16], 0.7)
                    f.text("w", xa + 16, (y0 + 70 + y1 + 16) / 2 + 6, "BECOMES", size=L.T_MICRO, alpha=0.6)
                if not live:
                    f.text("w", px0, (py0 + py1) / 2 + 6, "WAITING" if int(t * 2) % 2 else "WAITING _", size=L.T_SMALL,
                           alpha=0.5)
                    f.segments("w", [px0], [py1], [px1], [py1], 0.3)
            if not live:
                continue
            # ... and each one re-plots its read-out when a muon reaches it
            br = 0.55 + 0.45 * math.exp(-a / 0.8)
            with f.build(a, (px0 - 6, py0 - 6, px1 + 6, py1 + 6), key=40 + s, wave=0.25, line=0.22, marks=False):
                if s == 0:
                    self._stage_energy(f, px0, px1, py0, py1, e, a, br)
                elif s == 1:
                    self._stage_code(f, px0, px1, py0, py1, e, a, br)
                elif s == 2:
                    self._stage_light(f, px0, px1, py0, py1, e, a, br, t)
                else:
                    self._stage_sound(f, px0, px1, py0, py1, e, a, br)
            if s == 2:                  # the small bloom is image: it grows out of its point by itself
                self._stage_bloom(f, px0, px1, py0, py1, e, a, br, t)

    @staticmethod
    def _stage_energy(f, x0, x1, y0, y1, e, a, br):
        """The pulse of the photo-sensor: fast rise, slow decay, threshold line."""
        f.segments("w", [x0, x0], [y0, y1], [x0, x1], [y1, y1], 0.6)
        tt = np.linspace(0.0, 1.0, 180)
        v = e * (np.exp(-tt / 0.22) - np.exp(-tt / 0.018)) / 0.76
        n = int(len(tt) * min(1.0, a / 0.45)) + 1
        xs = x0 + 8 + tt * (x1 - x0 - 16)
        ys = y1 - 4 - np.clip(v, 0, 1) * (y1 - y0 - 14)
        f.polyline("w", xs[:n], ys[:n], 1.2 * br, width=L.LW_BOLD)
        yt = y1 - 4 - 0.12 * (y1 - y0 - 14)
        k = np.arange(x0 + 8, x1 - 8, 16.0)
        f.segments("r", k, np.full_like(k, yt), k + 8, np.full_like(k, yt), 0.9)
        f.text("r", x1 - 4, yt - 8, "THRESHOLD", size=L.T_MICRO, anchor="rs", alpha=0.9)
        f.text("w", x1 - 4, y0 + 16, f"PEAK {e * 3.3:.2f} V   DE {1.6 + 3.4 * e:.2f} MEV", size=L.T_SMALL,
               anchor="rs", alpha=0.95)

    @staticmethod
    def _stage_code(f, x0, x1, y0, y1, e, a, br):
        """12-bit ADC word, then the float that goes out over OSC."""
        val = int(e * 4095)
        bits = [(val >> (11 - i)) & 1 for i in range(12)]
        cw = min(46.0, (x1 - x0 - 10) / 12)
        n = int(min(12, a / 0.035))
        bx = x0 + 6 + np.arange(12) * cw
        ya, yb = y0 + 8, y0 + 8 + cw - 8
        f.segments("w", np.r_[bx, bx + cw - 8, bx + cw - 8, bx], np.r_[np.full(24, ya), np.full(24, yb)],
                   np.r_[bx + cw - 8, bx + cw - 8, bx, bx], np.r_[np.full(12, ya), np.full(24, yb), np.full(12, ya)], 0.6)
        for i in range(12):
            if i < n and bits[i]:
                f.rects("w", bx[i] + 3, y0 + 11, bx[i] + cw - 11, y0 + 5 + cw - 8, 0.95 * br + 0.1)
        ty = y0 + cw + 34
        f.text("w", x0 + 6, ty, hud.typed(f"ADC {val:04d}/4095   0x{val:03X}", a, delay=0.3), size=L.T_SMALL, alpha=0.9)
        f.text("r", x0 + 6, ty + 30, hud.typed(f"OSC  /MUON/C  ,F  {e:.3f}", a, delay=0.5), size=L.T_LABEL, alpha=0.95)

    @staticmethod
    def _stage_bloom(f, x0, x1, y0, y1, e, a, br, t):
        """A small bloom, as open as the muon was strong."""
        ox, oy = x0 + 110, y1 - 6
        grow = 1 - math.exp(-a / 0.3)
        f.set_clip(x0, y0 - 6, x1, y1 + 4)
        for j in range(7):
            s = (8 + 13 * j) * (0.35 + 1.1 * e) * grow
            phi = np.linspace(0, 2 * np.pi, 70)
            x, y = towers._loop(s, j * 1.3, 12, phi, t, 900)
            if j < 5:
                f.polyline("r", ox + x, oy + y, (1.2 - 0.12 * j) * br, width=L.LW_BOLD - 0.2 * j)
            else:
                f.dots("r", ox + x[::2], oy + y[::2], 1.6, 1.2 * br)
        f.set_clip()

    @staticmethod
    def _stage_light(f, x0, x1, y0, y1, e, a, br, t):
        """What the bloom is told: brightness, radius, how open, how long."""
        grow = 1 - math.exp(-a / 0.3)
        bx0, bx1 = x0 + 235, x1 - 6
        if bx1 - bx0 < 60:
            return
        chars = int((bx1 - bx0) / (L.T_SMALL * 0.61))
        f.text("w", bx0, y0 + 26, "BRIGHTNESS", size=L.T_MICRO, alpha=0.75)
        f.rect("w", bx0, y0 + 36, bx1, y0 + 58, 0.6)
        f.rects("r", bx0 + 3, y0 + 39, bx0 + 3 + (bx1 - bx0 - 6) * e * grow, y0 + 55, 0.95)
        lines = [f"BLOOM RADIUS {int(120 + 420 * e):3d} PX", f"OPEN {e:.2f}   DECAY 1.6 S"]
        if chars < 23:
            lines = [f"RADIUS {int(120 + 420 * e):3d} PX", f"OPEN {e:.2f}", "DECAY 1.6 S"]
        for k, ln in enumerate(lines):
            f.text("w", bx0, y0 + 88 + k * 25, hud.typed(ln[:chars], a, delay=0.2 + 0.12 * k), size=L.T_SMALL,
                   alpha=0.9 if k == 0 else 0.7)

    @staticmethod
    def _stage_sound(f, x0, x1, y0, y1, e, a, br):
        """The note it triggers: louder with the energy."""
        ym = (y0 + y1) / 2 + 4
        f.segments("w", [x0], [ym], [x1], [ym], 0.3)
        tt = np.linspace(0.0, 1.0, 420)
        n = int(len(tt) * min(1.0, a / 0.5)) + 1
        w = e * np.exp(-tt / 0.3) * np.sin(2 * np.pi * (26 + 20 * e) * tt) * (1 - np.exp(-tt / 0.006))
        xs = x0 + 6 + tt * (x1 - x0 - 12)
        f.polyline("w", xs[:n], ym - w[:n] * (y1 - y0) * 0.44, 1.0 * br, width=L.LW)
        db = 20 * math.log10(max(e, 1e-3))
        f.text("w", x1 - 4, y0 + 16, f"GAIN {db:+05.1f} DB   VEL {int(e * 127):03d}", size=L.T_SMALL,
               anchor="rs", alpha=0.95)

    # ------------------------------------------------------------------ columns
    def _left(self, f, t, ctx):
        if self.roles["card"] is None:
            return
        c0, c1 = self.roles["card"]
        x = c0 + 4.0
        y = L.HEAD_Y + 86
        wide = c1 - c0 >= 380
        ts = int(min(62, (c1 - c0 - 30) / (9 * 0.61)))
        B.tag(f, "w", x, y, "DETECTORS", t - T0, t0=0.1, size=ts, pad=10, bold=True, cps=45.0, key=50, commit=True)
        waiting = t < sd.T_BLOOM
        rows = ["COUNT     3  (L / C / R)", "TYPE      PLASTIC SCINTILLATOR", "READOUT   SIPM x 2 // COINCIDENCE",
                "SIGNAL    1 FLOAT PER TOWER", "LINK      OSC  /MUON/L /C /R", "RATE      ~0.2 MUONS /S EACH",
                f"STATE     {'WAITING' if waiting else 'LIVE'}"]
        chars = int((c1 - c0 - 4) / (L.T_SMALL * 0.61))
        # the rows are decoded one after the other, top to bottom (they used to pop in, one every 0.3 s)
        with f.build(t - T0 - 0.35, (x - 8, y + 38, c1 + 4, y + 62 + 6 * 25.5 + 12), flow="tb", wave=1.4, cps=90.0, key=51):
            hud.rows(f, x, y + 62, [r[:chars] for r in rows], size=L.T_SMALL, red=(6,))
        if t >= self.c_every and wide:
            yy = y + 62 + 8 * 26 + 30
            a_log = t - self.c_every            # the log is made when the first simulated muon is announced
            B.tag(f, "w", x, yy, "REPLY_LOG // SIMULATION", a_log, size=L.T_MICRO, pad=3, cps=90.0, key=52)
            f.text("w", x, yy + 30, B.resolve("T        E     ADC   BLOOM  GAIN", a_log, 110.0, 0.15, key=53),
                   size=L.T_MICRO, alpha=0.55)
            done = [m for m in self.demo if m[0] <= t][::-1]
            for r, (tm, e, _, _, _) in enumerate(done):
                line = f"{sd.tc(tm)[:8]} {e:.2f}  {int(e * 4095):04d}  {int(120 + 420 * e):3d}   {20 * math.log10(e):+05.1f}"
                if r == 0:                  # the reply that just came in is decoded on top of the log
                    line = B.resolve(line, t - tm, 110.0, 0.1, key=54)
                f.text("r" if r == 0 and t - tm < 1.5 else "w", x, yy + 54 + r * 22, line, size=L.T_MICRO,
                       alpha=0.95 if r == 0 else 0.65)

    def _right(self, f, t, ctx):
        """Right outer bay: the three replies side by side - the stronger the muon, the larger the bloom."""
        if self.roles["reps"] is None or t < self.c_more - 0.2:
            return
        x0, x1 = self.roles["reps"]
        x0 += 2.0
        y0 = L.HEAD_Y + 60
        ms = self.demo[1:4]
        w = (x1 - x0) / 3
        base = y0 + 330
        # the frame of the comparison is constructed on "The more powerful the muon" ...
        with f.build(t - (self.c_more - 0.2), (x0 - 8, y0 - 24, x1 + 4, base + 244), key=55, wave=0.45):
            f.tag("w", x0, y0, "MORE ENERGY = BRIGHTER = LOUDER", size=L.T_MICRO, pad=3)
            f.segments("w", [x0], [base], [x1], [base], 0.6)
            for i, (tm, e, _, _, _) in enumerate(ms):
                f.text("w", x0 + (i + 0.5) * w, base + 28, f"E {e:.2f}", size=L.T_SMALL, anchor="ms",
                       alpha=0.9 if t >= tm else 0.35)
            f.text("w", x0, base + 232, "LOUDNESS", size=L.T_MICRO, alpha=0.6)
        for i, (tm, e, _, _, _) in enumerate(ms):
            a = t - tm
            cx = x0 + (i + 0.5) * w
            if a < 0:
                continue
            grow = 1 - math.exp(-a / 0.3)
            for j in range(6):              # the bloom of this muon: the image, it grows by itself
                s = (5 + 9 * j) * (0.3 + 1.25 * e) * grow
                phi = np.linspace(0, 2 * np.pi, 60)
                x, y = towers._loop(s, j * 1.3, 12, phi, t, 950 + i)
                f.polyline("r", cx + x, base - 4 + y, (1.2 - 0.14 * j) * (0.6 + 0.4 * math.exp(-a / 1.0)), width=L.LW)
            hgt = 150 * e * grow
            bx = cx - 16
            # ... and each loudness meter is made when its muon arrives
            with f.build(a, (bx - 4, base + 40, bx + 36, base + 204), key=56 + i, wave=0.12, line=0.2, marks=False):
                f.rect("w", bx, base + 44, bx + 32, base + 200, 0.5)
                f.rects("w", bx + 3, base + 197 - hgt, bx + 29, base + 197, 0.9)

    def _bottom(self, f, t, ctx):
        panels = ctx.slots["panels"]
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        k, age = self._current(t)
        # the three panels are constructed at the cut, between the scopes of their towers (towers.scopes)
        ap = [t - T0 - lag for lag in (0.45, 0.65, 0.85)]
        if len(panels) > 0:
            x0, x1 = panels[0]
            flash = k is not None and age < 0.9
            incoming = t >= self.c_visitor
            word = "MUON" if flash else "INCOMING" if incoming else "WAITING"
            with f.build(ap[0], (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=60):
                hud.panel_header(f, x0, x1, y0, "STATE")
                if x1 - x0 >= 500 and not incoming:
                    f.text("w", x1, y0 + 44, f"T_WAIT {max(0.0, t - T0) if k is None else age:06.2f} S", size=L.T_SMALL,
                           anchor="rs", alpha=0.85)
                    f.text("w", x1, y0 + 70, "EXPECTED 1 EVERY ~5 S", size=L.T_MICRO, anchor="rs", alpha=0.6)
            # the state word is made again when a muon is caught (MUON) and when the visitor is announced
            # (INCOMING); WAITING simply resumes its blinking
            a_word = age if flash else t - self.c_visitor if incoming else 99.0
            if flash or incoming or int(t * 1.6) % 2 == 0 or t < self.c_wait:
                B.tag(f, "r" if flash or incoming else "w", x0 + 4, y0 + 84, word, min(ap[0] - 0.2, a_word), size=50, pad=8,
                      cps=55.0, key=61)
            if x1 - x0 >= 500 and incoming:
                with f.build(t - self.c_visitor, (x1 - 230, y0 + 24, x1 + 4, y0 + 78), key=62, wave=0.15, marks=False):
                    f.text("r", x1, y0 + 44, f"ETA {max(0.0, sd.T_BLOOM - t):05.2f} S", size=L.T_SMALL, anchor="rs")
                    f.text("w", x1, y0 + 70, "TARGET DET_C", size=L.T_MICRO, anchor="rs", alpha=0.7)
        if len(panels) > 1:
            x0, x1 = panels[1]
            w = (x1 - x0) / 4
            delay = self.demo[k][4] if k is not None else 0.0
            with f.build(ap[1], (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=63):
                hud.panel_header(f, x0, x1, y0, "CHAIN // ENERGY > CODE > LIGHT > SOUND")
                for s, name in enumerate(STAGES):
                    a = age - s * delay if k is not None else -1.0
                    xx = x0 + s * w
                    f.rect("w", xx + 4, y0 + 22, xx + w - 10, y0 + 70, 0.7)
                    f.text("w", xx + 6, y0 + 98, name if w >= 96 else name[:1], size=L.T_SMALL,
                           alpha=0.9 if a >= 0 else 0.45)
            for s, name in enumerate(STAGES):          # a stage fills up when the muon reaches it
                a = age - s * delay if k is not None else -1.0
                xx = x0 + s * w
                if a >= 0:
                    with f.build(a, (xx + 4, y0 + 22, xx + w - 10, y0 + 70), key=64 + s, wave=0.1, flow="lr", marks=False):
                        f.rects("r" if a < 1.3 else "w", xx + 8, y0 + 26, xx + w - 14, y0 + 66,
                                0.5 + 0.5 * math.exp(-a / 0.8))
        if len(panels) > 2:
            x0, x1 = panels[2]
            with f.build(ap[2], (x0 - 8, y0 - 24, x1 + 8, y1 + 8), key=68):
                hud.panel_header(f, x0, x1, y0, "SIMULATED")
                f.text("w", x0 + 2, y0 + 92, f"{sum(1 for m in self.demo if m[0] <= t):02d}", size=54)

    # ------------------------------------------------------------------ the visitor
    def _visitor(self, f, t, ctx):
        """One muon on its way to the centre detector while the voice names it. It arrives on 'A bloom'."""
        if t < self.c_visitor:
            return
        tw = ctx.towers["C"]
        ox, oy = tw.det
        ang = math.radians(33.0)
        x_top = ox - math.tan(ang) * (oy - L.FY0)
        u = float(np.clip((t - self.c_visitor) / (sd.T_BLOOM - self.c_visitor), 0.0, 1.0)) ** 1.6
        hx, hy = x_top + (ox - x_top) * u, L.FY0 + (oy - L.FY0) * u
        f.set_clip(L.FX0, L.FY0, L.FX1, L.FY1)
        k = np.arange(0, 1.0, 0.04)                          # dashed line of where it will go
        with f.build(t - self.c_visitor, (min(x_top, ox) - 10, L.FY0, max(x_top, ox) + 10, oy), flow="tb", wave=0.5,
                     marks=False, key=70):
            f.segments("r", x_top + (ox - x_top) * k, L.FY0 + (oy - L.FY0) * k, x_top + (ox - x_top) * (k + 0.02),
                       L.FY0 + (oy - L.FY0) * (k + 0.02), 0.45)
        f.segments("r", [x_top], [L.FY0], [hx], [hy], 1.2, width=L.LW_BOLD)
        f.dots("r", [hx], [hy], 6.0, 1.8)
        f.dots("w", [hx], [hy], 2.2, 1.2)
        f.rings("r", [hx], [hy], [14 + 10 * math.sin(t * 9.0) ** 2], 0.8, width=L.LW)
        f.set_clip()
        for tc, word, uu in ((self.c_visitor, "A VISITOR", 0.2), (self.c_messenger, "A MESSENGER", 0.48),
                             (self.c_muon, "A MUON", 0.76)):
            if t < tc:
                continue
            px, py = x_top + (ox - x_top) * uu, L.FY0 + (oy - L.FY0) * uu
            now = t < (self.c_messenger if word == "A VISITOR" else self.c_muon if word == "A MESSENGER" else 1e9)
            hud.callout(f, px, py, -70, 26 if uu < 0.3 else 34, word, red=now, alpha=1.0 if now else 0.55,
                        size=L.T_TAG, side=-1, build=t - tc)
