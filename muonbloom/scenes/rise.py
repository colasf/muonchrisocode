"""RISE - everything it was ends as light.   Scene 9, 08:32 - 10:06 (94 s).

A muon that stops in a detector gives everything away. As it slows down its energy loss rises
(Bethe-Bloch, dE/dx ~ 1/beta^2: the Bragg rise) until it stops; then it decays and the last of it
ends as scintillation light, about 10 000 photons per MeV climbing the towers to the
photomultipliers.

The three detector towers become columns of rising light (the scene draws the towers itself):
CENTRE = hero: the kicks of the music swell it, the strong ones bloom at its head, a fan of rays
opens above it; LEFT / RIGHT = a ladder of rungs flickering with the arps, lit in red by their own
detector hits. Linear, follows the sheet and the voice:
  9.0  08:32-09:11  RISE CRESCENDO   the columns fill and climb; the muon climbs the Bragg curve:
                                      speed falls, dE/dx rises, the light piles up
  9.1  09:11-09:35  VOICE            the identity card is stripped on the words - the speed, the
                                      mass, the name - and rewritten as light (mass 0, speed c);
                                      on "it ends as light" the centre column becomes a beam that
                                      goes to the top of the wall
  9.2  09:35-10:06  RISE BREAK       power down, scene 3's switching-on reversed (centre, right,
                                      left); muons keep rushing through the dark, one per kick,
                                      until one is seen: "But now, now, you know."
No score strip here: the fan and the beam need the height above the centre tower (up to the top).

Nothing has a fixed x. The towers stand in front of the wall and nobody knows yet where, how wide
or how tall: the columns of light, the ladders, the level, the fan and the beam hang on ctx.towers;
the identity card and the light / yield card take the usable columns OUTSIDE the towers (ctx.cols,
left of the first tower and right of the last one), scaled to the width they get; if only one such
column exists it gets the identity card (the voice strips it) and the yield card is dropped; the
Bragg curve / barcode / level go to ctx.slots["panels"], widest first.
"""
from __future__ import annotations

import math

import numpy as np

from .. import hud
from .. import layout as L
from .. import showdata as sd
from ..engine import CHAR_W, hash01, smoothstep
from ..show import Scene

T0, T1 = 512.0, 606.0                          # the scene on the sheet
PEAK = 551.0                                   # end of the crescendo: the muon stops
SECTIONS = [("9.0", "RISE CRESCENDO / GLOCKS", T0, PEAK), ("9.1", "VOICE OVER", PEAK, 575.0),
            ("9.2", "RISE BREAK", 575.0, T1)]
Y_LOW = 1196.0                                 # the ladders and the level stay above the bottom band

# Bethe-Bloch in plastic scintillator (PVT)
M_MU, M_E = 105.6583755, 0.51099895           # MeV
K_BB, ZA, RHO, I_EXC = 0.307075, 0.54141, 1.032, 64.7e-6
E0 = 3490.1                                   # total energy of the muon entering the tower (MeV)
YIELD = 10000.0                               # scintillation photons per MeV
E_MICHEL = 37.9                               # the decay electron it ends with (MeV)

CARD_W = 430.0                                 # a data card never gets wider than this
N_RUNG = 13
ARP = [0, 2, 4, 6, 8, 6, 4, 2, 1, 3, 5, 7, 9, 7, 5, 3]
CHORD = [0, 2, 1, 3]


def _bethe(Tk):
    g = 1.0 + Tk / M_MU
    b2 = 1.0 - 1.0 / g ** 2
    return K_BB * ZA * RHO / b2 * (np.log(2 * M_E * b2 * g * g / I_EXC) - b2), np.sqrt(b2)


def _level(t):
    """Crescendo of the rise: the columns ignite on the first beat (the music is loud at once), then
    climb to 1 over 9.0 and hold through 9.1."""
    t = np.asarray(t, np.float64)
    x = np.clip((t - T0) / (PEAK - T0), 0, 1)
    ignite = np.clip((t - T0) / 0.9, 0, 1) ** 0.5
    return np.where(t >= PEAK, 1.0, (0.24 + 0.76 * x ** 1.3) * ignite)


def _thousands(n):
    return f"{int(n):,}".replace(",", " ")


def _fit(options, width, size, pad=8.0):
    """First of the strings that fits `width` px at `size` (the last one if none does)."""
    for s in options:
        if len(s) * size * CHAR_W + pad <= width:
            return s
    return options[-1]


class Rise(Scene):
    name = "rise"
    towers = "own"
    scopes = True

    def __init__(self, ctx, seed=9):
        super().__init__(ctx)
        rng = np.random.default_rng(seed)
        self.tw = ctx.towers
        self.off = dict(sd.T_OFF)
        # voice-over cues
        self.cue_strip = sd.said("Everything it was, stripped away", 552.667)
        self.cue_speed = sd.said("The speed", 557.2)
        self.cue_mass = sd.said("The mass", 558.167)
        self.cue_name = sd.said("The name", 560.0)
        self.cue_light = sd.said("Wherever it began, it ends as light", 561.15)
        self.cue_here = sd.said("Here", 591.0, nth=0)
        if self.cue_here < T0:                                  # "Here, Right now..." of the intro matched
            self.cue_here = 591.0
        self.cue_rush = sd.said("Right now, something is rushing through you", 593.083)
        self.cue_feel = sd.said("You still cannot feel it", 599.55)
        self.know = sd.said("But now, now, you know", 602.2)
        # where the column heads arrive at the peak, and the top the beam reaches
        self.head = {k: t.top - (13.0 if k == "C" else 46.0) for k, t in self.tw.items()}
        self.top = L.FY0 + 6.0
        # range-energy table of the stopping muon
        Tk = np.geomspace(0.25, E0 - M_MU, 1500)
        dedx, beta = _bethe(Tk)
        R = np.concatenate([[0.0], np.cumsum(np.diff(Tk) / (0.5 * (dedx[1:] + dedx[:-1])))]) + 0.25 / dedx[0]
        self.tab = dict(Tk=Tk, dedx=dedx, beta=beta, lR=np.log10(R))
        self.lR0, self.lR1 = float(np.log10(R[-1])), float(np.log10(R[0]))
        self._layout(ctx)
        self.cols = {k: self._column(rng, t, int((26000 if k == "C" else 10000) * float(np.clip(t.w / 95.0, 0.6, 2.2))))
                     for k, t in self.tw.items()}
        self._build_events(rng, ctx)
        self.fan_u = rng.random(300)
        self.fan_s = np.where(rng.random(300) < 0.5, -1.0, 1.0)
        self.fan_ph = rng.uniform(0, 2 * np.pi, 300)
        self.pho = rng.random((220, 4))
        # rising motes of light between the towers: mostly near the towers, some everywhere
        nm = 1500
        m0, m1 = self.stage
        cxs = np.array([t.cx for t in self.tw.values()])
        near = rng.random(nm) < 0.62
        mx = np.where(near, cxs[rng.integers(0, len(cxs), nm)] + rng.normal(0, 190.0, nm), rng.uniform(m0, m1, nm))
        out = (mx < m0) | (mx > m1)
        self.mote_x = np.where(out, rng.uniform(m0, m1, nm), mx)
        self.mote = rng.random((nm, 4))

    # ------------------------------------------------------------------ layout
    def _layout(self, ctx):
        """Hang everything on the towers and on the usable columns outside them (see the module docstring)."""
        tws = sorted(self.tw.values(), key=lambda t: t.x0)
        hero = self.hero = self.tw["C"]
        # each side tower's ladder points towards the hero
        self.sides = [(k, t, 1.0 if t.cx < hero.cx else -1.0) for k, t in self.tw.items() if k != "C"]
        cols = [c for c in getattr(ctx, "cols", [])]
        left = [c for c in cols if c[1] <= tws[0].x0 and c[1] - c[0] >= 200.0]
        right = [c for c in cols if c[0] >= tws[-1].x1 and c[1] - c[0] >= 200.0]
        self.card = self.yld = None             # (x0, x1) of the identity card / of the light-yield card
        if left:
            a, b = left[0]
            self.card = (a, min(b, a + CARD_W))
        if right:
            a, b = right[-1]
            col = (max(a, b - CARD_W), b)
            if self.card is None:
                self.card = col                 # one outer column only: the identity card takes it
            else:
                self.yld = col
        if self.card is None:
            wide = [c for c in cols if c[1] - c[0] >= 240.0]
            if wide:
                a, b = max(wide, key=lambda c: c[1] - c[0])
                self.card = (a, min(b, a + CARD_W))
        # the stage between the cards: motes, level line
        boxes = [c for c in (self.card, self.yld) if c]
        lo = max([c[1] + 34.0 for c in boxes if c[1] <= hero.x0] + [L.FX0 + 70.0])
        hi = min([c[0] - 34.0 for c in boxes if c[0] >= hero.x1] + [L.FX1 - 70.0])
        self.stage = (lo, hi)
        prev = [t for t in tws if t.x1 <= hero.x0]
        nxt = [t for t in tws if t.x0 >= hero.x1]
        self.lvl = (prev[-1].x1 + 6.0 if prev else lo, nxt[0].x0 - 6.0 if nxt else hi)
        # room for the ladder rays of each side tower (towards the hero), and for the 'seen' tag
        # (the intensity read-out stands on the right of the hero: rays coming from that side stop before it)
        self.ray_room = {k: max(60.0, (hero.x0 - t.x1 - 104.0) if sg > 0 else (t.x0 - hero.x1 - 194.0))
                         for k, t, sg in self.sides}
        room_r = (nxt[0].x0 if nxt else hi) - hero.x1
        room_l = hero.x0 - (prev[-1].x1 if prev else lo)
        self.know_side = 1.0 if (room_r >= 430.0 or room_r >= room_l) else -1.0
        # bottom panels, widest first: Bragg curve, PMT barcode, level
        panels = sorted(ctx.slots["panels"], key=lambda q: q[0] - q[1])
        self.pan_bragg = panels[0] if panels and panels[0][1] - panels[0][0] >= 300.0 else None
        rest = panels[1:] if self.pan_bragg else panels
        self.pan_bar = next((q for q in rest if q[1] - q[0] >= 220.0), None)
        self.pan_lvl = next((q for q in rest if q is not self.pan_bar and q[1] - q[0] >= 120.0), None)

    # ------------------------------------------------------------------ build
    @staticmethod
    def _column(rng, tw, n):
        u = rng.uniform(-1.0, 1.0, n)
        spill = rng.random(n) < 0.04
        u[spill] *= rng.uniform(1.0, 1.4, spill.sum())
        return dict(cx=tw.cx, hw=tw.w / 2, bot=tw.bot, u=u.astype(np.float32),
                    ph=rng.random(n).astype(np.float32), spd=rng.uniform(0.08, 0.3, n).astype(np.float32),
                    r=(0.75 + 1.2 * rng.random(n) ** 3).astype(np.float32),
                    b=rng.uniform(0.45, 1.0, n).astype(np.float32))

    def _build_events(self, rng, ctx):
        cues, det = ctx.cues, ctx.det
        # bass: the strong kicks of the music bloom at the head of the centre tower
        kt, ka = cues.kicks(T0, self.off["C"] - 0.8)
        sel, last = [], -9.0
        for tk, ak in zip(kt, ka):
            if ak >= 0.9 and tk - last >= 0.95:
                sel.append((float(tk), float(min(ak, 1.5))))
                last = tk
        if not sel or sel[-1][0] < PEAK - 0.5:
            sel.append((PEAK, 1.4))
        self.bass = np.array([s[0] for s in sel])
        self.bass_a = np.array([s[1] for s in sel])
        self.bass_ang = rng.uniform(-math.pi, 0.0, (len(self.bass), 64))
        self.bass_len = rng.uniform(0.25, 1.0, (len(self.bass), 64))
        # arps: every onset of the music is a note on the ladders of the side towers
        ot, oa = cues.onset_t, cues.onset_a
        m = (ot >= T0) & (ot < self.off["R"])
        notes = []
        for k, (tn, an) in enumerate(zip(ot[m], oa[m])):
            bar, step = divmod(k, 16)
            idx = int(np.clip(CHORD[bar % 4] + ARP[step], 0, N_RUNG - 1))
            lv = float(_level(tn))
            vel = float(np.clip(an, 0.35, 1.2)) * (0.45 + 0.55 * lv) * (0.55 if tn >= PEAK else 1.0)
            notes.append((float(tn), idx, vel))
        self.notes = np.array(notes, np.float64).reshape(-1, 3)
        # the real hits of the detectors while they are on
        hits = []
        for key in sd.KEYS:
            tt, ee, ec = det.hits(key, T0 - 1.0, self.off[key])
            for th, e, echo in zip(tt, ee, ec):
                hits.append((float(th), key, float(e), bool(echo), float(rng.uniform(-0.38, 0.38)),
                             float(rng.normal(0, 0.09))))
        hits.sort()
        self.hits = hits
        # 9.2: muons still rushing through the dark, one per kick, filtered; one bright one when "you know"
        kt, ka = cues.kicks(self.off["L"] + 1.0, T1 - 0.4)
        rain = [(float(tk), float(60.0 + hash01(k, 31) * (L.W - 120.0)), float((hash01(k, 32) - 0.5) * 0.22),
                 float(0.22 + 0.16 * min(ak, 1.3))) for k, (tk, ak) in enumerate(zip(kt, ka))
                if abs(tk - self.know) > 0.7]
        rain.append((self.know, self.hero.x0 + 0.32 * self.hero.w, 0.0, 1.0))
        self.rain = rain
        soft = []
        for key in sd.KEYS:                                        # hits arriving while the tower is off: filtered
            tt, ee, ec = det.hits(key, self.off[key] + 0.5, T1)
            soft += [(float(th), key, float(e)) for th, e, echo in zip(tt, ee, ec) if not echo]
        self.soft = soft

    # ------------------------------------------------------------------ state
    def _power(self, key, t):
        return 1.0 - smoothstep(self.off[key], self.off[key] + 0.55, t)

    def _muon(self, t):
        """The stopping muon at time t: kinetic energy, beta, gamma, dE/dx, residual range."""
        tb = self.tab
        if t >= PEAK:
            return dict(Tk=0.0, beta=0.0, gam=1.0, dedx=float(tb["dedx"][0]), R=0.0, lR=self.lR1)
        lr = self.lR0 + (self.lR1 - self.lR0) * (max(t - T0, 0.0) / (PEAK - T0)) ** 0.9
        Tk = float(np.interp(lr, tb["lR"], tb["Tk"]))
        return dict(Tk=Tk, beta=float(np.interp(lr, tb["lR"], tb["beta"])), gam=1.0 + Tk / M_MU,
                    dedx=float(np.interp(lr, tb["lR"], tb["dedx"])), R=10 ** lr, lR=lr)

    def _state(self, t, ctx):
        lv = float(_level(t))
        pw = {k: float(self._power(k, t)) for k in self.tw}
        beam = float(smoothstep(self.cue_light, self.cue_light + 0.4, t))
        yh = {}
        for k, tw in self.tw.items():
            y = tw.bot - (tw.bot - self.head[k]) * lv
            if k == "C":
                y -= (self.head["C"] - self.top) * beam
            yh[k] = tw.bot - (tw.bot - y) * pw[k] ** 0.7
        mu = self._muon(t)
        e_dep = (E0 - M_MU) - mu["Tk"] + E_MICHEL * float(smoothstep(self.cue_light, self.cue_light + 1.2, t))
        pulse = 0.8 * ctx.cues.kick(t, tau=0.24) if t < self.off["C"] + 0.3 else 0.0
        return dict(lv=lv, pw=pw, yh=yh, mu=mu, n_gamma=YIELD * e_dep, e_dep=e_dep, pulse=pulse,
                    beam=beam * pw["C"])

    def _active_notes(self, t, span=0.6):
        n = self.notes
        if not len(n):
            return n
        a, b = np.searchsorted(n[:, 0], t - span, side="right"), np.searchsorted(n[:, 0], t, side="right")
        return n[a:b]

    def _rung_y(self, idx, yh):
        return Y_LOW - (idx + 1) / (N_RUNG + 1) * (Y_LOW - yh)

    # ----------------------------------------------------------------- render
    def draw(self, f, t, ctx):
        t = float(np.clip(t, T0, T1 - 1e-6))
        st = self._state(t, ctx)
        view = (L.FX0 + 2, L.FY0 + 2, L.FX1 - 2, Y_LOW)
        f.set_clip(*view)
        self._draw_motes(f, t, st)
        self._draw_fan(f, t, st)
        f.set_clip(self.stage[0], L.FY0 + 2, self.stage[1], Y_LOW)
        self._draw_bass(f, t, st, ctx)
        self._draw_photons(f, t, st)
        f.set_clip(L.FX0 + 2, L.FY0 + 2, L.FX1 - 2, L.FY1 - 2)
        self._draw_towers(f, t, st)
        self._draw_arps(f, t, st)
        self._draw_muons(f, t, st)
        self._draw_rain(f, t, ctx)
        f.set_clip()
        self._draw_level(f, t, st)
        self._draw_header(f, t, st)
        self._draw_card(f, t, st)
        self._draw_light(f, t, st)
        self._draw_bottom(f, t, st, ctx)
        invert = PEAK <= t < PEAK + 0.06 or self.cue_light <= t < self.cue_light + 0.06
        return {"invert": invert}

    # --- the towers --------------------------------------------------------------
    def _tower_frame(self, f, tw, inten, layer="w"):
        """Outline of a tower with its scintillator slabs and the detector module at its head."""
        if inten <= 0.004:
            return
        f.rect(layer, tw.x0, tw.top, tw.x1, tw.bot, inten, width=L.LW_BOLD)
        ys = np.arange(tw.top + 48.0, tw.bot - 1, 48.0)
        f.segments(layer, np.full_like(ys, tw.x0), ys, np.full_like(ys, tw.x0 + 8), ys, 0.8 * inten)
        f.segments(layer, np.full_like(ys, tw.x1 - 8), ys, np.full_like(ys, tw.x1), ys, 0.8 * inten)
        f.segments(layer, [tw.x0], [tw.top + tw.det_h], [tw.x1], [tw.top + tw.det_h], 0.7 * inten)

    def _draw_column(self, f, c, yh, t, gain, bands=(), flare=0.0):
        H = c["bot"] - yh
        if H < 3 or gain < 0.004:
            return
        q = ((c["ph"] + c["spd"] * t) % 1.0) ** 0.62
        d = (1.0 - q) * H
        cap = c["hw"] * 1.05
        shape = np.sqrt(np.clip(1.0 - (1.0 - np.minimum(d / cap, 1.0)) ** 2, 0.0, 1.0))
        x = c["cx"] + c["u"] * c["hw"] * shape * (1.0 + flare * np.exp(-d / 55.0))
        y = yh + d
        inten = c["b"] * gain * (0.16 + 0.84 * q ** 2.4)
        for yk, amp in bands:
            inten = inten * (1.0 + amp * np.exp(-((y - yk) / 9.0) ** 2))
        f.dots("w", x, y, c["r"], inten)

    def _draw_towers(self, f, t, st):
        lv, pw, yh = st["lv"], st["pw"], st["yh"]
        notes = self._active_notes(t)
        for k in ("L", "R", "C"):
            tw = self.tw[k]
            self._tower_frame(f, tw, 0.22 + 0.6 * pw[k])
            if k == "C":
                gain = (0.55 + 0.45 * lv) * (1.0 + 0.9 * math.tanh(st["pulse"])) * pw[k] * (1.0 + 0.8 * st["beam"])
                self._draw_column(f, self.cols[k], yh[k], t, gain, flare=0.7 * lv * (1 - st["beam"]))
                if st["beam"] > 0.01:     # it ends as light: a solid core burns up the column
                    w = 3.0 + 14.0 * st["beam"]
                    f.rects("w", tw.cx - w, yh[k], tw.cx + w, tw.bot, 0.8 * st["beam"])
            else:
                bands = [(self._rung_y(int(i), yh[k]), 2.6 * v * math.exp(-(t - tn) / 0.24)) for tn, i, v in notes]
                self._draw_column(f, self.cols[k], yh[k], t, (0.5 + 0.5 * lv) * pw[k], bands)
            # the head: the muon (red) sits on the light; once it has become light it is white
            if yh[k] < tw.bot - 4 and pw[k] > 0.02:
                if k == "C" and t >= self.cue_light:
                    f.dots("w", [tw.cx], [yh[k] - 3], 7.5, 1.8 * pw[k])
                else:
                    f.dots("r", [tw.cx], [yh[k] - 3], 7.0 if k == "C" else 5.0, 1.8 * pw[k])
                    f.dots("w", [tw.cx], [yh[k] - 3], 2.2, 1.0 * pw[k])
        # "now you know": the three towers answer once
        a = t - self.know
        if 0 <= a < 3.8:
            e = math.exp(-a / 0.9)
            for tw in self.tw.values():
                f.rect("w", tw.x0, tw.top, tw.x1, tw.bot, 1.4 * e, width=L.LW_FRAME)
                f.rects("w", tw.x0 + 3, tw.top + 3, tw.x1 - 3, tw.bot - 3, 0.12 * e)

    def _draw_arps(self, f, t, st):
        yh, pw = st["yh"], st["pw"]
        notes = self._active_notes(t, 0.7)
        for k, tw, sgn in self.sides:
            if pw[k] < 0.01 or yh[k] > Y_LOW - 90:
                continue
            edge = tw.x1 if sgn > 0 else tw.x0
            anchor = "ls" if sgn > 0 else "rs"
            room = self.ray_room[k]
            ys = np.array([self._rung_y(i, yh[k]) for i in range(N_RUNG)])
            f.segments("w", np.full(N_RUNG, edge), ys, np.full(N_RUNG, edge + sgn * 14), ys, 0.6 * pw[k], width=L.LW)
            for i in range(0, N_RUNG, 2):
                f.text("w", edge + sgn * 30, float(ys[i]) + 5, f"{i + 1:02d}", size=L.T_MICRO, alpha=0.4 * pw[k],
                       anchor=anchor)
            for j, (tn, idx, vel) in enumerate(notes):      # the arps of the music, mirrored on both ladders
                a = t - tn
                y = ys[int(idx)]
                e = math.exp(-a / 0.24) * pw[k]
                Ln = min((80 + 330 * vel) * (1 - math.exp(-a / 0.03)), room)
                xe = edge + sgn * (64 + Ln)
                f.segments("w", [edge + sgn * 64], [y], [xe], [y], 1.7 * vel * e, 0.5 * vel * e, width=L.LW)
                f.dots("w", [edge + sgn * 3], [y], 3.6, 1.8 * vel * e)
                f.crosses("w", [xe], [y], 6.0, 1.1 * e, width=L.LW)
            # the hits of this tower's own detector: red, with their energy
            for (th, key, e_h, echo, _, _) in self.hits:
                a = t - th
                if key != k or not (0 <= a < 1.6):
                    continue
                idx = int(round(e_h * (N_RUNG - 1)))
                y = ys[idx]
                e = math.exp(-a / 0.5) * pw[k]
                Ln = min((120 + 420 * e_h) * (1 - math.exp(-a / 0.05)) * (0.55 if echo else 1.0), room)
                xe = edge + sgn * (64 + Ln)
                f.segments("r", [edge + sgn * 64], [y], [xe], [y], 1.5 * e, 0.5 * e, width=L.LW_BOLD)
                f.dots("r", [edge + sgn * 3], [y], 4.4, 1.8 * e)
                f.rings("r", [xe], [y], [7.0 + 22.0 * (1 - e)], 0.9 * e)
                if not echo:
                    f.tag("r", edge + sgn * 66, y - 12, f"{L.NAMES[k]}  E {e_h:.3f}", size=L.T_MICRO, pad=3,
                          alpha=min(1.0, 2.5 * e), anchor=anchor)

    def _draw_muons(self, f, t, st):
        """Every real hit: a muon enters the tower from the top of the wall and is caught."""
        y0 = L.FY0 + 3
        for (th, key, e_h, echo, off, ang) in self.hits:
            if echo:
                continue
            tw = self.tw[key]
            dur = (tw.top - y0) / 2600.0
            a = t - (th - dur)
            if not (0 <= a < dur + 1.2):
                continue
            x = tw.cx + off * tw.w
            u = min(1.0, a / dur)
            xs = x - math.tan(ang) * (tw.top - y0)
            xh, yhh = xs + (x - xs) * u, y0 + (tw.top - y0) * u
            fade = 1.0 if a < dur else math.exp(-(a - dur) / 0.35)
            f.segments("r", [xs], [y0], [xh], [yhh], (0.5 + 0.5 * e_h) * fade * st["pw"][key], width=L.LW)
            if a < dur:
                f.dots("r", [xh], [yhh], 3.0, 1.6)
            else:
                e = math.exp(-(a - dur) / 0.16)
                f.dots("w", [x], [tw.top], 3.4, 1.6 * e)
                f.rings("r", [x], [tw.top], [8 + 70 * e_h * (1 - e)], 0.9 * e, width=L.LW)

    def _draw_rain(self, f, t, ctx):
        y0, y1 = L.FY0 + 3, L.FY1 - 3
        # the flux nobody sees: a faint drizzle over the whole wall once the detectors are dark
        dark = float(smoothstep(self.off["L"] + 0.8, self.off["L"] + 3.2, t))
        if dark > 0.01:
            k = np.arange(120)
            P = 1.3 + 2.3 * hash01(k, 41)
            tt = t + hash01(k, 42) * P
            cyc = np.floor(tt / P)
            a = tt - cyc * P
            dur = 0.2 + 0.2 * hash01(k, 43)
            x = 60.0 + hash01(k, cyc, 44) * (L.W - 120.0)
            ang = (hash01(k, cyc, 45) - 0.5) * 0.2
            u = np.clip(a / dur, 0.0, 1.0)
            fade = np.where(a < dur, 1.0, np.exp(-(a - dur) / 0.45))
            m = fade > 0.03
            xe = x + np.tan(ang) * (y1 - y0)
            inten = dark * (0.16 + 0.24 * hash01(k, 46)) * fade
            f.segments("r", x[m], np.full(int(m.sum()), y0), (x + (xe - x) * u)[m], (y0 + (y1 - y0) * u)[m],
                       inten[m] * 0.4, inten[m], width=1.2)
            hm = m & (a < dur)
            f.dots("r", (x + (xe - x) * u)[hm], (y0 + (y1 - y0) * u)[hm], 2.0, 3.0 * inten[hm])
        for (tm, x, ang, amp) in self.rain:
            a = t - tm
            if not (0 <= a < (6.0 if amp >= 1.0 else 1.3)):
                continue
            dur = (y1 - y0) / 5200.0
            u = min(1.0, a / dur)
            xe = x + math.tan(ang) * (y1 - y0)
            fade = 1.0 if a < dur else math.exp(-(a - dur) / (0.35 if amp < 1 else 0.7))
            f.segments("r", [x], [y0], [x + (xe - x) * u], [y0 + (y1 - y0) * u], (0.5 + 0.6 * amp) * amp * fade * 2.6
                       if amp < 1 else 0.5 + 1.1 * fade, width=1.0 + 0.9 * amp if amp < 1 else 3.0)
            if a < dur:
                f.dots("r", [x + (xe - x) * u], [y0 + (y1 - y0) * u], 2.2 + 2.2 * amp, 1.5 * max(amp, 0.5))
            if amp >= 1.0:                           # where it crosses the centre detector: rings
                twc = self.tw["C"]
                for j in range(3):
                    aj = a - 0.05 - 0.22 * j
                    if 0 <= aj < 2.2:
                        uj = aj / 2.2
                        f.rings("r", [x], [twc.top + 0.5 * twc.det_h], [14 + 330 * (1 - (1 - uj) ** 3)],
                                1.3 * (1 - uj) ** 1.6, width=L.LW_BOLD)
                f.dots("w", [x], [twc.top + 0.5 * twc.det_h], 5.0, 1.8 * math.exp(-a / 0.5))
            if amp >= 1.0 and a > 0.1:
                al = 1.0 - float(smoothstep(4.0, 6.0, a))
                tw = self.hero
                y = tw.top + 0.5 * tw.det_h + 24.0
                xx = x + (xe - x) * (y - y0) / (y1 - y0)
                sg = self.know_side
                ex = (tw.x1 + 60.0) if sg > 0 else (tw.x0 - 60.0)
                anc = "ls" if sg > 0 else "rs"
                f.segments("w", [xx, ex - sg * 34], [y, y - 60], [ex - sg * 34, ex], [y - 60, y - 60], 0.8 * al,
                           width=L.LW)
                f.dots("w", [xx], [y], 3.0, 1.4 * al)
                f.tag("r", ex + sg * 8, y - 46, "MU-", size=44, pad=8, alpha=al, bold=True, anchor=anc)
                rate = int(round(getattr(ctx, "RATE_YOU", 63.0)))
                f.text("w", ex + sg * 8, y - 6, hud.erode(f"ONE OF {rate} /S THROUGH YOU", 1 - min(1.0, (a - 0.1) * 3), 7),
                       size=L.T_LABEL, alpha=0.95 * al, anchor=anc)
                f.text("w", ex + sg * 8, y + 22, hud.typed(f"SEEN BY DET_C AT {sd.tc(self.know)}", a, cps=60, delay=0.6),
                       size=L.T_MICRO, alpha=0.7 * al, anchor=anc)
        for (tm, key, e_h) in self.soft:           # the filtered hits of the break
            a = t - tm
            if 0 <= a < 2.5:
                tw = self.tw[key]
                u = a / 2.5
                f.rings("w", [tw.cx], [tw.top], [20 + 200 * (1 - (1 - u) ** 2)], 0.3 * (1 - u) ** 1.5, width=2.0)

    # --- the centre: fan of rays, bass blooms, photons, level ----------------------------
    def _draw_fan(self, f, t, st):
        yh = st["yh"]["C"]
        twc = self.tw["C"]
        y_hi, y_lo = twc.top + 47.0, twc.top + 0.62 * (twc.bot - twc.top)
        amt = float(smoothstep(y_lo, y_hi, yh)) * st["lv"] * st["pw"]["C"] * (1.0 - st["beam"])
        if amt < 0.01:
            return
        cx = twc.cx
        n = int(30 + 250 * amt)
        u, s = self.fan_u[:n], self.fan_s[:n]
        ext = np.where(s < 0, cx - (L.FX0 + 8.0), (L.FX1 - 8.0) - cx)
        xe = cx + s * (u ** 1.3) * ext
        xs = cx + s * 46.0 * (0.25 + 0.75 * u)
        tw_ = 0.6 + 0.4 * np.sin(self.fan_ph[:n] + t * (2.0 + 7.0 * u))
        i0 = amt * (0.09 + 0.36 * (1 - u) ** 2) * tw_ * (1.0 + 0.6 * math.tanh(st["pulse"]))
        f.segments("w", xs, np.full(n, yh + 4.0), xe, np.full(n, L.FY0 + 3.0), i0, i0 * 0.3)

    def _draw_bass(self, f, t, st, ctx):
        twc = self.tw["C"]
        cx, yh, pw = twc.cx, st["yh"]["C"], st["pw"]["C"]
        if pw < 0.01:
            return
        a = t - self.bass
        for j in np.nonzero((a >= 0) & (a < 1.3))[0]:
            aj = float(a[j])
            u = aj / 1.3
            e = (1 - u) ** 2 * pw * min(1.0, self.bass_a[j]) * (0.4 + 0.6 * st["lv"])
            f.rings("w", [cx], [yh], [30 + 780 * (1 - (1 - u) ** 3)], 0.8 * e, width=L.LW)
            f.rings("w", [cx], [yh], [18 + 380 * (1 - (1 - u) ** 3)], 0.5 * e)
            grow = 1 - (1 - min(1.0, aj / 0.3)) ** 3
            ang = self.bass_ang[j]
            r0 = 24 + 90 * u
            r1 = r0 + self.bass_len[j] * 440 * grow
            f.segments("w", cx + np.cos(ang) * r0, yh + np.sin(ang) * r0, cx + np.cos(ang) * r1,
                       yh + np.sin(ang) * r1, 0.8 * e, 0.0)
            f.dots("w", cx + np.cos(ang) * r1, yh + np.sin(ang) * r1, 2.0, 1.3 * e)
        for (th, key, e_h, echo, _, _) in self.hits:        # a real hit of the centre detector: red
            aj = t - th
            if key != "C" or echo or not (0 <= aj < 1.6):
                continue
            u = aj / 1.6
            e = (1 - u) ** 2 * pw * (0.4 + 0.6 * e_h)
            f.rings("r", [cx], [yh], [18 + (160 + 420 * e_h) * (1 - (1 - u) ** 3)], 1.1 * e, width=L.LW_BOLD)
            f.rings("r", [cx], [yh], [10 + (90 + 240 * e_h) * (1 - (1 - u) ** 3)], 0.7 * e, width=L.LW)

    def _draw_motes(self, f, t, st):
        """The light piles up: short streaks climbing the wall between the towers, denser as it rises."""
        amt = st["lv"] * st["pw"]["C"]
        if amt < 0.02:
            return
        n = int(90 + 1400 * amt ** 1.5)
        p = self.mote[:n]
        x = self.mote_x[:n]
        H = Y_LOW - (L.FY0 + 3.0)
        ph = (p[:, 0] + t * (0.05 + 0.2 * p[:, 1]) * (1.0 + 0.8 * st["beam"])) % 1.0
        y = Y_LOW - ph * H
        ln = (10.0 + 40.0 * p[:, 2]) * (0.5 + ph)
        twk = 0.55 + 0.45 * np.sin(p[:, 3] * 6.283 + t * (3.0 + 5.0 * p[:, 1]))
        inten = amt * (0.12 + 0.36 * p[:, 1]) * twk * np.sin(np.pi * ph) ** 0.5 * (1.0 + 0.5 * math.tanh(st["pulse"]))
        inten = inten * (1.0 - 0.6 * st["beam"])
        f.segments("w", x, y, x, y + ln, inten, inten * 0.12, width=1.2)

    def _draw_photons(self, f, t, st):
        """Once it is light: photons leave the beam, sideways and up."""
        b = st["beam"]
        if b < 0.02:
            return
        twc = self.tw["C"]
        p = self.pho
        ph = (p[:, 0] + t * (0.25 + 0.5 * p[:, 1])) % 1.0
        sgn = np.where(p[:, 2] < 0.5, -1.0, 1.0)
        y0 = self.top + p[:, 3] * (Y_LOW - self.top)
        dx = 26.0 + 900.0 * ph ** 0.85
        x = twc.cx + sgn * dx
        y = y0 - 0.22 * dx
        ln = 26.0 + 60.0 * (1 - ph)
        inten = b * 0.75 * (1 - ph) ** 1.6
        f.segments("w", x, y, x + sgn * ln, y - 0.22 * ln, inten, inten * 0.2)

    def _draw_level(self, f, t, st):
        pw = st["pw"]["C"] * (1.0 - st["beam"])
        yh = st["yh"]["C"]
        twc = self.tw["C"]
        if pw < 0.01 or yh > twc.bot - 40:
            return
        mu = st["mu"]
        xl, xr = self.lvl
        y = max(yh, L.HEAD_Y + 30)
        if y > Y_LOW:
            return
        f.segments("r", [xl], [y], [xr], [y], 0.85 * pw, width=L.LW)
        f.tag("r", xl + 40, y - 10, f"LEVEL {st['lv']:.3f}", size=L.T_MICRO, pad=3, alpha=pw)
        f.tag("r", xr - 40, y - 10, f"DE/DX {mu['dedx']:07.3f} MEV/CM", size=L.T_MICRO, pad=3, alpha=pw, anchor="rs")
        # intensity read off the column
        x = twc.x1 + 30
        f.text("w", x, y + 28, "INTENSITY: MAX", size=L.T_MICRO, alpha=0.8 * pw)
        ys = np.arange(y + 54, min(twc.bot - 6, Y_LOW - 8), 24.0)
        q = 1.0 - (ys - y) / max(twc.bot - y, 1.0)
        v = np.clip(st["lv"] * (0.12 + 0.88 * q ** 2.4) * (1 + 0.5 * math.tanh(st["pulse"]))
                    + 0.04 * hash01(np.arange(len(ys)), int(t * 12)), 0, 0.99)
        for yy, vv in zip(ys, v):
            f.text("w", x, float(yy), f"{vv:.2f}", size=L.T_MICRO, alpha=0.7 * pw)

    # --- HUD ------------------------------------------------------------------------
    def _draw_header(self, f, t, st):
        k = 0 if t < SECTIONS[1][2] else 1 if t < SECTIONS[2][2] else 2
        code, name, s0, s1 = SECTIONS[k]
        if self.card:
            x0, x1 = self.card
            w = x1 - x0
            f.tag("r" if k < 2 else "w", x0, 266, _fit([f"{code} // {name}", f"{code} // RISE", code], w, L.T_LABEL),
                  size=L.T_LABEL, pad=4)
            if t >= self.off["L"] + 0.6:        # the dark: what the voice says, typed where the card was
                al = float(smoothstep(self.off["L"] + 0.6, self.off["L"] + 2.0, t))
                size = L.T_SMALL if w >= 300 else L.T_MICRO
                if 49 * size * CHAR_W <= w:
                    head = ["DETECTORS OFFLINE // MU FLUX UNCHANGED 1 /CM2/MIN"]
                else:
                    head = ["DETECTORS OFFLINE", _fit(["MU FLUX UNCHANGED 1 /CM2/MIN", "MU FLUX UNCHANGED"], w, size, 0)]
                for j, ln in enumerate(head):
                    f.text("w", x0, 306 + j * 24, ln, size=size, alpha=0.85 * al)
                y = 306 + len(head) * 24 + 22
                lines = [(self.cue_here, "HERE"), (self.cue_rush, "RUSHING THROUGH YOU"),
                         (self.cue_feel, "YOU STILL CANNOT FEEL IT"), (self.know, "NOW YOU KNOW")]
                for j, (tv, word) in enumerate(lines):
                    if t >= tv:
                        f.tag("r" if j == 3 else "w", x0, y + j * 34, hud.typed(word, t - tv, cps=40), size=size,
                              pad=4, alpha=al)
        if self.yld:
            x1 = self.yld[1]
            f.text("w", x1, 266, _fit(["RISE // SCENE 09 // DE/DX -> LIGHT", "RISE // SCENE 09"], self.yld[1] - self.yld[0],
                                     L.T_MICRO), size=L.T_MICRO, alpha=0.85, anchor="rs")
            f.text("w", x1, 290, sd.tc(t), size=L.T_MICRO, alpha=0.6, anchor="rs")

    def _draw_card(self, f, t, st):
        """Identity of the muon, stripped on the voice-over, rewritten as light."""
        pw = st["pw"]["L"]
        if pw < 0.01 or self.card is None:
            return
        fr = int(t * 30)
        gone = 1.0 - pw
        mu = st["mu"]
        x, x1 = self.card
        w = x1 - x
        small = w < 330.0
        rs = L.T_MICRO if small else L.T_SMALL          # row type size
        vx = x + (92.0 if small else 124.0)
        y0 = 326.0
        light = t >= self.cue_light
        f.tag("r" if light else "w", x, y0, hud.erode("LIGHT // IDENTITY" if light else "MU- // IDENTITY", gone, 1, fr),
              size=L.T_LABEL, pad=4, alpha=pw)
        # the big figure: its speed
        if t < self.cue_speed:
            big = f"{mu['beta']:.6f}"
        elif not light:
            big = hud.erode(f"{0.0:.6f}", (t - self.cue_speed) / 0.9, 3, fr)
        else:
            big = f"{1.0:.6f}"[: int((t - self.cue_light) * 30)]
        bsz = float(np.clip((w - 6) / (8 * CHAR_W), 36.0, 76.0))
        hud.big_number(f, x, 462, _fit(["SPEED // BETA = V/C", "SPEED"], w, L.T_MICRO), hud.erode(big, gone, 4, fr),
                       size=bsz, layer="r" if light else "w", alpha=pw)
        cs = self.cue_strip
        room = int((x1 - vx) / (rs * CHAR_W))            # characters a value may take
        origin = "15.21 KM // PI- DECAY" if room >= 21 else "15.21 KM"
        rows = [("NAME", "MUON  MU-", self.cue_name, "LIGHT"), ("CLASS", "LEPTON // GEN 2", cs, None),
                ("CHARGE", "-1 E", cs + 0.5, None),
                ("MASS", "105.6583755 MEV/C2", self.cue_mass, "0.0000000 MEV/C2"),
                ("SPEED", f"{mu['beta']:.6f} C", self.cue_speed, "1.000000 C"),
                ("GAMMA", f"{mu['gam']:.4f}", cs + 1.0, None),
                ("ENERGY", f"{(mu['Tk'] + M_MU) / 1000:.5f} GEV", cs + 1.5, None),
                ("LIFETIME", "2.1969811 US", cs + 2.0, None),
                ("ORIGIN", origin, cs + 2.5, None)]
        yy = 520.0
        for k, (lab, val, cue, after) in enumerate(rows):
            y = yy + k * 30
            a = t - cue
            lab_al = 0.85 if a < 0.5 else 0.3
            if light and after:
                lab_al = 0.95
                n = int(max(0.0, t - self.cue_light - 0.25 * k) * 34)
                val_s, lay = after[:n], "r" if lab == "NAME" else "w"
            elif a < 0:
                val_s, lay = val, "w"
            else:
                val_s, lay = hud.erode(val, (a - 0.15) / 0.9, 11 + k, fr), "w"
                if a < 1.4:     # a red cut runs through the value first
                    wv = len(val) * rs * CHAR_W * min(1.0, a / 0.3)
                    f.segments("r", [vx - 4], [y - 6], [vx - 4 + wv], [y - 6], 1.2 * (1 - a / 1.4) * pw, width=L.LW)
            f.text("w", x, y, hud.erode(lab, gone, 21 + k, fr), size=rs, alpha=lab_al * pw)
            f.text(lay, vx, y, hud.erode(val_s, gone, 31 + k, fr), size=rs, alpha=0.95 * pw)
        # hit stream: what the three detectors really caught
        y = 826.0
        hud.panel_header(f, x, x1, y - 26, hud.erode(_fit(["HIT_STREAM // DET_L DET_C DET_R", "HIT_STREAM"], w, L.T_MICRO),
                                                    gone, 41, fr), alpha=pw)
        n_max = int(w / (L.T_MICRO * CHAR_W))
        k = 0
        for (th, key, e_h, echo, _, _) in reversed(self.hits):
            if th > t:
                continue
            if k >= 15:
                break
            if n_max >= 29:
                line = f"{sd.tc(th)} {L.NAMES[key]} E {e_h:.3f} {'ECHO' if echo else 'HIT'}"
            elif n_max >= 21:
                line = f"{sd.tc(th)} {L.NAMES[key]} {e_h:.3f}"
            else:
                line = f"{sd.tc(th)[:8]} {key} {e_h:.2f}"
            f.text("r" if (k == 0 and t - th < 1.5) else "w", x, y + 30 + k * 21, hud.erode(line, gone, 50 + k, fr),
                   size=L.T_MICRO, alpha=(0.95 if k < 2 else 0.6) * pw)
            k += 1

    def _draw_light(self, f, t, st):
        """The light it becomes (second outer column, when there is one)."""
        pw = st["pw"]["R"]
        if pw < 0.01 or self.yld is None:
            return
        fr = int(t * 30)
        gone = 1.0 - pw
        mu = st["mu"]
        x0, x1 = self.yld
        w = x1 - x0
        rs = L.T_MICRO if w < 330.0 else L.T_SMALL
        f.tag("w", x0, 326, hud.erode("LIGHT // YIELD", gone, 61, fr), size=L.T_LABEL, pad=4, alpha=pw)
        bsz = float(np.clip((w - 6) / (10 * CHAR_W), 36.0, 56.0))
        hud.big_number(f, x0, 462, _fit(["SCINTILLATION PHOTONS", "PHOTONS"], w, L.T_MICRO),
                       hud.erode(_thousands(st["n_gamma"]), gone, 62, fr), size=bsz, layer="w", alpha=pw)
        v = {k: min(1.0, (0.2 + 0.8 * st["lv"]) * st["pw"][k]) for k in self.tw}
        rows = ["YIELD     10 000 /MEV", f"E_DEP     {st['e_dep']:09.3f} MEV",
                f"DE/DX     {mu['dedx']:07.3f} MEV/CM", f"RANGE     {mu['R']:09.4f} CM",
                f"PMT_L     {v['L'] * 0.71:.3f} V", f"PMT_C     {v['C'] * (0.8 + 0.2 * math.tanh(st['pulse'])):.3f} V",
                f"PMT_R     {v['R'] * 0.71:.3f} V", f"KICK      {st['pulse']:.3f}"]
        for k, r in enumerate(rows):
            f.text("w", x0, 520 + k * 30, hud.erode(r, gone, 70 + k, fr), size=rs, alpha=0.9 * pw)
        y = 826.0
        hud.panel_header(f, x0, x1, y - 26, hud.erode(_fit(["ARP_STREAM // L+R LADDERS", "ARP_STREAM"], w, L.T_MICRO),
                                                     gone, 81, fr), alpha=pw)
        if len(self.notes):
            n = self.notes[self.notes[:, 0] <= t][-15:][::-1]
            for k, (tn, idx, vel) in enumerate(n):
                line = f"{sd.tc(tn)} RUNG {int(idx) + 1:02d} VEL {vel:.2f}"
                f.text("r" if k == 0 else "w", x0, y + 30 + k * 21, hud.erode(line, gone, 90 + k, fr),
                       size=L.T_MICRO, alpha=(0.95 if k < 2 else 0.6) * pw)

    def _draw_bottom(self, f, t, st, ctx):
        y0, y1 = ctx.slots["y0"], ctx.slots["y1"]
        pwc = float(st["pw"]["C"])
        fr = int(t * 30)
        if self.pan_bragg and pwc > 0.01:
            # Bragg rise: dE/dx against residual range, log-log, the stopping point on the right
            g0, g1 = self.pan_bragg
            p = pwc
            hud.panel_header(f, g0, g1, y0, hud.erode(_fit(["BRAGG // DE/DX VS RESIDUAL RANGE", "BRAGG // DE/DX"],
                                                           g1 - g0, L.T_MICRO), 1 - p, 120, fr), alpha=p)
            px0, px1, py0, py1 = g0 + 16, g1 - 168, y0 + 26, y1 - 22
            tb = self.tab
            d0, d1 = math.log10(1.5), math.log10(float(tb["dedx"][0]) * 1.1)

            def PX(lr):
                return px0 + (self.lR0 - np.asarray(lr)) / (self.lR0 - self.lR1) * (px1 - px0)

            def PY(d):
                return py1 - (np.log10(d) - d0) / (d1 - d0) * (py1 - py0)

            f.segments("w", [px0, px0], [py0, py1], [px0, px1], [py1, py1], 0.6 * p)
            decs = list(range(math.ceil(self.lR1), math.floor(self.lR0) + 1))
            every = 1 if (px1 - px0) / max(len(decs), 1) >= 46 else 2
            for j, dec in enumerate(decs):
                xx = float(PX(dec))
                f.segments("w", [xx], [py1], [xx], [py1 + 6], 0.7 * p)
                if j % every == 0:
                    f.text("w", xx + 3, py1 + 20, f"1E{dec:+d}", size=L.T_MICRO, alpha=0.6 * p)
            sel = slice(None, None, 8)
            f.dots("w", PX(tb["lR"][sel]), PY(tb["dedx"][sel]), 1.0, 0.35 * p)
            mu = st["mu"]
            m = tb["lR"] >= mu["lR"]
            if m.sum() > 1:
                f.polyline("w", PX(tb["lR"][m][::-1]), PY(tb["dedx"][m][::-1]), 1.0 * p, width=L.LW)
            mx, my = float(PX(mu["lR"])), float(PY(mu["dedx"]))
            f.segments("r", [mx, px0], [py0 - 4, my], [mx, px1], [py1, my], 0.45 * p)
            f.dots("r", [mx], [my], 4.4, 1.6 * p)
            lab = "STOPPED" if t >= PEAK else f"{mu['dedx']:07.3f} MEV/CM"
            f.text("r", g1 - 4, y0 + 36, hud.erode(lab, 1 - p, 121, fr), size=L.T_SMALL, anchor="rs", alpha=p)
            f.text("w", g1 - 4, y0 + 62, hud.erode(f"BETA {mu['beta']:.4f}", 1 - p, 122, fr), size=L.T_MICRO,
                   anchor="rs", alpha=0.8 * p)
            f.text("w", g1 - 4, y0 + 84, hud.erode(f"R {mu['R']:.3f} CM", 1 - p, 123, fr), size=L.T_MICRO,
                   anchor="rs", alpha=0.8 * p)
        if self.pan_bar and pwc > 0.01:
            # centre tower: photomultiplier signal >> barcode
            g0, g1 = self.pan_bar
            hud.panel_header(f, g0, g1, y0, hud.erode("PMT_C >> BARCODE", 1 - pwc, 130, fr), alpha=pwc)
            cols = int(np.clip((g1 - g0) / 3.2, 45, 150)) // 3 * 3
            dt = 3.0 / cols
            kf = math.floor((t - 3.0) / dt)
            kk = kf + np.arange(cols)
            ts = kk * dt
            pulse = np.array([ctx.cues.kick(float(v), tau=0.2) for v in ts[::3]]).repeat(3)[:cols]
            dens = (0.05 + 0.5 * _level(ts) + 0.42 * np.tanh(0.8 * pulse)) * np.array(
                [float(self._power("C", float(v))) for v in ts])
            hud.barcode_lanes(f, g0, g1, y0 + 12, y1, dens, kk, lanes=3, seed=9)
        if self.pan_lvl:
            g0, g1 = self.pan_lvl
            p = float(st["pw"]["R"])
            if p > 0.01:
                hud.panel_header(f, g0, g1, y0, hud.erode("LEVEL", 1 - p, 140, fr), alpha=p)
                f.text("r" if st["beam"] > 0.5 else "w", g0, y0 + 84, hud.erode(f"{st['lv']:.3f}", 1 - p, 141, fr),
                       size=44, alpha=p)
                f.text("w", g0 + 2, y0 + 114, hud.erode("LIGHT" if st["beam"] > 0.5 else "RISING" if t < PEAK else "STOPPED",
                                                        1 - p, 142, fr), size=L.T_MICRO, alpha=0.8 * p)
