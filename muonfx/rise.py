"""RISE - everything it was ends as light. Ryoji Ikeda edition.   Scene 9, 08:32 - 10:06 (94 s).

A muon that stops in a detector gives everything away. As it slows down its
energy loss rises (Bethe-Bloch, dE/dx ~ 1/beta^2: the Bragg rise) until it
stops; then it decays and the last of it ends as scintillation light, about
10 000 photons per MeV climbing the towers to the photomultipliers.

The three detector towers become columns of rising light: CENTRE = hero (bass
hits bloom at its head, a fan of rays opens above it), LEFT / RIGHT = mirrored
glockenspiel arps on a pitch ladder. Linear (not a loop), follows the sheet,
local time 0 = 08:32:
  9.0  00-39 s  RISE CRESCENDO / GLOCKS  the columns fill and climb; the muon climbs the Bragg
                                          curve: speed falls, dE/dx rises, the light piles up
  9.1  39-63 s  VOICE OVER               the identity card is stripped on the words - the speed,
                                          the mass, the name - and rewritten as light (mass 0,
                                          speed c); the centre column becomes a beam
  9.2  63-94 s  RISE BREAK               power down, scene 3's turning-on sequence reversed
                                          (centre, right, left); muons keep rushing through the
                                          dark until one is seen: "now you know"
The voice-over cue times are estimates, to be synced to the audio.
"""
from __future__ import annotations

import math

import numpy as np

from .engine import Frame, hash01, smoothstep
from .hud import (BAR, BEAT, BOT, GAPS, MAIN, STRIP, TOWERS, barcode_lanes, big_number, erode, finish,
                  panel_header, show_time, timeline_strip, tower_frame)

T_SHOW = 8 * 60 + 32.0                        # scene start in the show
SECTIONS = [("9.0", "RISE CRESCENDO / GLOCKS", 0.0, 39.0), ("9.1", "VOICE OVER", 39.0, 63.0),
            ("9.2", "RISE BREAK", 63.0, 94.0)]
PEAK = 39.0                                   # the muon stops
CUE_STRIP, CUE_SPEED, CUE_MASS, CUE_NAME, CUE_LIGHT = 41.0, 45.5, 48.0, 50.5, 57.0
KNOW = 86.5                                   # "but now, now, you know": one muon is seen
VO = [(40.0, "EVERYTHING IT WAS"), (42.6, "STRIPPED AWAY"), (CUE_SPEED, "THE SPEED"), (CUE_MASS, "THE MASS"),
      (CUE_NAME, "THE NAME"), (54.0, "WHEREVER IT BEGAN"), (CUE_LIGHT, "IT ENDS AS LIGHT"),
      (66.5, "HERE. RIGHT NOW"), (72.5, "RUSHING THROUGH YOU"), (79.0, "YOU STILL CANNOT FEEL IT"),
      (KNOW, "NOW YOU KNOW")]
OFF = {"C": 63.4, "R": 64.2, "L": 65.0}      # scene 3 switched L, R, C on: power down in reverse, fast
HEAD = {"C": 640.0, "L": 905.0, "R": 905.0}   # where the column heads arrive at the peak
TOP = 332.0                                   # the beam reaches the top of the wall
CLIP_COL = (40.0, 330.0, 2960.0, 1612.0)

# Bethe-Bloch in plastic scintillator (PVT)
M_MU, M_E = 105.6583755, 0.51099895           # MeV
K_BB, ZA, RHO, I_EXC = 0.307075, 0.54141, 1.032, 64.7e-6
E0 = 3490.1                                   # total energy of the muon entering the tower (MeV)
YIELD = 10000.0                               # scintillation photons per MeV
E_MICHEL = 37.9                               # the decay electron it ends with (MeV)

# glockenspiel arps: A minor pentatonic, one rung per note, mirrored on both side towers
SCALE = [("A4", 0), ("C5", 3), ("D5", 5), ("E5", 7), ("G5", 10), ("A5", 12), ("C6", 15), ("D6", 17), ("E6", 19),
         ("G6", 22), ("A6", 24), ("C7", 27), ("D7", 29)]
N_RUNG = len(SCALE)
ARP = [0, 2, 4, 6, 8, 6, 4, 2, 1, 3, 5, 7, 9, 7, 5, 3]
CHORD = [0, 2, 1, 3]


def _bethe(Tk):
    g = 1.0 + Tk / M_MU
    b2 = 1.0 - 1.0 / g ** 2
    return K_BB * ZA * RHO / b2 * (np.log(2 * M_E * b2 * g * g / I_EXC) - b2), np.sqrt(b2)


def _level(t):
    """Crescendo of the rise: 0 -> 1 over 9.0, held through 9.1."""
    t = np.asarray(t, np.float64)
    return np.where(t >= PEAK, 1.0, 0.05 + 0.95 * np.clip(t / PEAK, 0, 1) ** 1.5)


def _power(key, t):
    return 1.0 - smoothstep(OFF[key], OFF[key] + 0.55, t)


def _thousands(n):
    return f"{int(n):,}".replace(",", " ")


class Rise:
    name = "rise"

    def __init__(self, T=94.0, seed=9):
        self.T = T
        rng = np.random.default_rng(seed)
        # range-energy table of the stopping muon
        Tk = np.geomspace(0.25, E0 - M_MU, 1500)
        dedx, beta = _bethe(Tk)
        R = np.concatenate([[0.0], np.cumsum(np.diff(Tk) / (0.5 * (dedx[1:] + dedx[:-1])))]) + 0.25 / dedx[0]
        self.tab = dict(Tk=Tk, dedx=dedx, beta=beta, lR=np.log10(R))
        self.lR0, self.lR1 = float(np.log10(R[-1])), float(np.log10(R[0]))
        self.cols = {k: self._column(rng, k, 26000 if k == "C" else 10000) for k in TOWERS}
        self._build_events(rng)
        self.fan_u = rng.random(300)
        self.fan_s = np.where(rng.random(300) < 0.5, -1.0, 1.0)
        self.fan_ph = rng.uniform(0, 2 * np.pi, 300)

    # ------------------------------------------------------------------ build
    @staticmethod
    def _column(rng, key, n):
        x0, x1, top, bot = TOWERS[key]
        u = rng.uniform(-1.0, 1.0, n)
        spill = rng.random(n) < 0.04
        u[spill] *= rng.uniform(1.0, 1.4, spill.sum())
        return dict(cx=(x0 + x1) / 2, hw=(x1 - x0) / 2, bot=bot, u=u.astype(np.float32),
                    ph=rng.random(n).astype(np.float32), spd=rng.uniform(0.08, 0.3, n).astype(np.float32),
                    r=(0.65 + 1.1 * rng.random(n) ** 3).astype(np.float32),
                    b=rng.uniform(0.45, 1.0, n).astype(np.float32))

    def _build_events(self, rng):
        n_bar = int(PEAK // BAR)
        # bass hits on the centre tower (TBC on the sheet): downbeats, twice a bar at the top of the rise,
        # every other bar under the voice
        bass = [b * BAR for b in range(2, n_bar + 1)] + [b * BAR + 2 * BEAT for b in range(n_bar - 4, n_bar)]
        bass += [PEAK] + [PEAK + k * 2 * BAR for k in range(1, 7)]
        self.bass = np.array(sorted(b for b in bass if b <= PEAK or b < OFF["C"] - 1.0))
        self.bass_ang = rng.uniform(-math.pi, 0.0, (len(self.bass), 64))
        self.bass_len = rng.uniform(0.25, 1.0, (len(self.bass), 64))
        # fast arps on the side towers, 16ths, denser as it rises, quieter under the voice
        notes = []
        for k in range(int(OFF["R"] / (BEAT / 4))):
            tt = k * BEAT / 4
            bar, step = divmod(k, 16)
            lv = float(_level(tt))
            if tt < PEAK and rng.random() > 0.45 + 0.55 * lv:
                continue
            if tt >= PEAK and rng.random() > 0.6:
                continue
            idx = int(np.clip(CHORD[bar % 4] + ARP[step], 0, N_RUNG - 1))
            vel = (1.0 if step % 4 == 0 else 0.78 if step % 2 == 0 else 0.58) * (0.45 + 0.55 * lv)
            notes.append((tt, idx, vel * (0.55 if tt >= PEAK else 1.0)))
        self.notes = np.array(notes, np.float64)
        # muons entering the towers: the rate follows the crescendo
        mus, tt = [], 0.4
        while tt < OFF["C"]:
            key = rng.choice(["L", "C", "R"], p=[0.3, 0.4, 0.3])
            x0, x1, top, bot = TOWERS[key]
            mus.append((tt, key, rng.uniform(x0 + 12, x1 - 12), rng.normal(0, 0.1)))
            tt += rng.exponential(1.0 / (0.5 + 3.0 * float(_level(tt)) if tt < PEAK else 1.0))
        self.mus = mus
        # 9.2: muons still rushing through the dark, filtered; one bright one when "you know"
        rain, tt = [], OFF["L"] + 1.2
        while tt < self.T - 0.5:
            rain.append((tt, rng.uniform(80, 2920), rng.normal(0, 0.06), 0.3))
            tt += rng.exponential(0.8)
        rain.append((KNOW, TOWERS["C"][0] + 30.0, 0.0, 1.0))
        self.rain = rain
        self.soft = sorted((rng.uniform(OFF["L"] + 2.0, self.T - 2.0), rng.choice(["L", "R"])) for _ in range(9))

    # ------------------------------------------------------------------ state
    def _muon(self, t):
        """The stopping muon at time t: kinetic energy, beta, gamma, dE/dx, residual range."""
        tb = self.tab
        if t >= PEAK:
            return dict(Tk=0.0, beta=0.0, gam=1.0, dedx=float(tb["dedx"][0]), R=0.0, lR=self.lR1)
        lr = self.lR0 + (self.lR1 - self.lR0) * (t / PEAK) ** 0.9
        Tk = float(np.interp(lr, tb["lR"], tb["Tk"]))
        return dict(Tk=Tk, beta=float(np.interp(lr, tb["lR"], tb["beta"])), gam=1.0 + Tk / M_MU,
                    dedx=float(np.interp(lr, tb["lR"], tb["dedx"])), R=10 ** lr, lR=lr)

    def _state(self, t):
        lv = float(_level(t))
        pw = {k: float(_power(k, t)) for k in TOWERS}
        yh = {}
        for k, (x0, x1, top, bot) in TOWERS.items():
            y = bot - (bot - HEAD[k]) * lv
            if k == "C" and t >= CUE_LIGHT:
                y -= (HEAD["C"] - TOP) * float(smoothstep(CUE_LIGHT, CUE_LIGHT + 0.4, t))
            yh[k] = bot - (bot - y) * pw[k] ** 0.7
        mu = self._muon(t)
        e_dep = (E0 - M_MU) - mu["Tk"] + E_MICHEL * float(smoothstep(CUE_LIGHT, CUE_LIGHT + 1.2, t))
        a = t - self.bass
        pulse = float(np.sum(np.where(a >= 0, np.exp(-np.maximum(a, 0) / 0.28), 0.0)))
        return dict(lv=lv, pw=pw, yh=yh, mu=mu, n_gamma=YIELD * e_dep, e_dep=e_dep, pulse=pulse,
                    beam=float(smoothstep(CUE_LIGHT, CUE_LIGHT + 0.4, t)) * pw["C"])

    def _active_notes(self, t, span=0.6):
        n = self.notes
        m = (n[:, 0] <= t) & (n[:, 0] > t - span)
        return n[m]

    def _rung_y(self, key, idx, yh):
        bot = TOWERS[key][3]
        return bot - (idx + 1) / (N_RUNG + 1) * (bot - yh)

    # ----------------------------------------------------------------- render
    def render(self, t, W=3000, H=1688):
        t = float(np.clip(t, 0.0, self.T - 1e-6))
        f = Frame(W, H)
        st = self._state(t)
        f.set_clip(*MAIN)
        self._draw_fan(f, t, st)
        self._draw_bass(f, t, st)
        f.set_clip(*CLIP_COL)
        self._draw_towers(f, t, st)
        self._draw_arps(f, t, st)
        self._draw_muons(f, t, st)
        self._draw_rain(f, t)
        f.set_clip()
        self._draw_level(f, t, st)
        self._draw_header(f, t, st)
        self._draw_card(f, t, st)
        self._draw_light(f, t, st)
        self._draw_strip(f, t, st)
        self._draw_bottom(f, t, st)
        invert = PEAK <= t < PEAK + 0.06 or CUE_LIGHT <= t < CUE_LIGHT + 0.06
        return finish(f, invert)

    # --- the towers --------------------------------------------------------------
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
            x0, x1, top, bot = TOWERS[k]
            tower_frame(f, k, 0.1 + 0.45 * pw[k])
            if k == "C":
                gain = (0.55 + 0.45 * lv) * (1.0 + 0.9 * math.tanh(st["pulse"])) * pw[k] * (1.0 + 0.8 * st["beam"])
                self._draw_column(f, self.cols[k], yh[k], t, gain, flare=0.7 * lv * (1 - st["beam"]))
                if st["beam"] > 0.01:     # it ends as light: a solid core burns up the column
                    cx = self.cols[k]["cx"]
                    w = 3.0 + 14.0 * st["beam"]
                    f.rects("w", cx - w, yh[k], cx + w, bot, 0.8 * st["beam"])
            else:
                bands = [(self._rung_y(k, int(i), yh[k]), 2.6 * v * math.exp(-(t - tn) / 0.16)) for tn, i, v in notes]
                self._draw_column(f, self.cols[k], yh[k], t, (0.5 + 0.5 * lv) * pw[k], bands)
            # the head: the muon (red) sits on the light; once it has become light it is white
            if yh[k] < bot - 4 and pw[k] > 0.02:
                cx = self.cols[k]["cx"]
                if k == "C" and t >= CUE_LIGHT:
                    f.dots("w", [cx], [yh[k] - 3], 7.0, 1.8 * pw[k])
                else:
                    f.dots("r", [cx], [yh[k] - 3], 6.5 if k == "C" else 4.5, 1.8 * pw[k])
                    f.dots("w", [cx], [yh[k] - 3], 2.0, 1.0 * pw[k])
        # "now you know": the three towers answer once
        a = t - KNOW
        if 0 <= a < 1.6:
            for k in TOWERS:
                x0, x1, top, bot = TOWERS[k]
                e = math.exp(-a / 0.35)
                f.rect("w", x0, top, x1, bot, 1.4 * e, width=1.6)
                f.rects("w", x0 + 3, top + 3, x1 - 3, bot - 3, 0.12 * e)

    def _draw_arps(self, f, t, st):
        yh, pw = st["yh"], st["pw"]
        notes = self._active_notes(t, 0.7)
        for k, sgn in (("L", 1.0), ("R", -1.0)):
            if pw[k] < 0.01 or yh[k] > TOWERS[k][3] - 60:
                continue
            x0, x1, top, bot = TOWERS[k]
            edge = x1 if sgn > 0 else x0
            ys = np.array([self._rung_y(k, i, yh[k]) for i in range(N_RUNG)])
            f.segments("w", np.full(N_RUNG, edge), ys, np.full(N_RUNG, edge + sgn * 10), ys, 0.55 * pw[k])
            for i, y in enumerate(ys):
                f.text("w", edge + sgn * 14, y + 4, SCALE[i][0], size=10, alpha=0.4 * pw[k],
                       anchor="ls" if sgn > 0 else "rs")
            for j, (tn, idx, vel) in enumerate(notes):
                a = t - tn
                y = ys[int(idx)]
                e = math.exp(-a / 0.16) * pw[k]
                L = (50 + 200 * vel) * (1 - math.exp(-a / 0.025))
                xe = edge + sgn * (40 + L)
                f.segments("w", [edge + sgn * 40], [y], [xe], [y], 1.25 * vel * e, 0.35 * vel * e)
                f.dots("w", [edge + sgn * 3], [y], 3.2, 1.6 * vel * e)
                f.crosses("w", [xe], [y], 5.0, 0.9 * e)
                if j == len(notes) - 1 and a < 0.4:
                    name, semi = SCALE[int(idx)]
                    f.text("w", xe + sgn * 12, y - 8, f"{name} {440 * 2 ** (semi / 12):07.2f} HZ", size=12,
                           alpha=min(1.0, 3 * e), anchor="ls" if sgn > 0 else "rs")

    def _draw_muons(self, f, t, st):
        for (tm, key, x, ang) in self.mus:
            a = t - tm
            if not (0 <= a < 0.9):
                continue
            top = TOWERS[key][2]
            y0 = MAIN[1] + 2
            dur = (top - y0) / 2600.0
            u = min(1.0, a / dur)
            xs = x - math.tan(ang) * (top - y0)
            xh, yh = xs + (x - xs) * u, y0 + (top - y0) * u
            fade = 1.0 if a < dur else math.exp(-(a - dur) / 0.2)
            f.segments("r", [xs], [y0], [xh], [yh], 0.55 * fade * st["pw"][key])
            if a < dur:
                f.dots("r", [xh], [yh], 2.6, 1.6)
            else:
                e = math.exp(-(a - dur) / 0.12)
                f.dots("w", [x], [top], 3.0, 1.6 * e)
                f.rings("w", [x], [top], [8 + 40 * (1 - e)], 0.6 * e)

    def _draw_rain(self, f, t):
        for (tm, x, ang, amp) in self.rain:
            a = t - tm
            if not (0 <= a < (6.0 if amp >= 1.0 else 1.2)):
                continue
            y0, y1 = 334.0, 1606.0
            dur = (y1 - y0) / 5200.0
            u = min(1.0, a / dur)
            xe = x + math.tan(ang) * (y1 - y0)
            fade = 1.0 if a < dur else math.exp(-(a - dur) / (0.35 if amp < 1 else 0.6))
            f.segments("r", [x], [y0], [x + (xe - x) * u], [y0 + (y1 - y0) * u], (0.5 + 0.6 * amp) * amp * fade,
                       width=1.0 + 0.6 * amp)
            if a < dur:
                f.dots("r", [x + (xe - x) * u], [y0 + (y1 - y0) * u], 2.0 + 2.0 * amp, 1.5 * amp)
            if amp >= 1.0 and a > 0.1:
                al = 1.0 - float(smoothstep(4.0, 6.0, a))
                y = 760.0
                xx = x + (xe - x) * (y - y0) / (y1 - y0)
                f.segments("w", [xx, xx + 40], [y, y - 40], [xx + 40, xx + 70], [y - 40, y - 40], 0.7 * al)
                f.tag("r", xx + 78, y - 33, "MU-", size=20, pad=5, alpha=al)
                f.text("w", xx + 78, y - 2, erode("ONE OF 17 /S THROUGH YOU", 1 - min(1.0, (a - 0.1) * 3), 7),
                       size=15, alpha=0.9 * al)
        for (tm, key) in self.soft:           # the filtered hits of the break
            a = t - tm
            if 0 <= a < 2.5:
                x0, x1, top, bot = TOWERS[key]
                u = a / 2.5
                f.rings("w", [(x0 + x1) / 2], [top], [20 + 180 * (1 - (1 - u) ** 2)], 0.22 * (1 - u) ** 1.5,
                        width=2.0)

    # --- the centre: fan of rays, bass blooms, level ---------------------------------
    def _draw_fan(self, f, t, st):
        yh = st["yh"]["C"]
        amt = float(smoothstep(1250.0, 700.0, yh)) * st["lv"] * st["pw"]["C"] * (1.0 - st["beam"])
        if amt < 0.01:
            return
        cx = self.cols["C"]["cx"]
        n = int(30 + 250 * amt)
        u, s = self.fan_u[:n], self.fan_s[:n]
        xe = cx + s * (u ** 1.3) * 1440.0
        xs = cx + s * 46.0 * (0.25 + 0.75 * u)
        tw = 0.6 + 0.4 * np.sin(self.fan_ph[:n] + t * (2.0 + 7.0 * u))
        i0 = amt * (0.06 + 0.32 * (1 - u) ** 2) * tw * (1.0 + 0.6 * math.tanh(st["pulse"]))
        f.segments("w", xs, np.full(n, yh + 4.0), xe, np.full(n, MAIN[1] + 1.0), i0, i0 * 0.3)

    def _draw_bass(self, f, t, st):
        cx = self.cols["C"]["cx"]
        yh = st["yh"]["C"]
        pw = st["pw"]["C"]
        a = t - self.bass
        for j in np.nonzero((a >= 0) & (a < 1.3))[0]:
            aj = float(a[j])
            u = aj / 1.3
            e = (1 - u) ** 2 * pw
            f.rings("w", [cx], [yh], [30 + 780 * (1 - (1 - u) ** 3)], 0.5 * e)
            f.rings("r", [cx], [yh], [18 + 320 * (1 - (1 - u) ** 3)], 0.9 * e, width=1.3)
            grow = 1 - (1 - min(1.0, aj / 0.3)) ** 3
            ang = self.bass_ang[j]
            r0 = 24 + 90 * u
            r1 = r0 + self.bass_len[j] * 440 * grow
            f.segments("w", cx + np.cos(ang) * r0, yh + np.sin(ang) * r0, cx + np.cos(ang) * r1,
                       yh + np.sin(ang) * r1, 0.5 * e, 0.0)
            f.dots("w", cx + np.cos(ang) * r1, yh + np.sin(ang) * r1, 1.7, 1.0 * e)

    def _draw_level(self, f, t, st):
        pw = st["pw"]["C"] * (1.0 - st["beam"])
        yh = st["yh"]["C"]
        if pw < 0.01 or yh > TOWERS["C"][3] - 40:
            return
        mu = st["mu"]
        xl, xr = TOWERS["L"][1] + 6, TOWERS["R"][0] - 6
        y = max(yh, MAIN[1] + 30)
        f.segments("r", [xl], [y], [xr], [y], 0.85 * pw, width=1.2)
        f.tag("r", xl + 40, y - 10, f"LEVEL {st['lv']:.3f}", size=13, pad=3, alpha=pw)
        f.tag("r", xr - 40, y - 10, f"DE/DX {mu['dedx']:07.3f} MEV/CM", size=13, pad=3, alpha=pw, anchor="rs")
        # intensity read off the column, the way the reference prints it
        x = TOWERS["C"][1] + 16
        f.text("w", x, y + 26, "INTENSITY: MAX", size=11, alpha=0.8 * pw)
        ys = np.arange(y + 50, TOWERS["C"][3] - 6, 21.0)
        q = 1.0 - (ys - y) / max(TOWERS["C"][3] - y, 1.0)
        v = np.clip(st["lv"] * (0.12 + 0.88 * q ** 2.4) * (1 + 0.5 * math.tanh(st["pulse"]))
                    + 0.04 * hash01(np.arange(len(ys)), int(t * 12)), 0, 0.99)
        for yy, vv in zip(ys, v):
            f.text("w", x, yy, f"{vv:.2f}", size=11, alpha=0.7 * pw)

    # --- HUD ------------------------------------------------------------------------
    def _draw_header(self, f, t, st):
        k = 0 if t < SECTIONS[1][2] else 1 if t < SECTIONS[2][2] else 2
        code, name, s0, s1 = SECTIONS[k]
        f.tag("r" if k < 2 else "w", 60, 530, f"{code} // {name}", size=16, pad=4)
        f.text("w", 2940, 530, "RISE // MUON BLOOM // SCENE 09 // DE/DX -> LIGHT", size=14, alpha=0.85, anchor="rs")
        if t >= OFF["L"] + 0.6:
            al = float(smoothstep(OFF["L"] + 0.6, OFF["L"] + 2.0, t))
            f.text("w", 60, 566, "DETECTORS OFFLINE // MU FLUX UNCHANGED 1 /CM2/MIN", size=14, alpha=0.8 * al)

    def _draw_card(self, f, t, st):
        """Identity of the muon, stripped on the voice-over, rewritten as light."""
        pw = st["pw"]["L"]
        if pw < 0.01:
            return
        fr = int(t * 30)
        gone = 1.0 - pw
        mu = st["mu"]
        x, y0 = 60.0, 612.0
        light = t >= CUE_LIGHT
        f.tag("r" if light else "w", x, y0, erode("LIGHT // IDENTITY" if light else "MU- // IDENTITY", gone, 1, fr),
              size=16, pad=4, alpha=pw)
        # the big figure: its speed
        if t < CUE_SPEED:
            big = f"{mu['beta']:.6f}"
        elif not light:
            big = erode(f"{0.0:.6f}", (t - CUE_SPEED) / 0.9, 3, fr)
        else:
            big = f"{1.0:.6f}"[: int((t - CUE_LIGHT) * 30)]
        big_number(f, x, 740, "SPEED // BETA = V/C", erode(big, gone, 4, fr), size=72, layer="r" if light else "w")
        rows = [("NAME", "MUON  MU-", CUE_NAME, "LIGHT"), ("CLASS", "LEPTON // GEN 2", CUE_STRIP, None),
                ("CHARGE", "-1 E", CUE_STRIP + 0.5, None),
                ("MASS", "105.6583755 MEV/C2", CUE_MASS, "0.0000000 MEV/C2"),
                ("SPEED", f"{mu['beta']:.6f} C", CUE_SPEED, "1.000000 C"),
                ("GAMMA", f"{mu['gam']:.4f}", CUE_STRIP + 1.0, None),
                ("ENERGY", f"{(mu['Tk'] + M_MU) / 1000:.5f} GEV", CUE_STRIP + 1.5, None),
                ("LIFETIME", "2.1969811 US", CUE_STRIP + 2.0, None),
                ("ORIGIN", "15.21 KM // PI- DECAY", CUE_STRIP + 2.5, None)]
        yy = 800.0
        vx = x + 130
        for k, (lab, val, cue, after) in enumerate(rows):
            y = yy + k * 30
            a = t - cue
            lab_al = 0.85 if a < 0.5 else 0.3
            if light and after:
                lab_al = 0.95
                n = int(max(0.0, t - CUE_LIGHT - 0.25 * k) * 34)
                val_s, lay = after[:n], "r" if lab == "NAME" else "w"
            elif a < 0:
                val_s, lay = val, "w"
            else:
                val_s, lay = erode(val, (a - 0.15) / 0.9, 11 + k, fr), "w"
                if a < 1.4:     # a red cut runs through the value first
                    w = len(val) * 9.6 * min(1.0, a / 0.3)
                    f.segments("r", [vx - 4], [y - 6], [vx - 4 + w], [y - 6], 1.2 * (1 - a / 1.4) * pw, width=1.6)
            f.text("w", x, y, erode(lab, gone, 21 + k, fr), size=15, alpha=lab_al * pw)
            f.text(lay, vx, y, erode(val_s, gone, 31 + k, fr), size=15, alpha=0.95 * pw)
        # hit stream: the latest light pulses in the towers
        y = 1100.0
        f.tag("w", x, y, erode("HIT_STREAM", gone, 41, fr), size=12, pad=3, alpha=pw)
        k = 0
        for (tm, key, xx, ang) in reversed(self.mus):
            if tm > t:
                continue
            if k >= 16:
                break
            line = f"{tm:07.3f} DET_{key} {xx:07.2f} {TOWERS[key][2]:06.1f} {abs(ang) * 57.3:05.2f} DEG"
            f.text("r" if k == 0 else "w", x, y + 30 + k * 19, erode(line, gone, 50 + k, fr), size=12,
                   alpha=(0.95 if k < 2 else 0.6) * pw)
            k += 1

    def _draw_light(self, f, t, st):
        """Right column: the light it becomes."""
        pw = st["pw"]["R"]
        if pw < 0.01:
            return
        fr = int(t * 30)
        gone = 1.0 - pw
        mu = st["mu"]
        x0, x1 = 2510.0, 2940.0
        f.tag("w", x0, 612, erode("LIGHT // YIELD", gone, 61, fr), size=16, pad=4, alpha=pw)
        big_number(f, x0, 740, "SCINTILLATION PHOTONS", erode(_thousands(st["n_gamma"]), gone, 62, fr), size=52,
                   layer="w")
        v = {k: min(1.0, (0.2 + 0.8 * st["lv"]) * st["pw"][k]) for k in TOWERS}
        rows = [f"YIELD     10 000 /MEV", f"E_DEP     {st['e_dep']:09.3f} MEV",
                f"DE/DX     {mu['dedx']:07.3f} MEV/CM", f"RANGE     {mu['R']:09.4f} CM",
                f"PMT_L     {v['L'] * 0.71:.3f} V", f"PMT_C     {v['C'] * (0.8 + 0.2 * math.tanh(st['pulse'])):.3f} V",
                f"PMT_R     {v['R'] * 0.71:.3f} V", f"MU RATE   {0.5 + 3.0 * st['lv']:.2f} /S"]
        for k, r in enumerate(rows):
            f.text("w", x0, 800 + k * 30, erode(r, gone, 70 + k, fr), size=15, alpha=0.9 * pw)
        y = 1100.0
        f.tag("w", x0, y, erode("NOTE_STREAM // L+R", gone, 81, fr), size=12, pad=3, alpha=pw)
        n = self.notes[self.notes[:, 0] <= t][-16:][::-1]
        for k, (tn, idx, vel) in enumerate(n):
            name, semi = SCALE[int(idx)]
            line = f"{tn:07.3f} {name:<3}{440 * 2 ** (semi / 12):08.2f} HZ  {vel:.2f}"
            f.text("r" if k == 0 else "w", x0, y + 30 + k * 19, erode(line, gone, 90 + k, fr), size=12,
                   alpha=(0.95 if k < 2 else 0.6) * pw)

    def _draw_strip(self, f, t, st):
        al = 1.0 - 0.6 * float(smoothstep(OFF["C"], OFF["L"] + 1.0, t))
        X, yb = timeline_strip(f, t, self.T, SECTIONS, "RISE // SCENE 09 // EVERYTHING IT WAS ENDS AS LIGHT",
                               T_SHOW, al)
        x0, y0, x1, y1 = STRIP
        tb = np.arange(0.0, self.T, BEAT)
        lv = _level(tb) * _power("C", tb)
        hh = 3 + 44 * lv
        past = tb <= t
        xb = X(tb)
        f.rects("w", xb, y0 + 1, xb + 2, y0 + 1 + hh, np.where(past, 0.95, 0.3) * al)
        xs = X(self.bass)
        f.rects("r", xs - 1, y0 + 1, xs + 3, y0 + 58, np.where(self.bass <= t, 1.0, 0.35) * al)
        nt, ni = self.notes[:, 0], self.notes[:, 1]
        xn = X(nt)
        yn = y1 - 3 - ni * 3.0
        f.rects("w", xn, yn - 2, xn + 2, yn, np.where(nt <= t, 0.9, 0.3) * al)
        placed = []
        for tv, word in VO + [(OFF[k], f"DET_{k} OFF") for k in ("C", "R", "L")]:
            xv = float(X(tv))
            row = sum(1 for p in placed if abs(p - xv) < 150) % 2
            placed.append(xv)
            f.tag("r" if tv <= t < tv + 2.5 else "w", xv, yb + 32 + 22 * row, word, size=12, pad=3,
                  alpha=(1.0 if tv <= t else 0.4) * al)

    def _draw_bottom(self, f, t, st):
        x0, y0, x1, y1 = BOT
        pw = st["pw"]
        fr = int(t * 30)
        # side towers: photomultiplier traces of the arps
        tt = t - 2.0 + np.linspace(0.0, 2.0, 400)
        nn = self.notes[(self.notes[:, 0] > t - 3.0) & (self.notes[:, 0] <= t)]
        sig = np.zeros_like(tt)
        for tn, idx, vel in nn:
            a = tt - tn
            ok = a >= 0
            sig[ok] += vel * np.exp(-a[ok] / 0.07) * (1 - np.exp(-a[ok] / 0.004))
        for gi, k in ((0, "L"), (3, "R")):
            g0, g1 = GAPS[gi]
            p = float(pw[k])
            if p < 0.01:
                continue
            panel_header(f, g0, g1, y0, erode(f"PMT_{k} // ARP", 1 - p, 100 + gi, fr))
            v = sig * _power(k, tt)
            xs = np.linspace(g0 + 4, g1 - 4, len(tt))
            f.polyline("w", xs, y1 - 12 - np.clip(v, 0, 1.4) * 72, 0.85 * p)
            f.segments("w", [g0 + 4], [y1 - 12], [g1 - 4], [y1 - 12], 0.3 * p)
            f.text("w", g1 - 6, y0 + 32, erode(f"{v[-1] * 0.71:.3f} V", 1 - p, 110 + gi, fr), size=13,
                   alpha=0.85 * p, anchor="rs")
        # Bragg rise: dE/dx against residual range, log-log, the stopping point on the right
        g0, g1 = GAPS[1]
        p = float(pw["C"])
        if p > 0.01:
            panel_header(f, g0, g1, y0, erode("BRAGG // DE/DX VS RESIDUAL RANGE", 1 - p, 120, fr))
            px0, px1, py0, py1 = g0 + 16, g1 - 150, y0 + 22, y1 - 22
            tb = self.tab
            d0, d1 = math.log10(1.5), math.log10(float(tb["dedx"][0]) * 1.1)

            def PX(lr):
                return px0 + (self.lR0 - np.asarray(lr)) / (self.lR0 - self.lR1) * (px1 - px0)

            def PY(d):
                return py1 - (np.log10(d) - d0) / (d1 - d0) * (py1 - py0)

            f.segments("w", [px0, px0], [py0, py1], [px0, px1], [py1, py1], 0.6 * p)
            for dec in range(math.ceil(self.lR1), math.floor(self.lR0) + 1):
                xx = float(PX(dec))
                f.segments("w", [xx], [py1], [xx], [py1 + 6], 0.7 * p)
                f.text("w", xx + 3, py1 + 18, f"1E{dec:+d}", size=10, alpha=0.6 * p)
            sel = slice(None, None, 8)
            f.dots("w", PX(tb["lR"][sel]), PY(tb["dedx"][sel]), 0.9, 0.35 * p)
            mu = st["mu"]
            m = tb["lR"] >= mu["lR"]
            if m.sum() > 1:
                f.polyline("w", PX(tb["lR"][m][::-1]), PY(tb["dedx"][m][::-1]), 1.0 * p, width=1.3)
            mx, my = float(PX(mu["lR"])), float(PY(mu["dedx"]))
            f.segments("r", [mx, px0], [py0 - 4, my], [mx, px1], [py1, my], 0.45 * p)
            f.dots("r", [mx], [my], 4.0, 1.6 * p)
            lab = "STOPPED" if t >= PEAK else f"{mu['dedx']:07.3f} MEV/CM"
            f.text("r", g1 - 6, y0 + 32, erode(lab, 1 - p, 121, fr), size=13, anchor="rs", alpha=p)
            f.text("w", g1 - 6, y0 + 54, erode(f"BETA {mu['beta']:.4f}", 1 - p, 122, fr), size=12, anchor="rs",
                   alpha=0.8 * p)
            f.text("w", g1 - 6, y0 + 74, erode(f"R {mu['R']:.4f} CM", 1 - p, 123, fr), size=12, anchor="rs",
                   alpha=0.8 * p)
        # centre tower: photomultiplier signal >> barcode
        g0, g1 = GAPS[2]
        if p > 0.01:
            panel_header(f, g0, g1, y0, erode("PMT_C >> BARCODE", 1 - p, 130, fr))
            cols = 220
            dt = 3.0 / cols
            kf = math.floor((t - 3.0) / dt)
            kk = kf + np.arange(cols)
            ts = kk * dt
            pulse = np.zeros(cols)
            for tb_ in self.bass[(self.bass > t - 5) & (self.bass <= t)]:
                a = ts - tb_
                pulse += np.where(a >= 0, np.exp(-np.maximum(a, 0) / 0.25), 0.0)
            dens = (0.05 + 0.55 * _level(ts) + 0.4 * np.tanh(pulse)) * _power("C", ts)
            barcode_lanes(f, g0, g1, y0 + 10, y1, dens, kk, lanes=3, seed=9)
