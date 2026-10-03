"""Everything the scenes read that is not geometry: the sheet timeline, the voice-over subtitles,
the audio cues and the (simulated) detector streams.

In the realtime app the three detector values arrive live over OSC; here they are rebuilt from the
'just muon sounds' stem (tools/analyze_audio.py), so the previews bloom where the music blooms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .layout import DATA, ROOT

# The audio is "muon bloom Mixed v1 scene 10 edit" (2026-10-02). Against V7 (tools/align_audio.py): identical up to
# 09:35; scene 10 is cut and its best part ends scene 9 (9.2, 09:35 - 09:48); scene 11 starts at 09:48 (V7 10:52,
# -64 s) and from 10:00 on everything is V7 - 75.56 s. The music ends at 12:38.5 (the file is padded with silence).
SHOW_END = 726.44               # 12:06.4, end of scene 11: the credits follow
TRACK_END = 758.48              # 12:38.5, the end of the music

T_BLOOM_CUT = 133.3             # the cut to the towers lands on the hit under "A bloom", not on 02:13.0

# (code, name, time in, time out, look) - times from Muon_Bloom_Scenes_TimeCodes_V2, moved to the scene 10 edit
SECTIONS = [
    ("1.0", "INTRO", 0.0, 31.0, "origin"),
    ("1.1", "BILLIONS BUILD", 31.0, 44.0, "star"),
    ("1.1", "VOICE OVER", 44.0, 56.0, "messenger"),
    ("1.2", "DRUMS ENTER", 56.0, 67.0, "messenger"),
    ("1.2", "ATMOSPHERE", 67.0, 72.4, "shower"),
    ("1.3", "YOU", 72.4, 101.0, "you"),
    ("1.4", "MUON", 101.0, 104.0, "muon"),
    ("2.0", "THESE DETECTORS", 104.0, 109.0, "detector"),
    ("2.1", "WAITING BREAK", 109.0, 112.0, "detector"),
    ("2.1", "VOICE OVER", 112.0, T_BLOOM_CUT, "detector"),
    ("2.2", "BLOOM", T_BLOOM_CUT, 139.0, "bloom"),
    ("3.1", "LEFT DETECTOR", 139.0, 151.0, "bloom"),
    ("3.2", "RIGHT DETECTOR", 151.0, 159.0, "bloom"),
    ("3.3", "CENTRE DETECTOR", 159.0, 180.0, "bloom"),
    ("4.0", "DATA ON / AMBIENT", 180.0, 236.0, "galaxy"),
    ("5.0", "COSMIC GROOVE", 236.0, 276.0, "sphere"),
    ("5.0", "FLOOD", 276.0, 299.0, "flood"),
    ("5.1", "COSMIC BREAK", 299.0, 322.0, "flood"),
    ("6.0", "DANCE", 322.0, 415.0, "dance"),
    ("7.0", "BREAK BEFORE DRUMS", 415.0, 422.0, "glitch"),
    ("7.1", "DRUMS IN", 422.0, 447.0, "glitch"),
    ("7.3", "TRANSITION", 447.0, 454.0, "glitch"),
    ("8.0", "OUTLAST", 454.0, 502.0, "outlast"),
    ("8.1", "OUTLAST BUILD", 502.0, 512.0, "outlast"),
    ("9.0", "RISE CRESCENDO", 512.0, 551.0, "rise"),
    ("9.1", "RISE VOICE OVER", 551.0, 575.0, "rise"),
    ("9.2", "RISE ENDING", 575.0, 588.0, "disintegrate"),     # scene 10 (cut) condensed: the ending of scene 9
    ("11.0", "NARRATIVE", 588.0, 664.44, "outro"),
    ("11.1", "CRESCENDO / ACCELERANDO", 664.44, 700.44, "outro"),
    ("11.2", "SWIRLING OUTRO", 700.44, SHOW_END, "outro"),
    ("12.0", "CREDITS", SHOW_END, TRACK_END, "credits"),
]

# detector life cycle: scene 2 reveals them, scene 3 switches them on, then they stay on to the end of the music
# (the outro draws their power-down under the credits)
T_REVEAL = 104.0
T_BLOOM = 133.3                 # the hit under "A bloom" (onset of the music stem)
T_ON = {"L": 147.35, "R": 160.0, "C": 170.5}
KEYS = ("L", "C", "R")

# the muons through one spectator: 1 /cm2/min reaches the ground, over AREA_YOU that is RATE_YOU a second. The
# voice says "more than fifty thousand of them will flood through your skin": the fifty-thousandth goes through
# you in the swirl of 11.2 (11:54), before the credits (the show got shorter with the scene 10 edit: 63 /s and
# 0.38 m2 no longer made it true).
RATE_YOU = 70.0
AREA_YOU = RATE_YOU * 60.0 / 1e4                # 0.42 m2
YEAR = 3.156e7


def spaced(n):
    """12 345 678: thousands set apart by a space, as on the counters."""
    return f"{int(round(n)):,}".replace(",", " ")


def through_you_in(seconds, short=False):
    """Muons through one spectator in `seconds`, written the way the read-outs write it:
    70 / 4 200 / 6 048 000 (short: 6.0 M) / 2.21 BN / 177 BN."""
    n = RATE_YOU * seconds
    if n >= 1e9:
        return f"{n / 1e9:.2f} BN" if n < 1e10 else f"{n / 1e9:.0f} BN"
    if short and n >= 1e6:
        return f"{n / 1e6:.1f} M"
    return spaced(n)


def mmss(t):
    """Seconds -> M:SS or MM:SS, the way the show length is written (12:06)."""
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def section_at(t):
    """(index, section, progress 0..1) of the sheet section holding t."""
    for k, s in enumerate(SECTIONS):
        if t < s[3] or k == len(SECTIONS) - 1:
            return k, s, float(np.clip((t - s[2]) / (s[3] - s[2]), 0.0, 1.0))


def look_span(look, t):
    """Start / end of the run of consecutive sections sharing `look` around time t."""
    k, _, _ = section_at(t)
    a = b = k
    while a > 0 and SECTIONS[a - 1][4] == look:
        a -= 1
    while b < len(SECTIONS) - 1 and SECTIONS[b + 1][4] == look:
        b += 1
    return SECTIONS[a][2], SECTIONS[b][3]


def scene_start(t):
    """Show time at which the scene on screen at t started: the start of the run of consecutive sections
    that share its look. What a scene builds its furniture from (`age = t - sd.scene_start(t)`)."""
    _, sec, _ = section_at(t)
    return look_span(sec[4], t)[0]


def tc(t):
    """Seconds -> MM:SS.mmm show time code."""
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:06.3f}"


# ----------------------------------------------------------------------------
# subtitles
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Cue:
    t: float
    text: str
    group: int
    end: float


@lru_cache(maxsize=1)
def subtitles(path: str | None = None) -> tuple[Cue, ...]:
    """Parse subtitletimecode.txt: 'text  MM:SS:FF' (FF = frames at 60 fps), blank line = new paragraph."""
    p = Path(path) if path else ROOT / "subtitletimecode.txt"
    raw, group = [], 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            group += 1
            continue
        m = re.match(r"^(.*?)\s+(\d{1,2}):(\d{2}):(\d{2})\s*$", line)
        if not m:
            continue
        t = int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 60.0
        raw.append((t, m.group(1).strip(), group))
    out = []
    for k, (t, text, g) in enumerate(raw):
        hold = min(7.0, max(2.6, 1.6 + 0.075 * len(text)))
        nxt = raw[k + 1][0] if k + 1 < len(raw) else 1e9
        out.append(Cue(t, text, g, min(t + hold, nxt)))
    return tuple(out)


def subtitle_at(t):
    """The cue on screen at t (or None) and its age."""
    for c in subtitles():
        if c.t <= t < c.end:
            return c, t - c.t
    return None, 0.0


def said(word, default=None, nth=0):
    """Time code of the voice-over cue matching `word` (case-insensitive): exact text first, then a cue
    starting with it, then one containing it. `nth` picks among several matches."""
    w = word.lower().strip(". ")
    cues = subtitles()
    for test in (lambda c: c.text.lower().strip(". ") == w, lambda c: c.text.lower().startswith(w),
                 lambda c: w in c.text.lower()):
        hits = [c.t for c in cues if test(c)]
        if len(hits) > nth:
            return hits[nth]
    return default


# ----------------------------------------------------------------------------
# audio cues
# ----------------------------------------------------------------------------

class Cues:
    def __init__(self, path: Path | None = None):
        path = path or (DATA / "cues.npz")
        self.ok = path.exists()
        if not self.ok:
            self.dt = 0.01
            self._loud = np.zeros(1, np.float32)
            self._bands = np.zeros((1, 5), np.float32)
            self._spec = np.zeros((1, 48), np.float32)
            self.spec_dt = 0.02
            self.kick_t = self.kick_a = self.onset_t = self.onset_a = np.zeros(0, np.float32)
            self.det_t = np.zeros(0, np.float32)
            self.det_k = np.zeros(0, np.int8)
            self.det_e = np.zeros(0, np.float32)
            self.det_echo = np.zeros(0, bool)
            return
        z = np.load(path)
        self.dt = float(z["dt"])
        self.spec_dt = float(z["spec_dt"])
        self._loud = z["loud"].astype(np.float32) / 255.0
        self._bands = z["bands"].astype(np.float32) / 255.0
        self._spec = z["spec"].astype(np.float32) / 255.0
        self.kick_t, self.kick_a = z["kick_t"], z["kick_a"]
        self.onset_t, self.onset_a = z["onset_t"], z["onset_a"]
        self.det_t, self.det_k, self.det_e, self.det_echo = z["det_t"], z["det_k"], z["det_e"], z["det_echo"]

    def _at(self, arr, t, dt, win=0.0):
        i = int(np.clip(t / dt, 0, len(arr) - 1))
        if win <= 0:
            return arr[i]
        j = int(np.clip((t - win) / dt, 0, i))
        return arr[j: i + 1].mean(0)

    def loud(self, t, win=0.08):
        """Loudness of the music 0..1 (sqrt-compressed), averaged over the last `win` seconds."""
        return float(self._at(self._loud, t, self.dt, win))

    def bands(self, t, win=0.05):
        """5 band envelopes 0..1: sub, bass, mid, high-mid, air."""
        return np.asarray(self._at(self._bands, t, self.dt, win))

    def spec(self, t, win=0.06):
        """48 log-spaced spectrum bins 0..1 (30 Hz .. 16 kHz)."""
        return np.asarray(self._at(self._spec, t, self.spec_dt, win))

    def loud_curve(self, t0, t1, n):
        """n samples of the loudness between t0 and t1 (max-pooled) - for timeline strips."""
        e = np.linspace(t0, t1, n + 1)
        i = np.clip((e / self.dt).astype(int), 0, len(self._loud) - 1)
        return np.array([self._loud[a: max(b, a + 1)].max() for a, b in zip(i[:-1], i[1:])])

    @staticmethod
    def _pulse(times, amps, t, tau):
        if not len(times):
            return 0.0
        i = int(np.searchsorted(times, t, side="right"))
        j = max(0, i - 12)
        a = t - times[j:i]
        return float((amps[j:i] * np.exp(-a / tau)).sum())

    def kick(self, t, tau=0.12):
        """Decaying pulse on every kick / low hit of the music (about 0..1.5)."""
        return self._pulse(self.kick_t, self.kick_a, t, tau)

    def onset(self, t, tau=0.08):
        return self._pulse(self.onset_t, self.onset_a, t, tau)

    def since_kick(self, t):
        i = int(np.searchsorted(self.kick_t, t, side="right"))
        return float(t - self.kick_t[i - 1]) if i else 99.0

    def kicks(self, t0, t1):
        m = (self.kick_t >= t0) & (self.kick_t < t1)
        return self.kick_t[m], self.kick_a[m]


# ----------------------------------------------------------------------------
# detectors
# ----------------------------------------------------------------------------

class Detectors:
    """The three detector streams. `value(key, t)` is what the app will receive over OSC (one float per
    tower, 0..1); hits are its rising edges with their peak as energy."""

    TAU = 0.32          # decay of the pulse a hit leaves on the stream (s)

    def __init__(self, cues: Cues):
        self.t, self.e, self.echo = {}, {}, {}
        for k, key in enumerate(KEYS):
            idx = np.nonzero(cues.det_k == {"L": 0, "C": 1, "R": 2}[key])[0]
            self.t[key] = cues.det_t[idx].astype(np.float64)
            self.e[key] = cues.det_e[idx].astype(np.float64)
            self.echo[key] = cues.det_echo[idx]

    # -- life cycle ----------------------------------------------------------
    @staticmethod
    def power(key, t):
        """0 = dark, (0..1) = standby (revealed, waiting), 1 = on."""
        if t < T_REVEAL:
            return 0.0
        if t < T_ON[key]:
            return 0.35
        return 1.0

    @staticmethod
    def online(key, t):
        return T_ON[key] <= t

    # -- events --------------------------------------------------------------
    def hits(self, key, t0, t1, echoes=True):
        """(times, energies, echo flags) of the hits of one tower in [t0, t1)."""
        tt = self.t[key]
        a, b = np.searchsorted(tt, t0), np.searchsorted(tt, t1)
        t, e, ec = tt[a:b], self.e[key][a:b], self.echo[key][a:b]
        if not echoes:
            m = ~ec
            return t[m], e[m], ec[m]
        return t, e, ec

    def last(self, key, t, echoes=False):
        """(age, energy) of the most recent hit before t; age = 99 if none."""
        tt = self.t[key]
        i = int(np.searchsorted(tt, t, side="right")) - 1
        while i >= 0 and not echoes and self.echo[key][i]:
            i -= 1
        if i < 0:
            return 99.0, 0.0
        return float(t - tt[i]), float(self.e[key][i])

    def count(self, key, t):
        tt = self.t[key]
        i = int(np.searchsorted(tt, t, side="right"))
        return int((~self.echo[key][:i]).sum())

    def total(self, t):
        return sum(self.count(k, t) for k in KEYS)

    # -- the float stream ------------------------------------------------------
    def value(self, key, t):
        """Simulated detector value at t (scalar or array), 0..1: a decaying pulse per hit plus a noise floor."""
        t = np.asarray(t, np.float64)
        tt, ee = self.t[key], self.e[key]
        flat = np.atleast_1d(t).ravel()
        o = np.zeros(flat.shape)
        hi = np.searchsorted(tt, flat, side="right")
        for back in range(1, 9):
            i = hi - back
            ok = i >= 0
            a = np.maximum(flat - tt[np.maximum(i, 0)], 0.0)
            o += np.where(ok & (a < 8 * self.TAU), ee[np.maximum(i, 0)] * np.exp(-a / self.TAU)
                          * (1 - np.exp(-a / 0.004)), 0.0)
        ph = {"L": 0.0, "C": 1.7, "R": 3.1}[key]
        noise = 0.018 + 0.012 * np.sin(flat * 37.0 + ph) * np.sin(flat * 5.3 + 2 * ph) + 0.008 * np.sin(flat * 131.0 + ph)
        return np.clip(o + noise, 0.0, 1.0).reshape(t.shape)


class Context:
    """One per render process: shared data handed to every scene."""

    def __init__(self, towers=None):
        from . import layout
        self.towers = towers or layout.load_towers()
        self.cues = Cues()
        self.det = Detectors(self.cues)
        self.gaps = layout.gaps(self.towers)
        self.bays = layout.bays(self.towers)
        self.slots = layout.bottom_slots(self.towers)             # bottom band with the three scopes
        self.slots_pre = layout.bottom_slots(self.towers, scopes=False)   # ... before the detectors are revealed
        self.sub = layout.sub_rect(self.towers)
        self.focus = layout.focus_point(self.towers)              # centre of the one-centre compositions
        self.focus_bay = layout.focus_bay(self.towers)
        self.cell_r = layout.right_cell(self.towers)
        self.cols = layout.columns(self.towers)                   # usable text columns, left to right

    # the muons through one spectator since the show started (the VO: "more than fifty thousand")
    RATE_YOU = RATE_YOU

    def through_you(self, t):
        return int(self.RATE_YOU * max(0.0, t))
