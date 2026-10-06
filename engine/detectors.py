"""The detector streams, live: what the three detectors send over OSC, instead of the hits rebuilt from the
muon stem (showdata.Detectors).

The engine receives the OSC messages, stamps every value with the show time of the first frame that can
still show it, and writes them in a block of shared memory; every scene worker reads it here. A scene only
asks `ctx.det` (hits, last, count, total, value, online, power), so this class takes the place of
showdata.Detectors without the scenes knowing:

    live     the hits are the rising edges of the values received (energy = their peak)
    both     the scripted hits of the previews and the live ones together

Two kinds of detectors are understood (the real format is not known yet):

    a stream     the value is sent all the time (tens of times a second): every time the value passes the
                 trigger level on its way up is a hit (the detector is ready for the next one as soon as
                 the value is under that level again)
    events       a message only when a muon is caught: after REARM seconds without a message the detector
                 is ready again, so every message above the trigger level is a hit; its value decays by itself

The trigger level is set in the engine, per detector (--det-level, the keys of the preview window, OSC
/muonbloom/level; kept in engine/detectors.json). It travels in the stream itself, as an entry between the
values, so that every worker changes level on the same value.

What stays scripted in both modes: when each detector is revealed and switched on (showdata.T_ON ...), and
everything a scene knows before it happens. A live hit is only known once it has happened.
"""
from __future__ import annotations

import mmap
import struct

import numpy as np

from muonbloom import showdata as sd

HEADER = 64
ENTRY = np.dtype([("t", "<f8"), ("v", "<f4", (3,)), ("set", "<u4")])     # engine/src/osc.h: DetectorRing::Entry
CONFIG = 0x80000000              # in `set`: the entry is not values but the trigger levels of the three detectors
LEVELS_AT = 24                   # in the header: the trigger levels now (three float32)
KEEP = 20000                     # values kept per detector for value(): a few minutes of a 100 Hz stream


class LiveDetectors(sd.Detectors):
    ON = 0.10                    # trigger level until the engine says another: a hit starts when the value rises above it ...
    RELEASE = 1.0                # ... and the detector is ready for the next one when it is under this part of it: every
                                 # time the value passes the level is a hit (user, 2026-10-06; it was 0.6 before)
    GAP = 0.0                    # seconds: no second hit sooner than this (0: none is held back; it was 0.05)
    PEAK = 0.08                  # seconds after its start during which a hit can still gain energy
    REARM = 0.25                 # seconds without a message after which a detector is ready again (events)
    DECAY = 0.9                  # seconds the value of a detector that went silent takes to come down

    def __init__(self, cues, shm, mode="live"):
        super().__init__(cues)
        self.mode = mode
        self._script = ({k: v.copy() for k, v in self.t.items()}, {k: v.copy() for k, v in self.e.items()},
                        {k: v.copy() for k, v in self.echo.items()})
        head = mmap.mmap(-1, HEADER, tagname=shm)
        self._cap, = struct.unpack_from("<I", head, 16)
        head.close()
        if not self._cap:
            raise RuntimeError(f"the shared memory of the detectors ({shm}) is not there")
        self._mm = mmap.mmap(-1, HEADER + self._cap * ENTRY.itemsize, tagname=shm)
        self._ent = np.frombuffer(self._mm, ENTRY, self._cap, HEADER)
        self._epoch = -1
        self._reset()

    def _reset(self):
        self._seen = 0
        lv = struct.unpack_from("<3f", self._mm, LEVELS_AT)     # (a worker started late: the levels of now)
        self._on = [v if 0.0 < v <= 1.0 else self.ON for v in lv]
        self._ht = {k: [] for k in sd.KEYS}
        self._he = {k: [] for k in sd.KEYS}
        self._armed = {k: True for k in sd.KEYS}
        self._st = {k: [] for k in sd.KEYS}         # the samples of each detector: times ...
        self._sv = {k: [] for k in sd.KEYS}         # ... and values
        self._arr = {}                              # the same as arrays, made when value() needs them
        self._publish()

    def _publish(self):
        st, se, sc = self._script
        for key in sd.KEYS:
            t, e = np.asarray(self._ht[key], np.float64), np.asarray(self._he[key], np.float64)
            ec = np.zeros(len(t), bool)
            if self.mode == "both":
                t, e, ec = np.r_[st[key], t], np.r_[se[key], e], np.r_[sc[key], ec]
                o = np.argsort(t, kind="stable")
                t, e, ec = t[o], e[o], ec[o]
            self.t[key], self.e[key], self.echo[key] = t, e, ec

    def poll(self):
        """Take in what has arrived since the last frame. Called by the worker before every frame."""
        count, = struct.unpack_from("<Q", self._mm, 8)
        epoch, = struct.unpack_from("<I", self._mm, 20)
        if epoch != self._epoch or count < self._seen:      # the clock jumped: the engine started the stream again
            self._epoch = epoch
            self._reset()
        if count == self._seen:
            return
        first = max(self._seen, count - self._cap)
        idx = np.arange(first, count) % self._cap
        t, v, said = self._ent["t"][idx].copy(), self._ent["v"][idx].copy(), self._ent["set"][idx].copy()
        self._seen = count
        changed = False
        start = 0
        for c in np.flatnonzero(said & CONFIG).tolist() + [len(t)]:     # the levels change between two values
            if c > start:
                changed |= self._take(t[start:c], v[start:c], said[start:c])
            if c < len(t):
                self._on = [float(x) if 0.0 < x <= 1.0 else self.ON for x in v[c]]
            start = c + 1
        if changed:
            self._publish()

    def _take(self, t, v, said):
        """The values of one stretch of the stream, under the trigger levels of that stretch."""
        changed = False
        for k, key in enumerate(sd.KEYS):
            on, off = self._on[k], self._on[k] * self.RELEASE
            mine = (said >> k) & 1 == 1
            if not mine.any():
                continue
            ht, he, st, sv = self._ht[key], self._he[key], self._st[key], self._sv[key]
            armed = self._armed[key]
            for ti, vi in zip(t[mine].tolist(), v[mine, k].tolist()):
                if vi != vi:                        # NaN: not a value
                    continue
                if st and ti - st[-1] > self.REARM:
                    # it said nothing for a while: a detector that only speaks when it is hit. Its last value
                    # came down by itself, and it is ready again
                    last_t, last_v = st[-1], sv[-1]
                    if last_v > 0.0:
                        t1 = last_t + self.REARM
                        t2 = min(t1 + self.DECAY, ti - 1e-3)
                        st.append(t1)
                        sv.append(last_v)
                        if t2 > t1:
                            st.append(t2)
                            sv.append(last_v * max(0.0, 1.0 - (t2 - t1) / self.DECAY))
                    armed = True
                if armed:
                    if vi >= on and (not ht or ti - ht[-1] >= self.GAP):
                        ht.append(ti)
                        he.append(min(vi, 1.0))
                        armed = False
                        changed = True
                else:
                    if ht and vi > he[-1] and ti - ht[-1] < self.PEAK:
                        he[-1] = min(vi, 1.0)
                        changed = True
                    if vi < off:
                        armed = True
                st.append(ti)
                sv.append(vi)
            self._armed[key] = armed
            if len(st) > 2 * KEEP:
                del st[:-KEEP], sv[:-KEEP]
            self._arr.pop(key, None)
        return changed

    def value(self, key, t):
        t = np.asarray(t, np.float64)
        st = self._st[key]
        if st:
            a = self._arr.get(key)
            if a is None:                           # + the way down of a detector that has gone silent
                last, lv = st[-1], self._sv[key][-1]
                a = self._arr[key] = (np.r_[st, last + self.REARM, last + self.REARM + self.DECAY],
                                      np.r_[self._sv[key], lv, 0.0])
            live = np.interp(t, a[0], a[1], left=0.0)
        else:
            live = np.zeros(t.shape)
        if self.mode == "both":
            live = np.maximum(live, super().value(key, t))
        return np.clip(live, 0.0, 1.0)
