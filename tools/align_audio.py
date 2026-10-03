"""Where does each moment of a new mix come from in the previous one?

When the music is re-edited (a scene cut, a passage moved, a new mix), the show timeline has to follow.
This compares the onset envelopes of the two files window by window and prints the runs of constant
offset: 'new 09:35.0 - 09:48.0 = old 10:12.3 - 10:25.3' etc. Unmatched stretches are new material.

  python tools/align_audio.py OLD.wav NEW.wav
  python tools/align_audio.py OLD.wav NEW.wav --win 6 --step 0.5 --csv out.csv
"""
from __future__ import annotations

import argparse
import sys
import wave

import numpy as np

HOP, N_FFT, FPS = 441, 2048, 100.0
N_BANDS = 32


def load_mono(path):
    with wave.open(str(path), "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    assert sw == 2 and sr == 44100, f"{path}: expected 16 bit 44.1 kHz"
    return np.frombuffer(raw, np.int16).reshape(-1, ch).astype(np.float32).mean(1) / 32768.0


def features(x):
    """-> (onset envelope, log band energies) at 100 frames a second."""
    win = np.hanning(N_FFT).astype(np.float32)
    n_fr = 1 + (len(x) - N_FFT) // HOP
    freqs = np.fft.rfftfreq(N_FFT, 1 / 44100)
    edges = np.geomspace(40.0, 15000.0, N_BANDS + 1)
    band = np.searchsorted(edges, freqs) - 1
    ok = (band >= 0) & (band < N_BANDS)
    L = np.zeros((n_fr, N_BANDS), np.float32)
    for s in range(0, n_fr, 4000):
        e = min(n_fr, s + 4000)
        idx = np.arange(s, e)[:, None] * HOP + np.arange(N_FFT)[None, :]
        P = np.abs(np.fft.rfft(x[idx] * win[None, :], axis=1)) ** 2
        B = np.zeros((e - s, N_BANDS), np.float32)
        np.add.at(B.T, band[ok], P[:, ok].T)
        L[s:e] = np.log10(B + 1e-7)
    flux = np.maximum(np.diff(L, axis=0, prepend=L[:1]), 0.0).sum(1)
    return flux.astype(np.float64), L


def ncc_all(ref, w):
    """Normalised cross-correlation of the window w against every position of ref."""
    n = len(w)
    w = (w - w.mean()) / (w.std() + 1e-9)
    size = 1 << int(np.ceil(np.log2(len(ref) + n)))
    c = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(w, size)), size)[: len(ref) - n + 1]
    cs = np.concatenate([[0.0], np.cumsum(ref)])
    cs2 = np.concatenate([[0.0], np.cumsum(ref * ref)])
    s1 = cs[n:] - cs[:-n]
    s2 = cs2[n:] - cs2[:-n]
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12))
    return c / (n * sd)


def tc(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def align(old_path, new_path, win=8.0, step=1.0, min_ncc=0.5):
    fo, _ = features(load_mono(old_path))
    fn, _ = features(load_mono(new_path))
    W = int(win * FPS)
    rows = []
    for s in range(0, len(fn) - W, int(step * FPS)):
        w = fn[s: s + W]
        if w.std() < 1e-6:
            rows.append((s / FPS, None, 0.0))
            continue
        c = ncc_all(fo, w)
        k = int(np.argmax(c))
        rows.append((s / FPS, k / FPS, float(c[k])))
    return rows, len(fo) / FPS, len(fn) / FPS


def runs(rows, win, tol=0.03, min_ncc=0.5):
    """Group consecutive windows that map with the same offset (old - new)."""
    out = []
    for t, o, q in rows:
        off = None if (o is None or q < min_ncc) else o - t
        if out and ((off is None and out[-1][2] is None) or
                    (off is not None and out[-1][2] is not None and abs(off - out[-1][2]) <= tol)):
            out[-1][1] = t
            out[-1][3].append(q)
        else:
            out.append([t, t, off, [q]])
    return [(a, b + win, off, float(np.mean(qs))) for a, b, off, qs in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--win", type=float, default=8.0, help="window length, s")
    ap.add_argument("--step", type=float, default=1.0, help="window step, s")
    ap.add_argument("--min-ncc", type=float, default=0.5, help="below this a window is called unmatched")
    ap.add_argument("--csv", default=None, help="also write every window: new_t, old_t, ncc")
    a = ap.parse_args()
    rows, d_old, d_new = align(a.old, a.new, a.win, a.step)
    print(f"old {tc(d_old)}   new {tc(d_new)}   window {a.win:g} s step {a.step:g} s")
    for t0, t1, off, q in runs(rows, a.win, min_ncc=a.min_ncc):
        if off is None:
            print(f"  new {tc(t0)} - {tc(t1)}   no match (ncc {q:.2f})")
        else:
            print(f"  new {tc(t0)} - {tc(t1)}   = old {tc(t0 + off)} - {tc(t1 + off)}   offset {off:+8.2f} s  ncc {q:.2f}")
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("new_t,old_t,ncc\n")
            for t, o, q in rows:
                f.write(f"{t:.2f},{'' if o is None else f'{o:.2f}'},{q:.3f}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
