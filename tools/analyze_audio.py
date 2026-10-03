"""Turn the two show stems into the cue data the visuals read (data/cues.npz).

  music stem  -> loudness, 5 band envelopes, 48-band spectrum, kick / onset times
  muon stem   -> simulated detector hits: the stem pans LEFT / CENTRE / RIGHT exactly like the
                 three towers, so every onset becomes a hit on one tower with an energy 0..1
                 (echo repeats of a hit are flagged). The realtime app will get the same thing
                 live from the detectors over OSC.

Run again whenever the audio changes:   python tools/analyze_audio.py
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MUSIC = Path(r"D:\muonchristo\audio\muon bloom Mixed v1 scene 10 edit NO MUONS SOUNDS.wav")
MUON = Path(r"D:\muonchristo\audio\muon bloom Mixed v1 just muons.wav")
N_FFT, HOP = 2048, 441                      # 10 ms hop at 44.1 kHz
DT = 0.01
BANDS = [(20, 120), (120, 400), (400, 1500), (1500, 5000), (5000, 16000)]
N_SPEC = 48
# hits that live in the music stem (scene 2.2 and scene 3 of the sheet), found on its loudness
SCRIPTED = [(133.3, 1, 1.0), (147.35, 0, 0.9), (160.0, 2, 0.9), (170.5, 1, 1.0)]


def load(path):
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    return sr, np.frombuffer(raw, np.int16).reshape(-1, ch).astype(np.float32) / 32768.0


def stft_feats(x, sr, spec=False):
    win = np.hanning(N_FFT).astype(np.float32)
    n_fr = 1 + (len(x) - N_FFT) // HOP
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    bidx = [np.nonzero((freqs >= lo) & (freqs < hi))[0] for lo, hi in BANDS]
    edges = np.geomspace(30.0, 16000.0, N_SPEC + 1)
    sidx = [np.nonzero((freqs >= a) & (freqs < b))[0] for a, b in zip(edges[:-1], edges[1:])]
    E = np.zeros((n_fr, len(BANDS)), np.float32)
    F = np.zeros((n_fr, len(BANDS)), np.float32)
    S48 = np.zeros((n_fr, N_SPEC), np.float32) if spec else None
    prev = None
    for s in range(0, n_fr, 4000):
        e = min(n_fr, s + 4000)
        idx = np.arange(s, e)[:, None] * HOP + np.arange(N_FFT)[None, :]
        S = np.abs(np.fft.rfft(x[idx] * win[None, :], axis=1)).astype(np.float32)
        L = np.log1p(200.0 * S)
        d = np.maximum(L - np.vstack([prev if prev is not None else L[:1], L[:-1]]), 0.0)
        prev = L[-1:]
        for b, ii in enumerate(bidx):
            E[s:e, b] = np.sqrt((S[:, ii] ** 2).sum(1))
            F[s:e, b] = d[:, ii].sum(1)
        if spec:
            for b, ii in enumerate(sidx):
                if len(ii):
                    S48[s:e, b] = np.sqrt((S[:, ii] ** 2).mean(1))
    return E, F, S48


def peaks(x, thr_rel=0.08, min_gap=0.08, win=0.5):
    w = int(win / DT)
    thr = np.convolve(x, np.ones(w) / w, mode="same") * 1.5 + thr_rel * np.percentile(x, 99.5)
    g = int(min_gap / DT)
    out, i, n = [], 1, len(x)
    while i < n - 1:
        if x[i] > thr[i] and x[i] >= x[i - 1] and x[i] > x[i + 1]:
            j = i + int(np.argmax(x[i: i + g]))
            out.append(j)
            i = j + g
        else:
            i += 1
    return np.array(out, int)


def u8(x, ref, span_db=60.0):
    """Level in dB under the reference, mapped to 0..255 (0 = -span_db or less, 255 = reference)."""
    db = 20.0 * np.log10(np.maximum(x, 1e-9) / ref)
    return np.clip((1.0 + db / span_db) * 255.0, 0, 255).astype(np.uint8)


def main():
    out = ROOT / "data" / "cues.npz"
    out.parent.mkdir(exist_ok=True)
    sr, a = load(MUSIC)
    E, F, S48 = stft_feats(a.mean(1), sr, spec=True)
    n = len(E)
    loud = np.sqrt((E.astype(np.float64) ** 2).sum(1))
    kick = peaks(F[:, 0] + F[:, 1], thr_rel=0.12, min_gap=0.14)
    onset = peaks(F.sum(1), thr_rel=0.1, min_gap=0.07)
    ka = (F[:, 0] + F[:, 1])[kick]
    oa = F.sum(1)[onset]
    spec = S48[::2]
    print(f"music: {n} frames ({n * DT:.1f} s), {len(kick)} kicks, {len(onset)} onsets", flush=True)

    sr, a = load(MUON)
    Em, Fm, _ = stft_feats(a.mean(1), sr)
    EL, _, _ = stft_feats(a[:, 0], sr)
    ER, _, _ = stft_feats(a[:, 1], sr)
    pk = peaks(Fm.sum(1))
    amps = np.array([float(Em[i:min(len(Em), i + 12)].sum(1).max()) for i in pk])
    # energy 0..1 on a log scale of two decades under the loudest hits of the stem (its 97th percentile), so
    # that a louder or quieter mix of the muon sounds gives the same spread of energies
    ref = float(np.percentile(amps, 97)) if len(amps) else 1.0
    hits = []
    last = {0: (-9.0, 0.0), 1: (-9.0, 0.0), 2: (-9.0, 0.0)}
    for i, amp in zip(pk, amps):
        j = min(len(Em), i + 12)
        eL, eR = float(EL[i:j].sum()), float(ER[i:j].sum())
        pan = (eR - eL) / (eR + eL + 1e-9)
        k = 0 if pan < -0.3 else 2 if pan > 0.3 else 1
        t = i * DT
        echo = (t - last[k][0] < 1.2) and (amp < 1.35 * last[k][1])
        last[k] = (t, amp)
        e = float(np.clip(np.log10(max(amp * 100.0 / ref, 1e-3)) / 2.0, 0.05, 1.0))   # ref / 100 .. ref -> 0 .. 1
        hits.append((t, k, e, echo))
    hits += [(t, k, e, False) for t, k, e in SCRIPTED]
    hits.sort()
    det_t = np.array([h[0] for h in hits], np.float32)
    det_k = np.array([h[1] for h in hits], np.int8)
    det_e = np.array([h[2] for h in hits], np.float32)
    det_echo = np.array([h[3] for h in hits], bool)
    for k, name in enumerate("LCR"):
        m = det_k == k
        print(f"detector {name}: {int(m.sum())} onsets, {int((m & ~det_echo).sum())} hits", flush=True)

    tmp = out.with_name("cues_tmp.npz")          # written aside, then swapped in: renders may be reading cues.npz
    np.savez_compressed(
        tmp, dt=DT, duration=n * DT,
        loud=u8(loud, np.percentile(loud, 99.9)), bands=u8(E, np.percentile(E, 99.9, axis=0)[None, :]),
        spec=u8(spec, np.percentile(spec, 99.9)), spec_dt=2 * DT,
        kick_t=(kick * DT).astype(np.float32), kick_a=(ka / np.percentile(ka, 95)).astype(np.float32),
        onset_t=(onset * DT).astype(np.float32), onset_a=(oa / np.percentile(oa, 95)).astype(np.float32),
        det_t=det_t, det_k=det_k, det_e=det_e, det_echo=det_echo)
    import os
    os.replace(tmp, out)
    print("saved", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
