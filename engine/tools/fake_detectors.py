"""Three pretend muon detectors: sends what the engine expects from the real ones, to try the live mode.

  python engine/tools/fake_detectors.py                    random hits, about one a second per detector
  python engine/tools/fake_detectors.py --rate 3 --port 9000 --host 127.0.0.1

OSC over UDP, 100 values a second per detector:   /muon/L  /muon/C  /muon/R   one float 0..1
(a hit is a fast rise and a decay of about 0.3 s, on a small noise floor - the stream of showdata.Detectors).
Start the engine with:  muonengine live --detectors live
"""
from __future__ import annotations

import argparse
import math
import random
import socket
import struct
import time


def osc(addr, value):
    a = addr.encode() + b"\0"
    a += b"\0" * (-len(a) % 4)
    return a + b",f\0\0" + struct.pack(">f", value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--rate", type=float, default=1.0, help="hits per second and per detector")
    a = ap.parse_args()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last = {k: (-99.0, 0.0) for k in "LCR"}             # time and energy of the last hit
    t0 = time.perf_counter()
    n = 0
    print(f"sending /muon/L /muon/C /muon/R to {a.host}:{a.port}  (Ctrl+C to stop)")
    while True:
        t = time.perf_counter() - t0
        for k in "LCR":
            if random.random() < a.rate / 100.0:
                last[k] = (t, random.uniform(0.35, 1.0))
                print(f"{t:8.2f}  hit {k}  {last[k][1]:.2f}")
            age = t - last[k][0]
            v = last[k][1] * math.exp(-age / 0.32) * (1 - math.exp(-age / 0.004)) + 0.018 + 0.01 * math.sin(37 * t)
            s.sendto(osc(f"/muon/{k}", min(max(v, 0.0), 1.0)), (a.host, a.port))
        n += 1
        time.sleep(max(0.0, t0 + n / 100.0 - time.perf_counter()))


if __name__ == "__main__":
    main()
