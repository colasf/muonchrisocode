"""A pretend Ableton: sends the time of the show to the engine, as TouchDesigner will on site.

  python engine/tools/fake_clock.py --from 180                       plays from 03:00, 60 messages a second
  python engine/tools/fake_clock.py --from 180 --late 12             each message leaves 0 to 12 ms late
  python engine/tools/fake_clock.py --from 180 --stop-at 186 --stop-for 3     a pause of the sender
  python engine/tools/fake_clock.py --from 180 --jump-at 186 --jump-to 420    a locate
  python engine/tools/fake_clock.py --from 180 --quit-at 186         the sender disappears
  python engine/tools/fake_clock.py --from 180 --step 35 --wobble 40 --odd 3     a rough time, as Ableton gives it:
                                                                     in steps of 35 ms, wandering 40 ms either way,
                                                                     and a value 3 s off now and then

OSC over UDP:  /muonbloom/time <seconds>  (one float; a double is taken too), sent all the time, also while
stopped: a time that stands still is how the engine knows that the sound has stopped.
The engine takes it from this machine only, unless it is started with --osc-allow <address of the sender>.
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
    return a + b",d\0\0" + struct.pack(">d", value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--from", dest="start", type=float, default=0.0, help="show time to start at (seconds)")
    ap.add_argument("--rate", type=float, default=60.0, help="messages a second")
    ap.add_argument("--late", type=float, default=0.0, help="each message leaves up to this many ms after its time was read")
    ap.add_argument("--stop-at", type=float, default=None)
    ap.add_argument("--stop-for", type=float, default=3.0)
    ap.add_argument("--jump-at", type=float, default=None)
    ap.add_argument("--jump-to", type=float, default=0.0)
    ap.add_argument("--quit-at", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.0, help="the time sent moves in steps of this many ms")
    ap.add_argument("--wobble", type=float, default=0.0, help="the time sent wanders this many ms either way (a slow wave of 7 s)")
    ap.add_argument("--odd", type=float, default=0.0, help="about every 5 s, a few values are this many seconds off")
    ap.add_argument("--duration", type=float, default=1e9, help="seconds this script runs")
    a = ap.parse_args()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.perf_counter()
    show0, wall0 = a.start, t0                          # the show time is show0 + (wall - wall0) while it runs
    stopped_until, jumped, stopped = None, False, False
    n = 0
    print(f"sending /muonbloom/time to {a.host}:{a.port} from {a.start:.2f} s (Ctrl+C to stop)", flush=True)
    while time.perf_counter() - t0 < a.duration:
        w = time.perf_counter()
        if stopped_until is not None:
            t = show0
            if w >= stopped_until:
                stopped_until, wall0 = None, w
                print(f"{w - t0:7.2f}  playing again from {show0:.2f}", flush=True)
        else:
            t = show0 + (w - wall0)
            if a.stop_at is not None and not stopped and t >= a.stop_at:
                stopped, show0, stopped_until = True, t, w + a.stop_for
                print(f"{w - t0:7.2f}  stopped at {t:.2f} for {a.stop_for} s", flush=True)
            if a.jump_at is not None and not jumped and t >= a.jump_at:
                jumped, show0, wall0, t = True, a.jump_to, w, a.jump_to
                print(f"{w - t0:7.2f}  jumped to {t:.2f}", flush=True)
            if a.quit_at is not None and t >= a.quit_at:
                print(f"{w - t0:7.2f}  gone at {t:.2f}", flush=True)
                return
        if stopped_until is None:                       # a rough sender (the true time stays t)
            sent = t
            if a.wobble > 0:
                sent += a.wobble / 1000.0 * math.sin(2.0 * math.pi * (w - t0) / 7.0)
            if a.odd and (w - t0) % 5.0 > 4.9:
                sent += a.odd
            if a.step > 0:
                sent = math.floor(sent / (a.step / 1000.0)) * (a.step / 1000.0)
            t = sent
        if a.late > 0:
            time.sleep(random.uniform(0.0, a.late) / 1000.0)
        s.sendto(osc("/muonbloom/time", t), (a.host, a.port))
        n += 1
        time.sleep(max(0.0, t0 + n / a.rate - time.perf_counter()))


if __name__ == "__main__":
    main()
