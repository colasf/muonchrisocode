"""Muon Bloom - preview renderer for the wall (2978 x 1400).

Examples
  python preview.py still 2:16 2:29.5 170.9          full-res PNG stills at these show times
  python preview.py still 2:52 --look bloom           force a look at a time (to audition it elsewhere)
  python preview.py board                             every still of the storyboard + contact sheets
  python preview.py board --only 3 4                  only the stills of scenes 3 and 4
  python preview.py wall 2:52 5:40                    stills mocked up on the photo of the wall
  python preview.py towers 0:26 2:53 9:22             the same frames with three tower placements, side by side
  python preview.py video 2:10 3:00 --scale 0.5       H.264 preview of a time range, with the audio
  python preview.py video --scale 0.5 --workers 24    the whole show (00:00 - 13:54), with the audio

Time codes are show time (the audio files start at 00:00): M:SS, M:SS.cc or seconds.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "previews"
sys.path.insert(0, str(ROOT))

from muonbloom import layout as L  # noqa: E402
from muonbloom import showdata as sd  # noqa: E402
from muonbloom.engine import font  # noqa: E402

WALL_PHOTO = Path(r"D:\muonchristo\Interactive_Stage_1_clean_v5 (1) (1).png")
WALL_ORIGIN = (11, 272)                  # where the 2978 x 1400 output sits in the 3000 x 1688 stage photo
MUSIC = Path(r"D:\muonchristo\audio\muonbloom V7 no muon sounds.wav")
MUON = Path(r"D:\muonchristo\audio\v7 just muon sounds.wav")


def parse_time(s):
    s = str(s)
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def name_for(t, look=None):
    _, sec, _ = sd.section_at(t)
    m = int(t // 60)
    return f"{m:02d}m{t - 60 * m:05.2f}s_{sec[0]}_{look or sec[4]}"


def _even(v):
    return int(v) // 2 * 2


_show = None


def _init():
    global _show
    from muonbloom.show import Show
    _show = Show()


def _render(job):
    """One video frame. An exception must not lose a long render: it goes back to the parent, which holds the
    previous frame and reports the time code."""
    t, W, H, look = job
    try:
        return _show.render(t, W, H, look=look), None
    except Exception:
        return None, traceback.format_exc()


def _save(img, path):
    """Save a PNG; if the file is locked (open in a viewer on Windows), fall back to name_b, name_c, ..."""
    p = Path(path)
    for suffix in ("", "_b", "_c", "_d", "_e", "_f"):
        q = p.with_name(p.stem + suffix + p.suffix)
        try:
            Image.fromarray(img).save(q)
            return str(q)
        except OSError:
            continue
    raise OSError(f"cannot write {path}")


def _render_to_file(job):
    t, W, H, look, path = job
    t0 = time.time()
    img = _show.render(t, W, H, look=look)
    return _save(img, path), time.time() - t0


def render_stills(times, scale=1.0, look=None, workers=1, folder="stills", tag="", numbers=None):
    W, H = _even(L.W * scale), _even(L.H * scale)
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    pre = [f"{n:03d}_" for n in numbers] if numbers else [""] * len(times)
    jobs = [(t, W, H, look, str(d / f"{q}{name_for(t, look)}{tag}.png")) for t, q in zip(times, pre)]
    out = []
    if workers > 1 and len(jobs) > 1:
        with Pool(min(workers, len(jobs)), initializer=_init) as pool:
            for path, dt in pool.imap(_render_to_file, jobs):
                print(f"{path}  ({dt:.1f}s)", flush=True)
                out.append(path)
    else:
        _init()
        for job in jobs:
            path, dt = _render_to_file(job)
            print(f"{path}  ({dt:.1f}s)", flush=True)
            out.append(path)
    return out


# ----------------------------------------------------------------------------
# storyboard
# ----------------------------------------------------------------------------

def load_board():
    """storyboard.txt:  '[*] M:SS[.cc] | caption'   (# = comment, blank lines ignored, * = also on the overview)"""
    p = ROOT / "storyboard.txt"
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.startswith("*")
        tcode, _, cap = line.lstrip("* ").partition("|")
        rows.append((parse_time(tcode.strip()), cap.strip(), key, len(rows) + 1))     # (time, caption, key, number)
    return rows


SCENE_NAMES = {1: "intro", 2: "detector", 3: "detectors_activated", 4: "data_on", 5: "cosmic_groove", 6: "dance",
               7: "glitch", 8: "outlast", 9: "rise", 10: "disintegrate", 11: "outro", 12: "credits"}


def _major(t):
    return int(sd.section_at(t)[1][0].split(".")[0])


def _sheet(chunk, cols, tw, first_index, title=None):
    th = int(tw * L.H / L.W)
    cap_h, pad = 92, 14
    head = 54 if title else 0
    f0, f1, f2 = font(30, True), font(22, True), font(18)
    n_rows = (len(chunk) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, head + n_rows * (th + cap_h) + (n_rows + 1) * pad),
                      (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    if title:
        dr.text((pad + 2, 14), title, font=f0, fill=(255, 255, 255))
    for k, ((t, cap, _, num), path) in enumerate(chunk):
        x = pad + (k % cols) * (tw + pad)
        y = head + pad + (k // cols) * (th + cap_h + pad)
        sheet.paste(Image.open(path).convert("RGB").resize((tw, th), Image.LANCZOS), (x, y))
        _, sec, _ = sd.section_at(t)
        m = int(t // 60)
        dr.text((x + 2, y + th + 6), f"{num:03d}  {m:02d}:{t - 60 * m:05.2f}   {sec[0]} {sec[1]}", font=f1,
                fill=(255, 214, 0))
        cue, _ = sd.subtitle_at(t)
        vo = f'VO: "{cue.text}"' if cue else ""
        dr.text((x + 2, y + th + 36), cap[:118], font=f2, fill=(230, 230, 230))
        dr.text((x + 2, y + th + 62), vo[:118], font=f2, fill=(150, 150, 150))
    return sheet


def contact_sheets(rows, paths, cols=3, tw=1280, name="board"):
    """Labelled contact sheets, one set per scene of the sheet: number, time code, section, caption, voice-over."""
    d = OUT / "sheets"
    d.mkdir(parents=True, exist_ok=True)
    per = cols * 4
    sheets = []
    items = list(zip(rows, paths))
    majors = sorted({_major(r[0]) for r in rows})
    for mj in majors:
        grp = [(i, it) for i, it in enumerate(items) if _major(it[0][0]) == mj]
        parts = [grp[k: k + per] for k in range(0, len(grp), per)]
        for pi, part in enumerate(parts):
            t0, t1 = part[0][1][0][0], part[-1][1][0][0]
            title = (f"MUON : BLOOM   //   SCENE {mj:02d}  {SCENE_NAMES.get(mj, '').upper().replace('_', ' ')}"
                     f"   //   {int(t0 // 60):02d}:{t0 % 60:04.1f} - {int(t1 // 60):02d}:{t1 % 60:04.1f}"
                     + (f"   //   {pi + 1}/{len(parts)}" if len(parts) > 1 else ""))
            sheet = _sheet([it for _, it in part], cols, tw, part[0][0] + 1, title)
            suffix = f"_{chr(97 + pi)}" if len(parts) > 1 else ""
            p = d / f"{name}_scene{mj:02d}_{SCENE_NAMES.get(mj, 'x')}{suffix}.jpg"
            sheet.save(p, quality=90)
            sheets.append(p)
            print(p, flush=True)
    return sheets


def overview_sheet(rows, paths, cols=6, tw=640, name="overview"):
    """The whole show at a glance: the stills marked with * in storyboard.txt."""
    d = OUT / "sheets"
    d.mkdir(parents=True, exist_ok=True)
    keys = [(r, p) for r, p in zip(rows, paths) if r[2]]
    if not keys:
        return None
    th = int(tw * L.H / L.W)
    cap_h, pad, head = 34, 10, 58
    f0, f1 = font(30, True), font(17, True)
    n_rows = (len(keys) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, head + n_rows * (th + cap_h + pad) + pad), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad + 2, 14), "MUON : BLOOM   //   THE WHOLE SHOW AT A GLANCE   //   2978 x 1400   //   13:22 + CREDITS",
            font=f0, fill=(255, 255, 255))
    for k, ((t, cap, _, num), path) in enumerate(keys):
        x = pad + (k % cols) * (tw + pad)
        y = head + (k // cols) * (th + cap_h + pad)
        sheet.paste(Image.open(path).convert("RGB").resize((tw, th), Image.LANCZOS), (x, y))
        _, sec, _ = sd.section_at(t)
        m = int(t // 60)
        dr.text((x + 2, y + th + 6), f"{num:03d}  {m:02d}:{t - 60 * m:04.1f}  {sec[0]} {sec[1]}"[:56], font=f1,
                fill=(255, 214, 0))
    p = d / f"{name}.jpg"
    sheet.save(p, quality=90)
    print(p, flush=True)
    return p


def placement_sheets(times, files, workers, tw=1280):
    """The same frames with several tower placements, side by side: one row per time, one column per file."""
    d = OUT / "sheets"
    d.mkdir(parents=True, exist_ok=True)
    cols = []
    for fpath in files:
        name = Path(fpath).stem
        os.environ["MUONBLOOM_TOWERS"] = str(Path(fpath).resolve())
        cols.append((name, render_stills(times, 1.0, None, workers, folder="placements", tag=f"_{name}")))
    th = int(tw * L.H / L.W)
    pad, head = 14, 40
    f1 = font(22, True)
    per = 4
    out = []
    for s0 in range(0, len(times), per):
        ts = times[s0: s0 + per]
        sheet = Image.new("RGB", (len(cols) * tw + (len(cols) + 1) * pad, len(ts) * (th + head) + pad), (24, 24, 24))
        dr = ImageDraw.Draw(sheet)
        for r, t in enumerate(ts):
            _, sec, _ = sd.section_at(t)
            for c, (name, paths) in enumerate(cols):
                x = pad + c * (tw + pad)
                y = pad + r * (th + head)
                dr.text((x + 2, y + 6), f"{int(t // 60):02d}:{t % 60:05.2f}  {sec[0]} {sec[1]}   //   {name}", font=f1,
                        fill=(255, 214, 0))
                im = Image.open(paths[s0 + r]).convert("RGB").resize((tw, th), Image.LANCZOS)
                sheet.paste(im, (x, y + head - 4))
        q = d / f"towers_{s0 // per + 1:02d}.jpg"
        sheet.save(q, quality=90)
        print(q, flush=True)
        out.append(q)
    return out


# ----------------------------------------------------------------------------
# wall mock-up
# ----------------------------------------------------------------------------

def wall_mock(paths, gain=2.3, ambient=0.07):
    """Projected light on the brick: photo darkened to night, plus light x brick albedo."""
    d = OUT / "wall"
    d.mkdir(parents=True, exist_ok=True)
    photo = np.asarray(Image.open(WALL_PHOTO).convert("RGB"), np.float32) / 255.0
    lin = photo ** 2.2
    night = lin * ambient * np.array([0.75, 0.85, 1.15], np.float32)       # dusk: darker, a little blue
    albedo = np.clip(lin / max(float(np.percentile(lin, 97)), 1e-3), 0.0, 1.0)
    x0, y0 = WALL_ORIGIN
    out = []
    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        if img.shape[1] != L.W:
            img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((L.W, L.H), Image.LANCZOS),
                             np.float32) / 255.0
        light = (img ** 2.2) * gain
        comp = night.copy()
        h = min(L.H, comp.shape[0] - y0)
        reg = comp[y0: y0 + h, x0: x0 + L.W]
        reg += light[:h] * (0.18 + 0.82 * albedo[y0: y0 + h, x0: x0 + L.W])
        res = np.clip(comp, 0, 1) ** (1 / 2.2)
        q = d / (Path(p).stem + "_wall.jpg")
        Image.fromarray((res * 255).astype(np.uint8)).save(q, quality=90)
        print(q, flush=True)
        out.append(q)
    return out


# ----------------------------------------------------------------------------
# video
# ----------------------------------------------------------------------------

def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def render_video(t0, t1, scale, fps, workers, look=None, name=None):
    W, H = _even(L.W * scale), _even(L.H * scale)
    n = int(round((t1 - t0) * fps))
    d = OUT / "video"
    d.mkdir(parents=True, exist_ok=True)
    stem = name or f"{name_for(t0, look)}_to_{int(t1 // 60):02d}m{t1 % 60:05.2f}s"
    mp4 = d / f"{stem}_{W}x{H}_{fps}fps.mp4"
    # the frames are full-range RGB: convert with the BT.709 matrix and tag the stream, so the red plays as drawn
    graph = ("[0:v]scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p,"
             "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709[v];"
             "[1:a][2:a]amix=inputs=2:normalize=0[a]")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", str(MUSIC),
           "-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", str(MUON),
           "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-c:a", "aac", "-b:a", "256k",
           "-movflags", "+faststart", "-shortest", str(mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    jobs = [(t0 + i / fps, W, H, look) for i in range(n)]
    ts = time.time()
    last = np.zeros((H, W, 3), np.uint8)
    failed = []
    with Pool(workers, initializer=_init) as pool:
        for i, (img, err) in enumerate(pool.imap(_render, jobs, chunksize=2)):
            if err:                             # hold the previous frame, keep going
                failed.append(jobs[i][0])
                print(f"  ERROR at {sd.tc(jobs[i][0])} (frame {i + 1}), previous frame held:\n{err}", flush=True)
                img = last
            last = img
            try:
                proc.stdin.write(img.tobytes())
            except OSError:
                sys.exit(f"ffmpeg stopped (exit code {proc.poll()}) - is {mp4.name} open in a player?")
            if i % max(1, n // 20) == 0 and i:
                el = time.time() - ts
                print(f"  frame {i + 1}/{n}  {el:6.1f}s  {(i + 1) / el:5.1f} fps  about {el * (n - i - 1) / (i + 1) / 60:.1f} "
                      f"min left", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"{n} frames in {time.time() - ts:.1f}s -> {mp4}", flush=True)
    if failed:
        print(f"{len(failed)} frames failed and show the previous frame: " + " ".join(sd.tc(t) for t in failed[:40])
              + (" ..." if len(failed) > 40 else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["still", "board", "wall", "video", "towers"])
    ap.add_argument("times", nargs="*", help="show times (M:SS[.cc] or seconds)")
    ap.add_argument("--look", default=None, help="force a look instead of the one scheduled at that time")
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = 2978x1400")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--only", nargs="*", default=None, help="board: only scenes whose sheet code starts with these")
    ap.add_argument("--every", type=int, default=1, help="board: keep one still out of N (quick checks)")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 4)))
    ap.add_argument("--towers", default=None, help="tower placement JSON (default data/towers.json, else placeholder)")
    ap.add_argument("--tag", default="", help="suffix added to the still file names")
    a = ap.parse_args()
    if a.towers:
        os.environ["MUONBLOOM_TOWERS"] = str(Path(a.towers).resolve())
    if a.cmd == "still":
        render_stills([parse_time(x) for x in a.times], a.scale, a.look, a.workers, tag=a.tag)
    elif a.cmd == "wall":
        paths = render_stills([parse_time(x) for x in a.times], 1.0, a.look, a.workers)
        wall_mock(paths)
    elif a.cmd == "board":
        rows = load_board()
        if a.only:
            rows = [r for r in rows if any(sd.section_at(r[0])[1][0].split(".")[0] == o for o in a.only)]
        rows = rows[:: max(1, a.every)]
        paths = render_stills([r[0] for r in rows], a.scale, None, a.workers, folder="board" + a.tag, tag=a.tag,
                              numbers=[r[3] for r in rows])
        contact_sheets(rows, paths, name="board" + a.tag)
        if not a.only and a.every == 1:
            overview_sheet(rows, paths, name="overview" + a.tag)
    elif a.cmd == "towers":
        files = [ROOT / "data" / n for n in ("towers.json", "towers_alt_example.json", "towers_alt2_example.json")]
        placement_sheets([parse_time(x) for x in a.times], files, a.workers)
    elif a.cmd == "video":
        if a.times:
            render_video(parse_time(a.times[0]), parse_time(a.times[1]), a.scale, a.fps, a.workers, a.look)
        else:                                   # no time range = the whole show, credits included
            render_video(0.0, sd.TRACK_END, a.scale, a.fps, a.workers, a.look, name="muonbloom_show")


if __name__ == "__main__":
    main()
