"""Make a folder of the show that runs on another computer as it is: nothing to install there.

    python engine/package.py                    writes engine/out/MuonBloom
    python engine/package.py E:\\MuonBloom       ... or the folder given (it must not exist, or be an old package)
    python engine/package.py --no-cache         without data/cache (142 MB; the shower worlds are then rebuilt on
                                                the first run, which takes a while)
    python engine/package.py --expires 2026-10-15
                                                a copy tied to this show and to a date (engine/src/lock.h): its
                                                engine is built for it (engine/build_lock), does not start after
                                                that day and only plays the files of the show as they are packaged
                                                here. The settings of the place can still be changed there
                                                (data/towers.json, engine/output.json, engine/detectors.json).

What goes in:

    engine/     muonengine.exe (build it first: engine\\build.bat), the shaders, the workers, run.bat, the OSC
                test tools, and the settings of this machine (output.json, detectors.json) if there are any
    muonbloom/  the scenes, with the Space Mono fonts beside them
    data/       towers, cues, city, figure
    tools/      test_card.py (the TEST CARD of the OUTPUT panel)
    python/     the Python of this machine in its embeddable form (downloaded from python.org) with the numpy
                and the Pillow of this machine: the engine uses it when it is there (src/main.cpp)
    run.bat     starts the show

Needs the network once (python.org and PyPI); what was downloaded is kept in engine/out/package_cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = "muonbloom_package.txt"              # in the folder: it was made here, so it may be replaced
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp")
RUN = """@echo off
rem MUON : BLOOM. Arguments are passed on to the engine, for example
rem     run.bat                                   the show: waits for the time sent by OSC, live detectors
rem     run.bat --from 180 --paused               (see engine\\run.bat for the others)
if "%~1"=="" (
    call "%~dp0engine\\run.bat" --paused --detectors live
) else (
    call "%~dp0engine\\run.bat" %*
)
"""


def say(s):
    print(s, flush=True)


def fonts():
    from muonbloom import engine as E
    out = []
    for bold in (False, True):
        p = getattr(E.font(22, bold), "path", None)
        if not (isinstance(p, str) and "SpaceMono" in p):
            sys.exit("Space Mono was not found on this machine (muonbloom/engine.py: _FONT_DIRS)")
        out.append(Path(p))
    return out


def python(dst, cache):
    v = "%d.%d.%d" % sys.version_info[:3]
    name = f"python-{v}-embed-amd64.zip"
    z = cache / name
    if not z.exists():
        say(f"downloading {name}")
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://www.python.org/ftp/python/{v}/{name}", z)
    with zipfile.ZipFile(z) as f:
        f.extractall(dst)
    site = dst / "Lib" / "site-packages"
    pth = next(dst.glob("python*._pth"))
    pth.write_text(pth.read_text().rstrip() + "\nLib\\site-packages\n")
    want = [f"numpy=={md.version('numpy')}", f"pillow=={md.version('pillow')}"]
    say("installing " + ", ".join(want))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-deps",
                           "--only-binary", ":all:", "--no-compile", "--target", str(site), "--cache-dir", str(cache / "pip"), *want])
    shutil.rmtree(site / "bin", ignore_errors=True)
    subprocess.check_call([str(dst / "python.exe"), "-c", "import numpy, PIL.ImageFont, mmap"])


def lock_hash(dst):
    """SHA-256 of the files of the show in a package: what engine/src/lock.h (intact) computes there."""
    files = []
    for d, exts in (("muonbloom", (".py", ".ttf")), ("muonbloom/scenes", (".py",)), ("engine/shaders", None), ("data", (".npz",))):
        files += [f"{d}/{p.name}" for p in (dst / d).iterdir() if p.is_file() and (exts is None or p.suffix.lower() in exts)]
    files += ["engine/worker.py", "engine/detectors.py", "tools/test_card.py", "subtitletimecode.txt"]
    h = hashlib.sha256()
    for name in sorted(files, key=lambda s: s.encode("utf-8")):
        b = (dst / name).read_bytes() if (dst / name).is_file() else b""
        h.update(name.encode("utf-8") + b"\0" + len(b).to_bytes(8, "little") + b)
    return h.hexdigest()


def build_locked(day, digest):
    """engine/build_lock/muonengine.exe: the engine that only plays that package, until that day."""
    vcvars = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    if not Path(vcvars).exists():
        sys.exit(f"Visual Studio 2022 not found: {vcvars}")
    eng = ROOT / "engine"
    bat = eng / "out" / "build_lock.bat"
    bat.parent.mkdir(exist_ok=True)
    bat.write_text("@echo off\n"
                   f'call "{vcvars}" >nul\n'
                   f'cmake -S "{eng}" -B "{eng / "build_lock"}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMUON_EXPIRES={day} '
                   f'-DMUON_LOCK_HASH={digest} || exit /b 1\n'
                   f'cmake --build "{eng / "build_lock"}" || exit /b 1\n')
    say(f"building the engine of this copy: until {day}, files {digest[:12]}...")
    subprocess.check_call(["cmd", "/c", str(bat)])
    return eng / "build_lock" / "muonengine.exe"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", nargs="?", default=str(ROOT / "engine" / "out" / "MuonBloom"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--expires", default="", help="YYYY-MM-DD: a copy tied to this show, that does not start after that day")
    a = ap.parse_args()
    if a.expires and not re.fullmatch(r"20\d\d-[01]\d-[0-3]\d", a.expires):
        sys.exit("--expires YYYY-MM-DD")
    sys.path.insert(0, str(ROOT))
    dst = Path(a.dest).resolve()
    exe = ROOT / "engine" / "build" / "muonengine.exe"
    if not a.expires and not exe.exists():
        sys.exit("engine\\build\\muonengine.exe is missing: run engine\\build.bat first")
    if dst.exists():
        if not (dst / MARK).exists():
            sys.exit(f"{dst} exists and is not a package made by this script: give another folder")
        say(f"replacing {dst}")
        shutil.rmtree(dst)
    ff = fonts()

    (dst / "engine" / "build").mkdir(parents=True)
    (dst / MARK).write_text("made by engine/package.py\n")
    for name in ("worker.py", "detectors.py", "run.bat", "OSC.txt", "output.json", "detectors.json") + (() if a.expires else ("claude_args.txt",)):
        if (ROOT / "engine" / name).exists():
            shutil.copy2(ROOT / "engine" / name, dst / "engine")
    shutil.copytree(ROOT / "engine" / "shaders", dst / "engine" / "shaders", ignore=SKIP)
    (dst / "engine" / "tools").mkdir()
    for name in ("fake_clock.py", "fake_detectors.py", "osc_send.py"):
        shutil.copy2(ROOT / "engine" / "tools" / name, dst / "engine" / "tools")
    shutil.copytree(ROOT / "muonbloom", dst / "muonbloom", ignore=SKIP)
    for p in ff:
        shutil.copy2(p, dst / "muonbloom")
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp", "cincinnati_osm.json", *(["cache"] if a.no_cache else []))
    shutil.copytree(ROOT / "data", dst / "data", ignore=skip)
    (dst / "tools").mkdir()
    shutil.copy2(ROOT / "tools" / "test_card.py", dst / "tools")
    shutil.copy2(ROOT / "subtitletimecode.txt", dst)
    (dst / "run.bat").write_text(RUN)
    if a.expires:                                   # the engine of this copy, made for the files that are in it now
        exe = build_locked(a.expires.replace("-", ""), lock_hash(dst))
    shutil.copy2(exe, dst / "engine" / "build")
    python(dst / "python", ROOT / "engine" / "out" / "package_cache")

    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    say(f"{dst}: {size / 1e6:.0f} MB. Copy the folder to the other computer and start run.bat")
    if a.expires:
        say(f"this copy does not start after {a.expires} and only plays the files of the show that are in it")


if __name__ == "__main__":
    main()
