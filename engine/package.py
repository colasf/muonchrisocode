"""Make a folder of the show that runs on another computer as it is: nothing to install there.

    python engine/package.py                    writes engine/out/MuonBloom
    python engine/package.py E:\\MuonBloom       ... or the folder given (it must not exist, or be an old package)
    python engine/package.py --no-cache         without data/cache (142 MB; the shower worlds are then rebuilt on
                                                the first run, which takes a while)

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
import importlib.metadata as md
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", nargs="?", default=str(ROOT / "engine" / "out" / "MuonBloom"))
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    dst = Path(a.dest).resolve()
    exe = ROOT / "engine" / "build" / "muonengine.exe"
    if not exe.exists():
        sys.exit("engine\\build\\muonengine.exe is missing: run engine\\build.bat first")
    if dst.exists():
        if not (dst / MARK).exists():
            sys.exit(f"{dst} exists and is not a package made by this script: give another folder")
        say(f"replacing {dst}")
        shutil.rmtree(dst)
    ff = fonts()

    (dst / "engine" / "build").mkdir(parents=True)
    (dst / MARK).write_text("made by engine/package.py\n")
    shutil.copy2(exe, dst / "engine" / "build")
    for name in ("worker.py", "detectors.py", "run.bat", "output.json", "detectors.json", "claude_args.txt"):
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
    python(dst / "python", ROOT / "engine" / "out" / "package_cache")

    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    say(f"{dst}: {size / 1e6:.0f} MB. Copy the folder to the other computer and start run.bat")


if __name__ == "__main__":
    main()
