# Engine review, 2026-10-02

Two independent reviewers read the engine as it stood at 13:14 on 2026-10-02 (uncommitted, branch
`engine`). They changed nothing in the repository. One covered the rendering path, one the live
player. The look session spot-checked the findings marked (checked) against the source; the rest is
relayed as reported. CONFIRMED = demonstrated by a run or an unambiguous code path; PLAUSIBLE = read
only, behaviour not observed.

Their test scripts are in
`C:\Users\xmg\AppData\Local\Temp\claude\D--muonchristo-claude\66fe6d4f-4163-4d04-9a56-a4ecb2c518c8\scratchpad\review_render`
and `...\review_live` (a session folder: copy what you want to keep).

## Status after the fixes (2026-10-02, evening)

The look session fixed the findings below in the engine worktree, at the user's request, and added a
time bar with a play / pause button to the preview window. Nothing is committed. "Tested" = observed
on the running engine or by a run; "not tested" = written and read, the situation was not provoked.

| | Finding | What was done | Tested |
|---|---|---|---|
| A1 | clock pinned to a sound that stopped | a position that stands still is not followed; after 0.4 s the sound is given up, the show goes on by the machine's timer and the output is opened again every 5 s; XAudio2's critical-error callback does the same | **not tested** (needs a sound output to disappear) |
| A2 | OSC open to the network, NaN seek | transport orders only from this machine unless `--osc-allow`; non-finite numbers dropped in the parser and in `seek`; detectors only at `<--det-prefix>/L C R` | tested (NaN, another address, valid orders) |
| A3 | silent failures, no restart | `Present` checked, device removed = exit 3; renderer errors printed; start-up failure = exit 2; `run.bat` restarts where it was; `engine/out/engine.log`; machine kept awake | restart tested with a kill; device removal **not tested** |
| A4 | reload takes over with a broken scene | the worker reports a failed warm-up, the take-over is refused, the title says why | tested |
| A5 | file change during a reload lost | a change restarts the reload | tested (two changes in a row) |
| A6 | starts without sound, silently | missing sound files = exit 2 with the paths; no output = loud warning + retries; root and shaders found from the exe | missing file tested |
| A7 | glyphs of dropped frames lost | frames that arrive after a flush are kept until the renderer has taken their glyphs | seeks ran without the "letters could not be drawn" message; no dedicated test |
| A8 | picture stops in Windows' modal loops | Alt / F10 menu swallowed; a timer keeps the loop turning during drag, resize, menu; QuickEdit off | **not tested by hand** |
| A9 | live detectors | detectors that only speak when hit are understood (silence re-arms); history of 1 M entries (a whole show); new workers read it at start; NaN refused | tested with a simulated stream |
| A10 | hung worker never noticed | a worker that does not answer for 60 s (240 s during a warm-up) is stopped; the set is started again; a reload that never says hello is given up after 90 s | **not tested** |
| A11 | frame times rounded to 6 decimals | sent with 17 significant digits: Python gets exactly k / 60 | by construction |
| A12 | OSC out-of-bounds read | bound check | malformed packets sent, no effect |
| notes | | `[` `]` value shown in the title; a tower placement with no room for the picture is not saved; `towers.json` written beside and moved | tower tool **not tested by hand** |
| B1 | line spacing ignored | lines with a spacing over 0.5 px are drawn splat by splat, as the reference | 00:18 to 00:33: from 6 - 7 % of pixels over 2 levels to under 0.04 % |
| B2 | backwards boxes drawn | emptied before upload (light ops, tag boxes, inverted rects) | synthetic blob: 0.2 levels (was 163 to 255) |
| B3 | `compare.py` cannot fail | compares with a finely sampled replay, limits (mean 0.02, 0.05 % over 2 levels, 0.005 % over 8), exit 1; probes now include 00:25, 00:30, 00:41, 13:50.6 | 25 probes + 14 sweep frames pass, worst mean 0.004 |
| B4 | flags and missing glyphs unread in live | counted, written to the console and the log once a second | not provoked |
| B5 | outermost pixel ring | splats outside [0, W-1) x [0, H-1) left out of the integral, as the reference | synthetic blob: single pixels at a crossing still differ (up to 12 levels; the reference itself depends on its sampling there). Discs on the ring: not done |
| B6 | discs over 30 px | edge blurred along the radius | credits 13:50.6: from 59 levels to 6 on 0.003 % |
| B7 | corrupt glyph record crashes | size checked before advancing | corrupt blob: no crash |
| B8 | lines wider than 153 px | the shader clamps its passes like the CPU | synthetic 120 px line: identical |
| B9 | `dim` on text rounds | **not done** (at most 2 levels; the reference truncates, a blend cannot) | |
| B10 | bench waits for ever without a device | the waits give up when the device is gone or after 5 s | not provoked |

Checks after the fixes: `check_drawlist.py --sweep 7` 119 frames, 0 to check; `compare.py --probes`
and `--sweep 60` all pass; `tools/check_frames.py` 18 looks identical; `bench` of the whole show:
50 039 frames, none dropped, 1 late (scene time median 6.4 ms, 95 % 20.7 ms, worst 62 ms; GPU
median 2.3 to 4.1 ms).

## A. Live player

Scope: `src/main.cpp`, `live.cpp`, `pool.*`, `player.h`, `audio.h`, `osc.h`, `worker.py`,
`detectors.py`, tools. The exe was not run; `worker.py` and `detectors.py` were driven standalone.

1. **The show clock is pinned to a sound that has stopped advancing.** `audio.h:86-98`, `192-195`,
   `live.cpp:548`. PLAUSIBLE (code path unambiguous (checked), XAudio2 behaviour on device loss not
   tested). If the audio endpoint disappears while its buffer is still queued, `position()` keeps
   returning true with a frozen time and `follow()` hard-sets the clock back each time it gets 80 ms
   ahead: the picture replays the same few frames forever, never reaches the end, never loops.
   Fix: when the reported position has not moved for about 100 ms, stop following and run on the
   machine timer; register the XAudio2 critical-error callback and reopen the device.
2. **Anyone on the network can pause, seek or reload the show; a NaN seek kills the clock.**
   `live.cpp:506-509`, `406-411`, `osc.h:30`. CONFIRMED (checked). The socket is bound to all
   interfaces with no sender filter; `/muonbloom/pause` leaves an unattended show paused for good.
   `/muonbloom/seek NaN` passes `std::clamp`, so `clock.set(NaN)`: the picture sits on frame 0 and
   `t >= showEnd` is never true. With `--detectors live`, any address whose last part is L, C, R,
   1, 2, 3, left ... counts as a detector, whatever its prefix. Fix: reject non-finite arguments,
   accept transport orders only from an allowed address (or behind a flag), require the `/muon`
   prefix.
3. **Failures are silent and nothing restarts the engine.** `live.cpp:129`, `player.h:82`,
   `live.cpp:429`, `610`, `run.bat:8`. CONFIRMED for the exit code and the unprinted error (checked:
   `Present()` is unchecked), PLAUSIBLE for device loss. After a Direct3D device removal the engine
   keeps running with a frozen or black picture; `Player::error` is never printed in live; "a Python
   worker stopped at start-up" returns exit code 0; `run.bat` has no restart loop and no log file;
   nothing keeps the display awake (`SetThreadExecutionState`). Fix: exit non-zero on device removed
   and on start-up failure, run under a loop that restarts at the right time code, log to a file,
   keep the display awake.
4. **A reload takes over even when the scene being played does not import or raises.**
   `worker.py:99-107`, `player.h:120-133`. CONFIRMED (checked). Scene modules are imported lazily
   inside `Show.render`; the warm-up catches the exception and still answers `W`, so the new workers
   replace the good ones and every frame of that scene fails (picture frozen on the last good
   frame). The safety only holds for modules imported at worker start. Fix: the worker reports a
   failed warm-up and `reloadStep` refuses the take-over.
5. **A file change that arrives during a reload is lost.** `live.cpp:522`. CONFIRMED by reading.
   `watch.poll()` returns true once; if a reload is in progress nothing happens. A second Enter in
   the tower tool during a reload leaves the scenes on the previous placement until R or another
   change. The window is long in dance / glitch: a new placement rebuilds the shower world,
   14.5 to 17 s measured for the first dance frame. Fix: keep the request pending, and restart a
   reload that is under way.
6. **The show can start without sound and say nothing.** `main.cpp:267-275`, `live.cpp:371`,
   `audio.h:60`, `CMakeLists.txt:28-29`. CONFIRMED by reading. If the stems are not at
   `<root>\..\audio` the list is empty, no message, the engine plays mute on the machine timer. The
   repository root and the shader folder are compiled in as absolute paths of the development
   machine. Fix: make missing sound an error unless `--no-audio`; print the files and root in use.
7. **Glyphs carried by a frame dropped after a clock jump are lost.** `pool.cpp:148-172`,
   `worker.py:93`. CONFIRMED (engine side by reading, worker side by run: first frame at t = 100
   carries 188 glyph definitions, the same frame again carries 0). A seek or an audio hard-resync
   within about 100 ms of a new character appearing leaves that character at that size missing until
   a reload. Fix: ingest orphaned frames too, or tell the workers to forget what they sent on a
   flush.
8. **Picture and Spout stop while Windows runs a modal loop on the window thread.**
   `live.cpp:43-57`, `88-97`. PLAUSIBLE, not tried. Alt or F10 in the preview window, holding or
   dragging the title bar, resizing; a click in the console (QuickEdit) blocks the next print.
   Fix: swallow `WM_SYSCOMMAND` / `SC_KEYMENU`, switch QuickEdit off, or draw on another thread.
9. **Live detectors: limits that depend on the still unknown format.** `engine/detectors.py`.
   CONFIRMED by run. Lines 83-94: a detector that sends one message per hit and nothing in between
   gives a single hit, ever (4 messages of 1.0 gave 1 hit); same for a value that stays above 0.06.
   Lines 75 and 27: workers started by a reload or respawn only get the last 65 536 entries (the
   counter on screen went from 300 to 218 at 300 entries a second); `value()` keeps about 67 s.
   A NaN value comes out of `value()` as NaN. Fine: types and shapes match `showdata.Detectors`,
   values outside 0..1 are clipped, the 6 workers build the same history, hit lists reset on seek
   and loop. Known: rise, disintegrate, galaxy and the sphere tracks read their hits once at
   construction, so live hits never reach them.
10. **A worker that hangs is never noticed.** `pool.cpp:173-177`, `live.cpp:539`, `player.h:119`.
    PLAUSIBLE. Only a dead worker triggers a respawn; if a scene hangs at a given time, all six stop
    one after the other and the picture freezes with no message. A new set that never says hello
    keeps `reloading()` true, which also blocks the respawn of dead workers. Fix: a time limit per
    order, and on reloads.
11. **Frames are not computed at exactly k / 60.** `pool.cpp:218`. CONFIRMED by run (checked: the
    time is sent as `%.6f`). `int(t * 30)`, the animation step used by sphere, rise, flood,
    disintegrate, outro and muon, is one lower on 8 340 of 50 040 frames: those steps last 3, 1, 2
    frames instead of 2, 2, 2, and differ from the previews. Fix: send the frame index, or 9
    decimals.
12. **Out-of-bounds read on a crafted OSC packet.** `osc.h:100-102`. CONFIRMED by reading. For an
    `s` argument `n - d` underflows when the type tags end at the last byte of the packet, and
    `strnlen` runs past the datagram into the stack buffer. Read only, no crash expected. Fix:
    return when `d >= n`. The rest of the parser held up (truncated arguments, bundles, oversize).

Nothing serious found in: shared buffers (no race; an oversize blob gives an `E` and the worker
carries on), leaks and orphans on reload or exit (by reading), clock drift (the picture follows the
sound; pause, seek, `--no-audio`, `--offset`, loop are sound), DPI and paths with spaces, the tower
tool's file write (not atomic, but the watcher's delays make a half-read unlikely), memory of one
worker over 10 008 frames (150.7 to 156.5 MB, handles constant).

Also noted:
- The "dropped" counter does not count a stall or jump of more than 1 s, a loop restart, a reload
  take-over, or anything after the renderer (Present, Spout, MadMapper).
- The value tuned with `[` `]` is shown nowhere, so it cannot be copied into `--offset`.
- The engine paces on the sound, not on a display: MadMapper's output will repeat or skip a frame
  now and then.
- A tower placement the tool allows (three towers leaving no bay over 40 px) makes `Context()`
  raise; the engine then cannot start, so the tool cannot be used to repair the file.

Section 11 claims found false: "if the new code does not import, the old workers keep drawing"
(false for scene modules, finding 4); "takes over under a second" (14.5 to 17 s for a new tower
placement during dance or glitch); "Enter writes towers.json and the scenes take it" (not during a
reload). Untestable without running the exe: the bench figures, "about 3 frames dropped" on reload,
Spout and MadMapper.

## B. Rendering path

Scope: `muonbloom/drawlist.py`, `src/drawlist.h`, `renderer.*`, `gpu.h`, `shaders/`, the two check
tools, against `muonbloom/engine.py`. The built `muonengine.exe render` was run on blobs the
reviewer made.

Overall: the renderer is faithful. Against a reference whose lines are sampled 4 times finer, the
GPU picture is within about 1 level of 255 on real frames, glitch frames included.

1. **`spacing` is ignored, so the streak field does not match the reference.**
   `shaders/segments.hlsl:74-111`, `muonbloom/drawlist.py:64`. CONFIRMED. The star's hair
   (`scenes/origin.py:432-433`, `width=1.5, spacing=0.8`), on screen from 18.35 s to 33.35 s, has a
   grain in the reference that the GPU draws smooth: 5.9 to 7.4 % of pixels more than 2 levels off,
   max 17 to 21, mean 0.34 to 0.40, no brightness bias. A look decision: is that grain wanted?
2. **Reversed rects are drawn where the reference does nothing.** `src/renderer.cpp:517-520`,
   `shaders/rects.hlsl:39-46`, `shaders/finish.hlsl:88-92`. CONFIRMED mechanism, latent: none in
   3334 recorded frames. A box with x1 < x0 or y1 < y0 is an empty slice in the reference and a full
   rect on the GPU (occlude, dim, invert). A future look tweak with a negative width would be
   invisible in the preview and a block on the wall. Fix: drop such boxes in `DrawList._lightop` /
   `finish`, or in C++ before upload.
3. **The GPU check cannot go red.** `tools/compare.py:118-137`. CONFIRMED by reading. It prints
   figures and always exits 0; "MISSING GLYPHS" and "FLAGS" are only notes; its documented run
   samples 35 frames and misses 18 to 33 s entirely; `check_drawlist.py`'s documented sweep runs at
   half size. Fix: compare against a replay with spacing forced small so a tight threshold (about
   2 levels) is possible, and exit non-zero.
4. **`FrameInfo.flags` and `missing_glyphs` are read only by the `render` command.**
   `src/renderer.h:33,60`, `main.cpp:219-222`. CONFIRMED by grep. If `glitch.make_post` changes its
   closure variable names, or a scene adds another `f.post`, the wall shows the frame without the
   effect and without any message; same for a glyph that is missing or did not fit the atlas.
   Fix: have the live player show or log both.
5. **The outermost pixel ring differs.** `shaders/segments.hlsl:62`, `shaders/dots.hlsl:34` versus
   `muonbloom/engine.py:251`. CONFIRMED. The reference drops any splat whose 2 x 2 footprint is not
   wholly inside the frame; the GPU integrates up to the clip rect. One pixel wide, 952 of 3334
   sampled frames. Fix: limit the integration interval to [0, W-1) x [0, H-1), as splats do.
6. **Discs larger than 30 px use a sharper edge.** `shaders/dots.hlsl:53-56`, `renderer.cpp:12`.
   CONFIRMED. A 1-pixel ring is 31 to 40 levels off; in the show: the star disc at 40 to 47 s and
   the credits at 821 to 831 s. Fix: give the fallback the 1-pixel tent blur of the splat.
7. **A corrupt glyph record crashes the renderer.** `src/renderer.cpp:264-267`. CONFIRMED (a glyph
   with w = h = 0xFFFFFFFF wraps the pointer; exit 0xC0000005). The Python writer cannot produce it;
   144 other corruptions gave no crash. Fix: reject when `n > end - p` or w / h exceed the atlas.
8. **Lines wider than 153 design px are wrong.** `src/renderer.cpp:509`, `shaders/segments.hlsl:53`.
   CONFIRMED, latent (the show's widest line is 4.0).
9. **`dim` on text rounds instead of truncating.** `muonbloom/drawlist.py:351` versus
   `muonbloom/engine.py:564`. CONFIRMED, at most 2 levels.
10. **`gpuMs()` and `finish()` spin forever if the device is removed.** `src/gpu.h:74-76,85`.
    PLAUSIBLE; used by `bench` and `render` only.

Nothing serious found in: the blob format (order, sizes, alignment, counts, empty arrays, NaN),
the primitives, text (anchors, bold, vertical, non-ASCII), tags and ordering, finish (exposure,
bloom, tonemap, red, invert, dither), the glitch post-process, state between frames, other sizes,
bypasses of the recorder (scenes only reach the pixels through recorded primitives, `f.post` and
`f.invert_rects`), capacity (largest blob 5.1 MB of 48 MB; atlas 6.8 % used).

Section 11 claim found false: "mean difference at most 0.06 levels; under 0.1 % of the pixels
further than 2 levels" does not hold from 18 to 33 s (6 to 7 %, finding 1).

Not covered: `set_view` zoom and `scale_rect` are used by no scene, so no check exercises them;
no full-show GPU comparison (about 40 real frames at full size); device loss; the glyph protocol
with 6 workers and dropped frames; font fallback on a machine without Space Mono.
