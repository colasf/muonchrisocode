# MUON : BLOOM — realtime engine brief

Handover for the session that builds the realtime engine. Read this first; the rest of the
repository is the Python preview it has to reproduce.

**State on 2026-10-02: steps 1 to 4 of section 7 are built and the show holds 60 fps on the
development laptop. What exists, how to run it and what is left is in section 11.**

## 1. What the show is

- An audio-visual piece projected on a brick wall (about 28 x 13 m) for the BLINK festival,
  Cincinnati, October 2026. Output canvas that covers the wall: **2978 x 1400**.
- 13:22 of fixed audio plus credits (the file runs to 13:54). Stems are in
  `D:\muonchristo\audio` (V7: `muonbloom V7 no muon sounds.wav`, `v7 just muon sounds.wav`).
- Three towers stand in front of the wall, each with a muon detector at its head. Each
  detector streams **one float over OSC** (three floats in all). The towers' position, width
  and height are **not known yet**: everything is laid out from `data/towers.json`, and the
  engine needs a tool to place the towers on site and write that file.
- Look: black field, white lines, red as the only accent, Space Mono type, dense data detail.
  Fonts: `D:\muonchristo\touchdesigner\muonchristo\font\SpaceMono-*.ttf`.
- On site the pipeline is **TouchDesigner + this engine** (no Chataigne). How the two share
  the work is still open, see section 8.

## 2. The approach

**Do not port the scenes.** The engine is a GPU renderer for the drawing calls the Python
scenes already make; the scenes stay in Python and run inside the engine every frame.

- One source of truth for the graphics: the look is still being tweaked, in another session,
  on branch `build-ups`, in `D:\muonchristo\claude`.
- Whether some or all scenes are later rewritten in C++ is decided once the look is finished
  and the engine has been measured on the show machine. Nothing in the steps below depends
  on that decision.
- The Python CPU renderer (`muonbloom/engine.py`) stays as the reference: the engine's output
  is compared with it frame by frame.

## 3. Why it is feasible: measurements

`python tools/profile_logic.py` runs the whole show with a frame that only counts what the
scenes ask it to draw (no rasterisation). On the development machine, 2026-10-02:

| | per frame |
|---|---|
| median over the show | 4.2 ms |
| 95 % of frames | under 17 ms |
| worst frame | 27 ms |

Per scene (median / p95 ms): origin 1.6 / 10, star 2.4 / 14, messenger 1.6 / 4.5,
shower 4.7 / 7.8, you 5.8 / 13.5, muon 3.1 / 4.3, detector 3.0 / 4.7, bloom 7.5 / 16.6,
**galaxy 17.6 / 21.9**, sphere 11.9 / 16.2, flood 7.9 / 15, dance 6.3 / 10, glitch 5.9 / 10.5,
outlast 2.6 / 3.8, rise 3.7 / 4.5, **disintegrate 13.7 / 21.8**, outro 3.5 / 5.1, credits 3.2 / 4.7.

- 30 fps: fine everywhere. 60 fps: galaxy and disintegrate are too slow as they are; bloom and
  sphere peak just over the limit.
- Largest draw list of a frame: about 135 000 line segments, 70 000 single points,
  46 000 discs, 7 000 filled rects, 4 000 glyphs. Small for a GPU.
- Not included: handing the draw list to the GPU, and the one pixel-space effect (section 5).
- The 2 to 6 s a preview frame takes today is CPU rasterisation, which the engine replaces.

## 4. What exists (read-only for the engine session)

```
muonbloom/
  engine.py      CPU reference renderer: class Frame = THE CONTRACT (section 5), cameras
  build.py       build-up animations (pure Python, runs before the primitives: nothing to port)
  layout.py      canvas, frame, header, towers -> bays / columns / focus point / bottom slots
  showdata.py    SECTIONS (time -> look), subtitles, audio cues (Cues), detector streams (Detectors)
  hud.py         shared furniture: frame, subtitles, strips, panels, counters, callouts
  towers.py      tower faces, blooms, strings, scopes
  show.py        Show.render(t): scene.draw + towers + scopes + counter + edge ticks + frame + subtitles
  scenes/*.py    one module per look (16)
data/towers.json tower rectangles (placeholder);  data/cues.npz  audio analysis (tools/analyze_audio.py)
preview.py       CLI: stills, storyboard, video with audio
tools/           check_frames.py (determinism + error sweep), filmstrip.py, profile_logic.py
```

- A scene is a pure function of show time `t`, the audio cues and the detector streams. No
  state is kept between frames. Scenes only need numpy and the standard library (PIL is used
  by `engine.py` alone).
- `Show.render(t, W, H)` is the whole frame. `W` can be anything; all coordinates are given
  in design pixels (2978 x 1400) and scaled by `Frame.s = W / 2978`.

## 5. The contract: `engine.Frame`

Everything on the wall is light added on black, in **two scalar layers**: `"w"` (white) and
`"r"` (red, shown as RGB 1.0 / 0.045 / 0.035).

The build-up animations (`Frame.build`, `build.Block`) act **before** the primitives: by the
time a call reaches the raw primitives below it has already been transformed. A draw-list
recorder therefore overrides exactly these (as `tools/profile_logic.py` does):

| raw primitive | what it draws |
|---|---|
| `_segments(layer, x0, y0, x1, y1, i0, i1, width, spacing)` | anti-aliased lines, intensity interpolated from `i0` to `i1`, clipped to the clip rect. Intensity is light per pixel of length, times `s ** 0.35`. `width` > 1.25 is drawn as parallel passes. `polyline`, `rect`, `rings`, `crosses` all end here. |
| `_dots(layer, x, y, r, i)` | filled anti-aliased discs (radius in design px, times `sqrt(view zoom)`); dropped if the centre is outside the clip rect |
| `pixels(layer, x, y, i, snap)` | single-pixel points (lattices, point clouds) |
| `_rects(layer, x0, y0, x1, y1, i)` | filled rectangles, snapped to pixels, clipped, additive |
| `points(layer, x, y, w)` | raw splats in pixel space (rare) |
| `_text(layer, x, y, s, size, alpha, anchor, bold)` | Space Mono text. Anchor is PIL's (`ls`, `rs`, `ms`: left / right / middle, baseline). Text is an 8-bit layer of its own per colour: it does not add up with itself. |
| `_tag(layer, x, y, s, size, alpha, anchor, pad, bold, ref, wipe)` | inverted label: solid box with the letters cut out. `ref` sizes the box, `wipe` < 1 draws that fraction of it from the left. Returns its box. |
| `text_vertical(layer, x, y, s, size, alpha)` | text rotated 90 degrees counter-clockwise |
| `occlude(x0, y0, x1, y1)` | blacks out everything drawn **so far** under a rect (both layers and text) |
| `dim(x0, y0, x1, y1, factor)`, `scale_rect(layer, ...)` | multiplies what was drawn so far |
| `set_clip(...)`, `set_view(zoom, cx, cy, sx, sy)` | state: screen clip rect, 2D zoom |

Things to get right:

- **Order matters.** `occlude` and `dim` only affect what was drawn before them, so the draw
  list is ordered. Between two of them the geometry is purely additive and can be batched.
- **Tags always win over geometry.** Their boxes are remembered and the geometry under them is
  zeroed at the end of the frame, even geometry drawn after the tag. A tag also clears the
  other colour's text under its box.
- **Finish** (`Frame.finish`, called through `hud.finish`): layers times `exposure`, text
  added, `post` callables, bloom (8-level pyramid, weights `hud.BLOOM`, gain 0.75 by default),
  soft-knee tonemap (knee 0.72) per layer, white + red times the red colour, optional inverted
  rects (white field, black lines, red stays red), dither. `Frame.noglow_rects` (the subtitle
  box, `hud.subtitle`) are left out of the bloom: black on its level 1, as a source and as a
  sum (`engine.bloom(mute=)`, section `noglow` of the draw list, `Renderer::mute`).
- **One pixel-space effect:** `scenes/glitch.py: make_post` tears, repeats and smears the
  light buffers on the beat (06:55 to 07:34). It has to be rewritten as a shader, or kept on
  the CPU for that scene.
- Options a scene returns to the show (`show.Scene.draw` docstring): invert, invert_rect,
  bloom_gain, exposure, towers, tower_dim, tower_outline, burst_gain, burst_size, scopes,
  cell, cell_alpha, cell_age, edge_ticks, edge_alpha, edge_kw, frame, frame_alpha.

## 6. Inputs of a frame

- **Time**: seconds from the start of the audio. The engine owns the clock (or follows one,
  section 8) and plays the audio.
- **Audio cues**: `data/cues.npz`, analysed offline from the stems (loudness, bands, 48-bin
  spectrum, kicks, onsets). Fixed audio, so this stays offline.
- **Detectors**: in the previews the three streams are rebuilt from the muon stem
  (`showdata.Detectors`: hits with a time, an energy and an echo flag, `value(key, t)` as the
  float a tower streams). Live, they come over OSC. A scene only asks `ctx.det`, so a live
  class with the same methods can replace it. See the open point about anticipation below.
- **Towers**: `data/towers.json` (x0, x1, top, bot, det_h for L, C, R). `layout.py` derives
  every position from it; scenes never use fixed x positions.

## 7. Order of work

1. **Draw list.** A `Frame` subclass that records the raw primitives of section 5 into flat
   arrays, in order, plus a file format for one frame and for a time range. New file, owned by
   this session (suggested: `muonbloom/drawlist.py`).
2. **Renderer.** C++ program that plays a recorded draw list at 2978 x 1400 with the two
   layers, text, occlusion, bloom and tonemap, and is compared with the Python stills of the
   same times.
3. **Live.** Embed Python (the same interpreter version as the previews, numpy bundled), call
   `Show.render` every frame, audio playback and clock, reload a scene module when its file
   changes, keep the last good frame if a scene raises.
4. **Show features.** OSC input for the detectors, the tower placement tool, output to
   TouchDesigner.
5. **Measure on the show machine**, then optimise or port only what fails.

Anything under `engine/` is this session's. Language, graphics API, window and build system
are open; Windows is the development machine.

Status, 2026-10-02: steps 1, 2 and 4 are done as written (C++, Direct3D 11, output by Spout).
Step 3 is done differently: Python is not embedded, the scenes run in worker processes ahead
of the clock (section 11 says why). Step 5 was done on the laptop, which the user takes as
the reference; nothing had to be optimised or ported.

## 8. Decisions

Given by the user on 2026-10-02:

- **Frame rate: 60 fps.**
- **Show machine: more powerful than the development laptop** (RTX 4080 Laptop, 32 threads).
  The laptop is the benchmark: what holds there holds on site.
- **Output: the picture goes by Spout to MadMapper**, which does the mapping on the wall.
- **Sound: Ableton plays it.** The sound must not drive the engine's clock from inside the
  engine: the engine follows a time it is given (`/muonbloom/time`, section 11).
- **Detectors: TouchDesigner processes the muon data and sends it to the engine by OSC.**

The chain on site:

```
Ableton        the sound, and the time of the show
TouchDesigner  the muon data, processed       --OSC-->  engine
engine         the picture, 2978 x 1400       --Spout-> MadMapper   the mapping on the wall
```

Still open (ask the user):

- **Date**: when the show has to run on site, and when the show machine and the detectors are
  available.
- **How Ableton's time reaches the engine.** The engine takes `/muonbloom/time <seconds>` by
  OSC. Who sends it is not settled: TouchDesigner (it can read Ableton's song time with
  TDAbleton and pass it on with the muon data) or a Max for Live device in the set.
- **Is the detector signal a stream or one message per hit?** TouchDesigner will send
  `/muon/L`, `/muon/C`, `/muon/R` with one float 0..1 (user, 2026-10-02), and the trigger
  level in that signal is set in the engine (section 11). Both kinds are understood; which
  one it is, and its rate, will be known when TouchDesigner sends it.
- **Scripted versus live.** Several scenes know a hit before it happens: the "INCOMING / ETA"
  countdown towards a tower, the times each detector switches on (`showdata.T_ON`), the galaxy
  labels that look ahead. With real detectors that is impossible: it has to be decided which
  moments are scripted and which react to live data.
- **Towers**: are their fronts projected on or masked.
- **Full C++ or not**: after the look is finished (section 2).

## 9. Working rules

- **Ownership.** The look session owns `muonbloom/` (except `drawlist.py`), `preview.py`,
  `storyboard.txt`. This session owns `engine/` and `muonbloom/drawlist.py`, and only reads
  the rest. If the engine needs a change in a scene or in `engine.py`, ask the user to pass it
  to the look session.
- **Branches.** This folder (`D:\muonchristo\engine`) is a git worktree on branch `engine`,
  created from `build-ups` (commit cd75082). The look continues on `build-ups` in
  `D:\muonchristo\claude`; merge `build-ups` into `engine` to pick up look changes. `main`
  does not contain the build-ups yet. Remote: `git@github.com:colasf/muonchrisocode.git`.
- **Git.** Commit and push only when the user asks. Images and video are not versioned
  (`.gitignore`).
- **Rendering load.** The look session renders with up to 24 processes; the machine has
  32 threads.
- **Checks that must stay green**: `python tools/check_frames.py` (every look identical across
  processes) and `python tools/check_frames.py --sweep 5` (no exception over the show).
- **Reporting to the user**: short and plain; say what was verified and what was not.

## 10. Changes to the drawing contract

The look session appends here whenever a tweak adds or changes a drawing call.

- 2026-10-02: contract as described in section 5 (commit cd75082).

What the recorder (`muonbloom/drawlist.py`) relies on, beyond the table of section 5. A tweak
that changes one of these needs a word to the engine session:

- It overrides `_segments`, `_dots`, `points`, `_rects`, `_text`, `_tag`, `text_vertical`,
  `occlude`, `dim`, `scale_rect`, `set_view`, `set_clip`, `flush`, `finish`. `pixels` is not
  overridden: it ends in `points`, in pixel space, and is recorded there.
- `occlude`, `dim`, `text_vertical` and `_tag` are copied from `engine.py` (their rounding to
  pixels is part of the picture). A change in those four has to be copied again.
- `scenes/glitch.py make_post` is read as data: `drawlist.post_ops` replays its random draws.
  If `make_post` changes, `engine/tools/check_drawlist.py` reports "post ... DIFFERENT" and
  `post_ops` has to follow. Any other `Frame.post` callable is not drawn by the engine.
- `finish(palette=...)` is ignored (the show does not use it).
- A scene must not keep detector data it read in its constructor if the detectors are live.

## 11. State of the engine (2026-10-02)

### What is built

```
muonbloom/drawlist.py        DrawList: an engine.Frame that records the raw primitives into one binary
                             blob per frame (format at the top of the file); replay(): a blob
                             rasterised with the reference code
engine/src, engine/shaders   muonengine.exe: C++ / Direct3D 11 renderer of the blobs, live player
engine/worker.py             scene worker: runs Show.render on a DrawList, writes the blob in shared memory
engine/detectors.py          live detector streams for the scenes (replaces showdata.Detectors)
engine/third_party/spout     Spout2 SDK, DirectX 11 part, unchanged (BSD)
engine/tools                 check_drawlist.py, compare.py, fake_clock.py, fake_detectors.py, osc_send.py
engine/build.bat, run.bat    build (Visual Studio 2022, CMake, Ninja; nothing else to install), play
```

**How 60 fps is reached.** The brief said "embed Python and call `Show.render` every frame".
One interpreter cannot do it: the scene logic alone takes up to 60 ms a frame when several
run side by side (galaxy 22 ms median, bloom 49 ms worst). The scenes are pure functions of
the show time, so the engine starts 6 Python processes and asks each for a different frame,
6 frames (100 ms) ahead of the clock. The renderer draws the frame that is due. The scenes
are untouched. Cost: a live detector value reaches the wall about 0.1 s after it arrives.

**The renderer** draws a frame in 2 to 4 ms on the laptop (about 1 ms when the GPU is kept
busy). Lines are the exact integral of the reference's splats (lines drawn with a spacing
over half a pixel - the streak field, the whirl - keep their grain: their splats are summed
one by one); discs come from a table built the way `engine._disc` builds them (over 30 px:
the same edge, computed); text is PIL's own glyph bitmaps on whole pixels, as PIL places
them; bloom, tonemap, inverted rects and the glitch post-process are shaders.

### Run it

```
engine\build.bat
engine\run.bat --paused --detectors live    the show: waits for Ableton's time, takes the detectors from OSC
engine\run.bat                          at the desk: plays by itself on the machine's timer, no sound
engine\run.bat --sound                  at the desk, with the stems of the previews (the engine follows them)
engine\run.bat --from 180 --paused      start somewhere, paused
engine\run.bat --loop                   start again at the end
engine\build\muonengine.exe bench       the whole show in real time, no window: frames in time per scene
```

`run.bat` is the way to run it unattended: if the engine stops by itself (a crash, the
graphics driver reset) it is started again where it was; it is not started again when
somebody closed it, or when it cannot start as it is set up. The messages of the engine
also go to `engine/out/engine.log`. The machine is kept awake while it runs.

Keys: Space play / pause; Left / Right 5 s (Shift 30 s, Ctrl one frame); Page Up / Down
scene; Home start; F full screen; B time bar; T tower placement; R reload the scenes; `[` `]`
picture 5 ms earlier / later than the clock (the value is shown in the title: put it in
`--offset`); P the output panel, K the test card, S a snapshot; with live detectors, D detector meters, 1 2 3 0 and Up / Down the trigger
level; Esc quit. The title bar shows the time code, the scene, where the clock comes from,
the frame rate, the frames dropped, the time the scenes take and the trigger levels.

Added 2026-10-03, in the bar under the picture (`engine/src/gui.h`, drawn with a small bitmap font):

- the time code `MM:SS:FF` (FF = frames, as in the timings sheet), right of the play / pause button;
- `SOUND` button, or key O: a file dialog chooses the WAV file(s) the engine plays and follows from then
  on (PCM WAV only; several files = stems that start together). Shift+O: no sound from the engine again.
  The choice is not kept for the next start (on site Ableton plays the sound);
- `COMMENT` button, or key C: a line to type a comment in; Enter writes it to `comments.txt` in the
  repository folder as `[ ] MM:SS:FF | scene | date | text`, Esc cancels. The show is paused while it is
  typed (not when it follows `/muonbloom/time`). The comments show as yellow marks on the time line,
  grey once their line starts with `[x]`.
- `MOVE` in the OUTPUT panel (`<` `>` `UP` `DOWN`, a pixel a click, ten with Shift, `0` puts it back; up to 400):
  what is sent to the display of the output is shifted by that many pixels, to sit the picture on the wall. Whole
  pixels, nothing filtered; what leaves the display is cut. Only the output: not the preview, not Spout. Kept in
  `engine/output.json` (`"move": [x, y]`); `Renderer::output(..., sx, sy)`.
- `GLOW`, `RED`, `WEIGHT` in the OUTPUT panel (three sliders; a click on a name: back to what the scenes give):
  GLOW 0 .. 1 = how much of the glow of the scenes is kept (`bloom_gain` times it); RED 0 .. 3 = gain of the red
  layer before the tonemap (thin red lines come up, full red stays); WEIGHT 0 .. 1.5 = pixels added to the width of
  every line (a 1 px hairline starts to gain above 0.25: a line is drawn in several passes from 1.25 px on).
  `Renderer::tune`, `gRed` / `gWeight` in the frame constants; kept in `engine/output.json`. At 1, 1, 0 the
  picture is the reference (compare.py). They change the picture itself: preview, Spout and output alike.
- `CLAUDE` button, or key A: the same line, but Enter sends it to Claude Code (`claude -p`, started in the
  repository folder, with the time code and the scene on screen in front of the text). The answer comes in a
  panel above the bar; Shift+A (or Shift and the button) hides and shows it. Every prompt of one run of the
  engine goes on in one conversation. Nobody can approve a permission there: what it may do is on its command
  line (`Claude::ARGS` in `live.cpp`: edit files, run python), or the one line of `engine/claude_args.txt`
  when that file exists. Never in the Spout or HDMI output.
  While the prompt is typed, dragging on the picture draws on the frame (light blue; Backspace on an empty
  line takes the last stroke away). A prompt sent with a drawing writes the frame with the drawing to
  `snapshots/MM-SS-FF_scene_drawn.png` (and the frame alone to `snapshots/untouched/`) and names that file in
  the prompt, so that Claude reads it (`Sketch`, `drawnSnapshot` in `live.cpp`).

Added 2026-10-03 (evening), for the site: the picture goes to the media server of the production (a
Disguise) over HDMI, and three things are set from the window (`engine/src/live.cpp`: `OutWindow`,
`OutputPanel`; `Renderer::output`, `lift`, `card`).

- `OUTPUT` button in the bar, or key P: the output panel, above the bar (in the preview window only).
  - **LIFT**: raises the mid levels of the picture, `v -> v (1 + lift) / (1 + lift v)` on the strongest
    channel, black and white staying where they are (0 = the picture as rendered; 1 takes a level 0.6
    to 0.75 and 0.42 to 0.59). Drag the slider; a click on the name puts it back to 0. It is applied
    before the dither, in the last pass of the renderer, so it is in the Spout output, the HDMI output
    and the preview alike. For a wall where the dim greys sink into the stray light.
  - **TEST CARD** (or key K): the test card instead of the show, on every output (the show goes on
    underneath). `tools/test_card.py` makes it (`engine/out/testcard_3000x1688.bgra`, and a PNG in
    `previews/testcard/`); the engine runs that tool by itself when the card is missing or older than
    `data/towers.json`. The OUTPUT button is red while the card is on. The lift does not touch the card.
  - **OUTPUT**: `OFF`, or the display that takes the raster: a window that fills that display, without
    border or mouse pointer, on top of everything there, which never takes the keyboard. It shows the
    delivery raster (3000 x 1688) with its top left corner at the top left of the display, black but for
    the picture of the show at 11, 272 of it - copied, not drawn: **pixel for pixel**. So the HDMI output
    has to be set to 3840 x 2160 at 60 Hz (Windows display settings), RGB full range (NVIDIA control
    panel), and the media server takes the top left 3000 x 1688 of what it captures. A display smaller
    than the raster gets it scaled to fit, and the panel says so in yellow (fine as a second preview,
    not as the feed). The display this window is on cannot be chosen. If the display of the output goes
    away (a cable), the title and the log say OUTPUT LOST and it is taken again as soon as it is back.
  - What is set in the panel is kept in `engine/output.json` (lift, display) and used at the next start,
    also when `run.bat` starts the engine again after a crash. The test card is never kept.
  - At start: `--output 2` (a display as the panel numbers them: the log lists them; or its device name;
    `off`), `--lift 0.6`, `--card`, `--raster 3000x1688`, `--picture-at 11,272`, `--raster-at 0,0`.
- The Spout sender is unchanged (the 2978 x 1400 picture): MadMapper can still be fed, at the same time.

Added 2026-10-04, to annotate pictures of the show (`engine/src/live.cpp`: `snapshot`; `gui::savePng`):

- `SNAPSHOT` button in the bar, or key S: the picture that is on screen (2978 x 1400, as it leaves the
  engine: with the lift, with the tower tool or the test card when they are on, never the bar) is written
  to `snapshots/MM-SS-FF_scene.png` in the repository folder; the name is the time code of that frame. A
  second snapshot of the same frame gets `_2`: a picture that is there is never written over. The show
  does not stop (the PNG is packed by a thread of its own).
  A copy that nobody touches goes to `snapshots/untouched/`: what was drawn on a snapshot is what differs
  from that copy. A line `[ ] MM:SS:FF | scene | date | snapshot snapshots/<file>` is added to
  `comments.txt`, so that the snapshot has its mark on the time line and is read with the comments.
  Shift+S, or Shift and the button, opens the folder. PNG files are not versioned (`.gitignore`).

- **Time bar**: under the picture, in the preview window only (never in the Spout output).
  A play / pause button at its left, then one block per scene; click or drag to move in the
  show; the title shows the time and the scene under the mouse. B or `--no-bar` hides it.

- **MadMapper**: add the Spout input "MuonBloom" (2978 x 1400, 60 fps). The engine and
  MadMapper must run on the same graphics card (a Spout texture is shared on one card only;
  on the laptop the engine takes the NVIDIA card).
- **Tower placement** (T): the three towers are laid over the picture, also in the Spout
  output, so they are placed looking at the wall. 1 2 3 select a tower, Tab the handle
  (whole, left edge, right edge, top, detector height), arrows move by 1 px (Shift 10 px),
  or drag with the mouse in the preview window. Enter writes `data/towers.json` and the
  scenes take it; Esc leaves without writing. A placement that leaves less than 300 px of
  wall free in one piece is not saved (the scenes could not start with it).
- **Reload**: when a file of `muonbloom/`, `data/towers.json`, `data/cues.npz`,
  `subtitletimecode.txt`, `engine/worker.py` or `engine/detectors.py` changes, a new set of
  workers is started; it takes over when it has built the scene that is playing (under a
  second in most scenes, about 3 frames dropped; 15 s and more in dance and glitch after a
  new tower placement, the old picture playing meanwhile). If the new code does not start,
  or the scene that is playing raises with it, the old workers keep drawing and the title
  says why. A change that arrives during a reload starts the reload again. If a scene
  raises on a frame, that frame is skipped: the last good picture stays, the error goes to
  the console. A worker that dies, or does not answer for a minute, gets the set started
  again.
- **Clock**: the sound is played by Ableton, so the time of the show comes from outside:
  `/muonbloom/time <seconds>` by OSC (a float or a double), sent all the time, many times a
  second, also while Ableton is stopped. The engine runs on the machine's timer and is pulled
  onto that time: it takes the freshest of the times received over half a second (the others
  arrived late), so the jitter of the messages does not reach the picture.
  - the time moves: the show plays there (it starts by itself, also from `--paused`);
  - the time stands still for 0.4 s: the show pauses there (`/muonbloom/pause` does it at once);
  - the time jumps by more than 80 ms: the show is moved there;
  - nothing arrives for 1.5 s: the show goes on by the machine's timer, and the title and the
    log say so. A sender that only speaks when the time changes would look like this when
    Ableton stops: it has to send all the time.
  Without any `/muonbloom/time` the engine runs on its own timer (Space, the time bar).
  `python engine/tools/fake_clock.py --from 180` is a pretend Ableton to try it.
  The picture has to leave ahead of the sound by the delay of the chain (TouchDesigner or
  the device that reads Ableton's time, Spout, MadMapper, the projectors): `--offset MS`.
- **OSC** (UDP port 9000, `--osc PORT`): `/muonbloom/time <seconds>`, `/muonbloom/play`,
  `/muonbloom/pause`, `/muonbloom/seek <seconds>`, `/muonbloom/reload`. Only from this
  machine by default: `--osc-allow 10.0.0.5,10.0.0.6` adds addresses (needed if
  TouchDesigner or Ableton run on another machine), `--osc-allow any` takes them from anybody.
- **Detectors**: by default the scripted hits of the previews. `--detectors live` takes them
  from OSC instead, from any address (`/muon/L`, `/muon/C`, `/muon/R` with one float 0..1, or
  `/muon` with three floats; `--det-prefix` changes `/muon`); `--detectors both` adds them
  to the scripted ones. This is what TouchDesigner will send (user, 2026-10-02). Two kinds
  of signal are understood: one that streams its value (a hit is a rise above the trigger
  level; the detector is ready again once the value is back under 0.6 times that level) and
  one that only sends a message when it is hit (every message above the trigger level is a
  hit once the detector has been silent for 0.25 s).
  `python engine/tools/fake_detectors.py` sends pretend detectors to try it.
- **Trigger level** of the detectors, one per detector, 0.10 unless set:
  - at start: `--det-level 0.25` (the three) or `--det-level 0.2,0.3,0.25` (L, C, R);
  - while it runs, in the preview window: D shows three meters at the top left (L, C, R from
    the top: the value in white, red above the level; its peak of the last second; the level
    in yellow; a lamp at each hit), 1 2 3 choose a detector (0: the three), Up / Down move
    its level by 0.01 (Shift 0.05). The meters are in the window only, never in Spout;
  - by OSC: `/muonbloom/level <v>`, `/muonbloom/level <L> <C> <R>`, `/muonbloom/level/L <v>`.
  What is set while it runs is written to `engine/detectors.json` and used at the next
  start (the command line wins over the file). A new level applies to the values that
  arrive after it: the hits already found stay. The other constants (release at 0.6 of the
  level, no second hit within 50 ms) are in `engine/detectors.py`.
- **Sound**: none by default (Ableton plays it). For work at the desk, `--sound` plays the
  two stems of the previews (looked for in the `audio` folder beside the repository; the
  engine refuses to start if they are asked for and not there) and `--audio file.wav` plays
  other files; the engine then follows that sound and ignores `/muonbloom/time`. If the
  sound output is lost while it plays, the picture goes on by the machine's timer and the
  output is opened again as soon as it is back.
- Other options: `--sound`, `--audio file.wav` (several), `--volume 0..1`, `--no-spout`,
  `--spout NAME`, `--offset MS`, `--workers N`, `--lead N`, `--python PATH`, `--root FOLDER`
  (default: two levels above the exe), `--log FILE`, `--no-log`.
- **Exit code**: 0 stopped by somebody, 2 it cannot start as it is set up, 3 it failed while
  running (the graphics device is gone ...).

### Checks (run them after merging `build-ups` into `engine`)

```
python tools/check_frames.py                       the look session's check: must stay green
python engine/tools/check_drawlist.py --sweep 7    a recorded frame, rasterised by the reference code, is the preview
python engine/tools/compare.py --probes            the GPU picture against the reference, in floats (exit 1 on a wrong picture)
python engine/tools/check_output.py 2:58 7:20.1    what the HDMI output is sent, byte for byte: the picture untouched in the raster
engine\build\muonengine.exe bench                  60 fps over the whole show
```

### Measured on the laptop, 2026-10-02

- `check_drawlist.py --sweep 7`: 119 frames, all within 2 levels of the preview on isolated
  pixels (text rounding), glitch operations identical to `make_post`.
- `compare.py --probes` and `--sweep 60` (39 frames, full size), after the review fixes:
  against the reference with its lines sampled four times finer, the mean difference is at
  most 0.004 levels of 255 and no frame has more than 0.003 % of its pixels further than
  2 levels. (Against the reference as the previews draw it, a line differs by up to a few
  levels on isolated pixels: the previews sample a line every half pixel, the engine
  integrates it.)
- `bench`, whole show, 6 workers, 6 frames ahead, after the review fixes: 50 039 frames,
  none dropped, 1 shown a few milliseconds late. Scene time per frame: median 6.4 ms, 95 %
  under 21 ms, worst 62 ms. GPU time per frame: median 2.3 to 4.1 ms, worst 18 ms.
- The review of 2026-10-02 and what was done about each finding: `engine/REVIEW.md`.

### Not done, or not verified

- The output window (2026-10-03) was tried on a second display of 1920 x 1200 only: smaller than the
  raster, so what was seen on a real display is the scaled case. The pixel-for-pixel case is checked
  byte for byte on a 3840 x 2160 target without a display (`check_output.py`), not on a 4K display, and
  not through an HDMI capture: the test card is there to check that on site (gratings even, red lines
  red, the steps near black and near white all there). The loss of the display (a cable pulled) was not
  tried. The OUTPUT panel was driven with posted mouse messages, not by hand.
- The SNAPSHOT button and its key (2026-10-04) were driven with posted messages too, not by hand.
- The clock by OSC was tried with a pretend sender only (`fake_clock.py`: play, a stop, a
  locate, the sender disappearing; with messages up to 12 ms late the clock stays on the
  freshest ones). Not tried with Ableton or TouchDesigner. The delay of the chain has to be
  measured on site, picture against sound, and set with `--offset`.
- The engine's own sound (`--sound`) was only run muted.
- MadMapper itself was not tried: the Spout output was checked with a receiver built from
  the Spout SDK (right size, 60 fps, right picture).
- The tower tool and the keys were not tried by hand (the overlay was checked in the Spout
  output).
- Live detectors: the path works with the pretend detectors, and so does the trigger level
  (option, OSC, the file; three levels in one run gave 47, 29 and 2 hits on the same kind of
  signal). The Up / Down keys were not pressed by hand. Which moments stay scripted is not
  decided (section 8), and several scenes compute things from hits they know in advance.
- 60 fps scene logic in one process (to cut the 0.1 s of look-ahead) would need the heavy
  scenes optimised by the look session: galaxy, disintegrate, sphere, bloom. Not needed for
  the frame rate.
