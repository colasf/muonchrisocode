# MUON : BLOOM — wall previews

Python preview of the whole performance for the 2978 x 1400 output that covers the wall.
Every frame is a pure function of the show time, the audio cues and the three detector
streams, so the same structure can be ported to the realtime C++ app.

## Look at the previews

| what | where |
|---|---|
| the whole show at a glance (42 key stills) | `previews/sheets/overview.jpg` |
| one contact sheet per scene of the sheet: still number, time code, section, caption, voice-over line | `previews/sheets/board_sceneNN_*.jpg` |
| every still at full resolution (2978 x 1400), named `NNN_time_section_look.png` | `previews/board/` |
| key stills mocked up on the photo of the wall | `previews/wall/` |
| the same 12 frames with three tower placements, side by side | `previews/sheets/towers_*.jpg` |
| the list of stills (edit it to add / move a still; `*` = also on the overview) | `storyboard.txt` |

To comment, the still number (001 - 117) or the time code is enough.

## Render

```
python preview.py board                         # all the stills of storyboard.txt + contact sheets + overview
python preview.py board --only 5 6              # only the scenes 5 and 6 of the sheet
python preview.py still 2:53 9:22.5             # single stills (show time, M:SS) -> previews/stills
python preview.py still 2:53 --look galaxy      # audition another look at that time
python preview.py wall 2:53 4:13                # stills composited on the wall photo
python preview.py towers 0:26 2:58 9:22         # the same frames with three tower placements
python preview.py video 2:10 3:00 --scale 0.5   # H.264 with the audio (music + muon stem)
python preview.py video --scale 0.5 --workers 24  # the whole show, 00:00 - 13:54 (about 20 min) -> previews/video
python preview.py still 2:53 --towers data/towers_alt2_example.json --tag _ALT2
python tools/analyze_audio.py                   # rebuild data/cues.npz when the audio changes
python tools/check_frames.py                    # every look: identical across processes? errors?
python tools/check_frames.py --sweep 5          # one low-res frame every 5 s over the show, errors only
python tools/filmstrip.py 12.0 13.4 --crop 620,1180,2340,1354   # frame-by-frame sheet of a region -> previews/strips
python tools/test_card.py                       # the test card for the site, in the 3000 x 1688 delivery raster -> previews/testcard
                                                # (the engine shows it: OUTPUT panel, TEST CARD)
python tools/wall_proof.py 7:20.1 2:58          # a still as it should land on the wall: four projectors, brick, stray light
                                                # (a simulation from the projector study) -> previews/wall/proof_*
```

A full-resolution frame takes 2 to 6 s; the board uses all the cores. The first dance / glitch
frame with a new tower placement takes about 35 s more (the showers are rebuilt, then cached in
`data/cache`).

## How it is built

```
muonbloom/
  engine.py      additive line renderer: white layer + red layer, bloom, tonemap
  layout.py      frame, header, subtitle box, bottom band; towers -> bays, columns, focus point
  showdata.py    sheet timeline (SECTIONS), subtitles, audio cues, detector streams
  build.py       build-up animations: how every data element is constructed and taken apart
  hud.py         frame, subtitles, edge ticks, strips, rulers, tags, barcodes, callouts
  towers.py      tower faces, blooms, strings, scopes, the incoming muon
  human.py       the figure of YOU / FLOOD / OUTRO (a real mesh, baked in data/human.npz)
  city.py        downtown Cincinnati around 9th St x Vine St, where the wall is, as line work
  show.py        time -> look -> scene.draw() + towers + frame + subtitles
  scenes/        one module per look
data/
  towers.json    tower placement (x0, x1, top, bot, det_h per tower): the Site 3.1 drawing since 2026-10-03
                 (estimates, to confirm on site); towers_td_placeholder.json = the placement used before
  cues.npz       loudness, bands, spectrum, kicks of the music; hits of the muon stem
  cincinnati.npz streets, buildings and river of downtown Cincinnati, baked by tools/build_city.py
                 (map data (c) OpenStreetMap contributors, ODbL: the credit line is set in every city view)
```

### The towers

The three towers stand in front of the wall for the whole show. Their position, width and
height will only be final on site (`data/towers.json` holds the Site 3.1 drawing: scaffold towers
on the sixths of the wall, 3.45 m tall on the sides, 6.5 m in the centre), so nothing is placed
with fixed coordinates:

* `data/towers.json` holds the three rectangles; the realtime app will write the same data
  from its placement tool.
* `layout.py` derives everything else from them: the free bays, the text columns
  (`ctx.cols`), the bottom-band slots, the subtitle box (it steps aside from a tower under
  its left edge) and the focus point (`ctx.focus`, where a composition with one centre goes).
* Rule for every scene: nothing important behind or right against a tower. Textures and
  long lines may pass behind.
* A tower that is not lit is drawn as a dark band (before "These detectors" at 01:44).

### Nothing fades: everything that shows data builds up

A strip, panel, card, note, counter, label or tag never fades in and never just appears: it is
constructed, and taken apart when it leaves. Only the image itself (3D views, fields, rings,
blooms, the light of the towers) keeps intensity ramps.

* `with f.build(age, rect):` around the drawing calls of a block does it (`engine.Frame.build`,
  `build.Block`): lines are drawn by a pen with a bright head, short marks are thrown out and
  fall back, outlines and curves are traced, bars grow, text is decoded out of random glyphs
  and its figures spin before they lock, tags are pushed out; registration brackets frame the
  block while it is made. `age` is the time since the block appeared: negative draws nothing,
  large draws as usual. `build.io(age, left)` gives the age of a block that also leaves.
* The first 16 seconds use hand-made choreographies built from the helpers of `build.py`
  (`pen`, `tag`, `open_box`, `decode`, `roll`, `flash` ...).
* Everything is a pure function of that age, so it ports to the realtime app unchanged.
* A build can only be judged in motion: `tools/filmstrip.py` tiles consecutive frames of a
  region on one sheet.

### The detectors

The realtime app receives one float per tower over OSC. For the previews the three streams
are rebuilt from the `v7 just muon sounds.wav` stem, which is panned left / centre / right
like the towers: every onset is a hit on one tower with an energy (`showdata.Detectors`).
A scene only ever asks `ctx.det` for hits and values, so live data can replace it directly.

### Subtitles

`subtitletimecode.txt` (`text  MM:SS:FF`, FF = frames at 60 fps) drives the box at the top
right. Single words (YOU, A MUON, A BLOOM, WAITING, NOTHING) are set as a large tag.

### Between two shows: the standby (12:38.5 - 20:00)

The loop is 20 minutes (`showdata.LOOP_END`). When the music has ended the wall does not go
dark: `scenes/standby.py` runs to the next show, on the detectors alone (no voice, no sound to
follow). Its header says where the loop is and counts down to the next show.

| | | |
|---|---|---|
| 13.0 | FOLLOW, 12:38.5 - 13:40 | two QR codes to scan (Instagram: Tyrell, Christo Squier) |
| 13.1 | RECORD, 13:40 - 15:50 | the last 45 seconds as a stack of lines, now at the bottom; a muon raises the line at its tower |
| 13.2 | COINCIDENCE, 15:50 - 18:00 | every muon caught sends a circle over the wall from its tower; red points where the circles of two towers cross |
| 13.3 | VISUALS BY, 18:00 - 18:30 | the Tyrell logo constructed across the wall, white only: construction lines that leave the centre tower, pens that trace its outline, a rain of leaning tracks that hatches it, a front that crosses the wall and lays the ink (outlines in `data/tyrell_logo.npz`, `tools/build_logo.py`) |
| 13.4 | FLUX, 18:30 - 20:00 | the lattice of crosses is the metre grid of the wall: every square metre says how many muons went through it in its last second |

* The times are the four rows `13.x` of `showdata.SECTIONS`: move them there, the scenes follow.
* The hits come from `ctx.det` like everywhere else. With live detectors they are the real ones;
  for the previews, hits are dealt at random after the end of the muon stem
  (`showdata.Detectors`, the same at every run).
* The QR codes are written as modules in `standby.py` (`QR`), so the show needs no new library.
  `python tools/build_qr.py` makes the rows for an address (needs `segno` and `zxing-cpp`),
  `python tools/build_qr.py --check` reads the two codes back.
* At 20:00 everything has been taken apart: the wall is black for the first second of the show.
