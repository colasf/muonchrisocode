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
  hud.py         frame, subtitles, edge ticks, strips, rulers, tags, barcodes, callouts
  towers.py      tower faces, blooms, strings, scopes, the incoming muon
  show.py        time -> look -> scene.draw() + towers + frame + subtitles
  scenes/        one module per look
data/
  towers.json    tower placement (x0, x1, top, bot, det_h per tower) - placeholder for now
  cues.npz       loudness, bands, spectrum, kicks of the music; hits of the muon stem
```

### The towers

The three towers stand in front of the wall for the whole show. Their position, width and
height are not known yet, so nothing is placed with fixed coordinates:

* `data/towers.json` holds the three rectangles; the realtime app will write the same data
  from its placement tool.
* `layout.py` derives everything else from them: the free bays, the text columns
  (`ctx.cols`), the bottom-band slots, the subtitle box (it steps aside from a tower under
  its left edge) and the focus point (`ctx.focus`, where a composition with one centre goes).
* Rule for every scene: nothing important behind or right against a tower. Textures and
  long lines may pass behind.
* A tower that is not lit is drawn as a dark band (before "These detectors" at 01:44).

### The detectors

The realtime app receives one float per tower over OSC. For the previews the three streams
are rebuilt from the `v7 just muon sounds.wav` stem, which is panned left / centre / right
like the towers: every onset is a hit on one tower with an energy (`showdata.Detectors`).
A scene only ever asks `ctx.det` for hits and values, so live data can replace it directly.

### Subtitles

`subtitletimecode.txt` (`text  MM:SS:FF`, FF = frames at 60 fps) drives the box at the top
right. Single words (YOU, A MUON, A BLOOM, WAITING, NOTHING) are set as a large tag.
