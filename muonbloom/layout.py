"""Wall canvas, frame and tower geometry shared by every scene.

The output that covers the wall is 2978 x 1400. The frame, the subtitle box (top
right) and the three detector towers are the fixed furniture; everything else is
laid out from them. The towers are *data*, not constants: their rectangles are
read from data/towers.json (the placement tool of the realtime app will write the
same file), and the bottom-band panels / stage bays are computed from them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

W, H = 2978, 1400
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# frame (from the TouchDesigner scenes, made symmetric)
FX0, FY0, FX1, FY1 = 28.0, 46.0, 2950.0, 1354.0
FRAME = (FX0, FY0, FX1, FY1)
HEAD_Y = 220.0                                   # bottom of the header band
SUB = (1469.0, FY0, FX1, HEAD_Y)                 # subtitle box, top right
STRIP0 = (FX0, FY0, SUB[0], HEAD_Y)              # score strip, top left, while the subtitle box is open
STRIP = STRIP0                                   # ... and at the frame being drawn: the show sets it (hud.strip_rect),
#                                                  the strip grows over the whole header when the box is away
MAIN = (FX0, HEAD_Y, FX1, FY1)                   # everything under the header
VIEW = (FX0 + 12.0, HEAD_Y + 20.0, FX1 - 12.0, 1190.0)   # main view above the bottom band
BOT = (FX0 + 24.0, 1214.0, FX1 - 24.0, 1340.0)   # bottom data band (split in gaps by the towers)
CELL = (FX0, 1252.0, 499.0, FY1)                 # bottom-left counter cell (the "Year" box)
CELL_R = (2479.0, 1252.0, FX1, FY1)              # its mirror, bottom right (free before the towers are revealed)
CENTER = (W / 2.0, (HEAD_Y + FY1) / 2.0)

# line weights / type sizes tuned for a 28 m brick wall (1 px ~ 1 cm)
LW_HAIR, LW, LW_BOLD, LW_FRAME = 1.0, 1.6, 2.4, 3.0
T_MICRO, T_SMALL, T_LABEL, T_TAG, T_SUB = 14, 17, 20, 22, 60


@dataclass(frozen=True)
class Tower:
    key: str            # "L", "C", "R"
    x0: float
    x1: float
    top: float          # y of the top edge = where the muon detector sits
    bot: float = FY1
    det_h: float = 120.0    # height of the detector module at the head of the tower

    @property
    def cx(self):
        return 0.5 * (self.x0 + self.x1)

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def det(self):
        """Position of the detector (origin of the blooms)."""
        return (self.cx, self.top)


# placeholder placement, measured on the TouchDesigner 'bloom' scene. CENTRE = hero (tallest)
DEFAULT_TOWERS = {
    "L": Tower("L", 523.0, 618.0, 688.0),
    "C": Tower("C", 1441.0, 1536.0, 390.0, det_h=175.0),
    "R": Tower("R", 2359.0, 2455.0, 688.0),
}
ORDER = ("L", "C", "R")
NAMES = {"L": "DET_L", "C": "DET_C", "R": "DET_R"}


def load_towers(path: Path | None = None) -> dict[str, Tower]:
    """Tower rectangles: explicit path, else $MUONBLOOM_TOWERS, else data/towers.json, else the placeholder."""
    import os
    path = Path(path or os.environ.get("MUONBLOOM_TOWERS") or (DATA / "towers.json"))
    if not path.exists():
        return dict(DEFAULT_TOWERS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: Tower(k, **raw[k]) for k in ORDER}


def save_towers(towers: dict[str, Tower], path: Path | None = None):
    path = path or (DATA / "towers.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: dict(x0=t.x0, x1=t.x1, top=t.top, bot=t.bot, det_h=t.det_h) for k, t in towers.items()}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def gaps(towers: dict[str, Tower], pad: float = 24.0, x0: float = FX0, x1: float = FX1) -> list[tuple[float, float]]:
    """Free x-intervals between the towers (and the frame), left to right: where panels can live."""
    edges = [x0] + [v for t in sorted(towers.values(), key=lambda t: t.x0) for v in (t.x0, t.x1)] + [x1]
    out = []
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a > 2 * pad + 40:
            out.append((a + pad, b - pad))
    return out


def bays(towers: dict[str, Tower]) -> list[tuple[float, float]]:
    """The stage areas between towers (no padding): [edge..L], [L..C], [C..R], [R..edge]."""
    return gaps(towers, pad=0.0)


def sub_rect(towers: dict[str, Tower], margin: float = 24.0) -> tuple[float, float, float, float]:
    """Subtitle box, top right. Its left edge steps aside when a tower stands under it, so a beam or a
    bloom rising from that tower (the centre one, with the placeholder placement) is never cut by the box."""
    x0 = SUB[0]
    for t in sorted(towers.values(), key=lambda t: t.x0):
        if t.x0 - margin < x0 < t.x1 + margin:
            x0 = t.x1 + margin
    return (x0, SUB[1], SUB[2], SUB[3])


COL_X0, COL_X1 = FX0 + 64.0, FX1 - 64.0          # text stays clear of the spectrum ticks on the frame edges


def columns(towers: dict[str, Tower], pad: float = 28.0, min_w: float = 120.0) -> list[tuple[float, float]]:
    """Usable x-intervals for text / panels, left to right: the bays between the towers, kept `pad` away from
    every tower and inside the edge ticks. A scene must take its columns from here (never fixed x positions):
    the placement, the width and the number of usable bays change with the towers."""
    out = []
    for a, b in bays(towers):
        x0 = max(a + (pad if a > FX0 else 0.0), COL_X0)
        x1 = min(b - (pad if b < FX1 else 0.0), COL_X1)
        if x1 - x0 >= min_w:
            out.append((x0, x1))
    return out


def focus_bay(towers: dict[str, Tower]) -> tuple[float, float]:
    """The bay that hosts the compositions with ONE centre: wide, as central as possible, and rather on the
    left of the subtitle box than under it."""
    def score(ab):
        a, b = ab
        c = 0.5 * (a + b)
        return (b - a) - 0.35 * abs(c - W / 2.0) - (40.0 if c > SUB[0] else 0.0)
    return max(bays(towers), key=score)


def focus_point(towers: dict[str, Tower]) -> tuple[float, float]:
    """Where a composition with ONE centre goes (the red dot, the star, the figure, the whirl): the middle of
    the focus bay, never behind a tower. The towers stand in front of the wall for the whole show, so the
    centre of the canvas is not a safe place: with the placeholder placement it is hidden by the centre tower."""
    a, b = focus_bay(towers)
    return (0.5 * (a + b), 0.5 * (HEAD_Y + VIEW[3]))


def right_cell(towers: dict[str, Tower]) -> tuple[float, float, float, float]:
    """Mirror of the counter cell, bottom right, clear of the last tower."""
    last = max(towers.values(), key=lambda t: t.x1)
    return (max(CELL_R[0], last.x1 + 24.0), CELL_R[1], FX1, FY1)


SCOPE_W = 250.0


def bottom_slots(towers: dict[str, Tower], scopes: bool = True) -> dict:
    """Bottom band, re-flowed around the towers:
         cell    the framed counter cell at the far left (the 'Year' box)
         scopes  one oscilloscope slot right of each tower base
         panels  what is left between a scope and the next tower: free for the scene, left to right
    """
    tws = sorted(towers.values(), key=lambda t: t.x0)
    y0, y1 = BOT[1], BOT[3]
    cell = (FX0, CELL[1], min(CELL[2], tws[0].x0 - 24.0), FY1)
    sc, panels = {}, []
    for i, t in enumerate(tws):
        nxt = tws[i + 1].x0 if i + 1 < len(tws) else FX1
        a = t.x1 + 46.0
        if scopes and nxt - 24.0 - a >= 150.0:          # room for a scope (dropped when two towers stand close)
            sc[t.key] = (a, y0 + 18.0, min(a + SCOPE_W, nxt - 24.0), y1)
            a = sc[t.key][2] + 26.0
        else:
            a = t.x1 + 24.0
        if nxt - 24.0 - a > 120:
            panels.append((a, nxt - 24.0))
    if cell[2] - cell[0] < 250.0:                 # a tower stands at the left edge: the counter moves to a panel
        cell = None
        for i, (a, b) in enumerate(panels):
            if b - a >= 471.0:
                cell = (a, CELL[1], a + 471.0, FY1)
                rest = (a + 471.0 + 26.0, b)
                panels[i: i + 1] = [rest] if rest[1] - rest[0] > 120 else []
                break
        if cell is None:
            cell = (FX0, CELL[1], FX0, FY1)       # nowhere to put it: zero width, the show skips it
    return dict(cell=cell, scopes=sc, panels=panels, y0=y0, y1=y1)
