"""The show: maps time to a look, draws it, then lays the shared furniture on top.

    look = scene class (muonbloom/scenes/*.py) registered under the name used in showdata.SECTIONS
    frame = scene.draw(f, t, ctx)  +  towers overlay (if the scene wants it)  +  frame  +  subtitles

A scene is a pure function of the show time `t` (seconds from the start of the audio), the audio
cues and the detector streams: the same structure the realtime app will have.
"""
from __future__ import annotations

import importlib

import numpy as np

from . import hud, towers
from . import layout as L
from . import showdata as sd
from .engine import Frame

# look name -> (module, class)
LOOKS = {
    "origin": ("origin", "Origin"),
    "star": ("origin", "Origin"),
    "messenger": ("messenger", "Messenger"),
    "shower": ("shower", "Shower"),
    "you": ("you", "You"),
    "muon": ("muon", "Muon"),
    "detector": ("detector", "Detector"),
    "bloom": ("bloom", "Bloom"),
    "sphere": ("sphere", "Sphere"),
    "galaxy": ("galaxy", "Galaxy"),
    "flood": ("flood", "Flood"),
    "dance": ("dance", "Dance"),
    "glitch": ("glitch", "Glitch"),
    "outlast": ("outlast", "Outlast"),
    "rise": ("rise", "Rise"),
    "disintegrate": ("disintegrate", "Disintegrate"),
    "outro": ("outro", "Outro"),
    "credits": ("outro", "Outro"),
}


class Scene:
    """Base class of a look. Override draw(); the class attributes say what the show adds on top."""
    name = "scene"
    towers = "auto"         # "auto": dark before the reveal, then faces + compact blooms; "none": dark; "own": the scene draws them
    scopes = True           # the three detector scopes in the bottom band (once the detectors are revealed)
    cell = True             # bottom-left cell: the muons through one spectator since 00:00
    edge_ticks = True       # spectrum ticks on the left / right frame edges
    frame = True
    strip_grows = True      # the score strip takes the whole header band while the subtitle box is away

    def __init__(self, ctx):
        self.ctx = ctx

    def draw(self, f, t, ctx):
        """Draw the look at show time t. May return a dict of finishing options:
        invert (bool), invert_rect, bloom_gain, exposure, towers ('auto' / 'own' / 'none'), tower_dim,
        tower_outline, burst_gain, burst_size, scopes, cell, cell_alpha, cell_age (seconds since the counter
        cell started to build), edge_ticks, edge_alpha, edge_kw, frame, frame_alpha."""
        return {}


class Show:
    def __init__(self, ctx=None):
        self.ctx = ctx or sd.Context()
        self._scenes = {}

    def scene(self, look):
        mod, cls = LOOKS[look]
        key = (mod, cls)
        if key not in self._scenes:
            m = importlib.import_module(f"muonbloom.scenes.{mod}")
            self._scenes[key] = getattr(m, cls)(self.ctx)
        return self._scenes[key]

    def render(self, t, W=L.W, H=L.H, look=None, subtitles=True):
        ctx = self.ctx
        _, sec, _ = sd.section_at(t)
        sc = self.scene(look or sec[4])
        f = Frame(W, H)
        # the score strip fills the header while the subtitle box is away (unless the scene uses that slot)
        L.STRIP = hud.strip_rect(t, ctx.sub) if sc.strip_grows else L.STRIP0
        opt = sc.draw(f, t, ctx) or {}
        f.set_clip()
        f.set_view()
        tw_mode = opt.get("towers", sc.towers)
        if tw_mode == "auto":
            towers.overlay(f, ctx, t, dim=opt.get("tower_dim", 0.3), gain=opt.get("burst_gain", 1.0),
                           size=opt.get("burst_size", 1.0), outline=opt.get("tower_outline", 0.16))
        elif tw_mode == "none":                 # the towers never go away: unlit, they are dark bands on the wall
            for tw in ctx.towers.values():
                towers.dark(f, tw)
        if opt.get("scopes", sc.scopes) and t >= sd.T_REVEAL:
            towers.scopes(f, ctx, t)
        if opt.get("cell", sc.cell):
            hud.through_you_cell(f, ctx, t, alpha=opt.get("cell_alpha", 1.0), age=opt.get("cell_age"))
        if opt.get("edge_ticks", sc.edge_ticks):
            hud.edge_ticks(f, t, ctx.cues, alpha=opt.get("edge_alpha", 1.0), **opt.get("edge_kw", {}))
        if opt.get("frame", sc.frame):
            hud.frame(f, alpha=opt.get("frame_alpha", 1.0))
        if subtitles:
            hud.subtitle(f, t, ctx.sub)
        return hud.finish(f, invert=opt.get("invert", False), invert_rect=opt.get("invert_rect"),
                          bloom_gain=opt.get("bloom_gain", 0.75), exposure=opt.get("exposure", 1.0))


def render_frame(t, W=L.W, H=L.H, look=None, _cache={}):
    """Convenience for worker processes: one Show per process."""
    if "show" not in _cache:
        _cache["show"] = Show()
    return _cache["show"].render(t, W, H, look=look)
