#!/usr/bin/env python3

from typing import Any, Literal
from collections.abc import Iterable, Mapping
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import numpy as np
from contextlib import contextmanager

PWD = Path(__file__).resolve().parent

PEPE = PWD/'figures/.based'
WOJAK = PWD/'figures/.wojak'

def welcome_msg():
    if PEPE.exists():
        return PEPE.read_text()
    if WOJAK.exists():
        return WOJAK.read_text()


def mkdir(d: str, p=PWD):
    (p/d).mkdir(parents=True, exist_ok=True)


def savefig(title:str | None = None, dpi=600, WD=PWD, format='.pdf'):
    if title:
        if not title.endswith(format): title += format
        _ = 'figures', 'resources', 'images', ''
        p = next(p for d in _ if (p := WD / d).is_dir())
        plt.savefig(p / title, dpi=dpi)


def _marks(spec) -> Any:
    """None | 50 | [50, 60] | {50: "50 Hz split"} -> [(value, label), ...]"""
    if spec is None:
        return []
    if isinstance(spec, Mapping):
        return list(spec.items())
    if np.isscalar(spec):
        return [(spec, None)]
    return [(v, None) for v in spec]


# Default style for vertical / horizontal lines
_REF = dict(color="gray", ls="--", lw=1.0)

def decorate(
    ax: Axes, *,
    xlabel=None, ylabel=None, title=None,
    xlim=None, ylim=None,
    vlines=None, hlines=None, ref=_REF,
    legend: bool | dict | None = None,          # None = only if there are labelled artists
    **set_kw,                                   # anything else ax.set accepts
) -> Axes:
    given = dict(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim)
    ax.set(**{k: v for k, v in given.items() if v is not None}, **set_kw)
    for v, lbl in _marks(vlines):
        ax.axvline(v, label=lbl, **ref)
    for v, lbl in _marks(hlines):
        ax.axhline(v, label=lbl, **ref)

    has_labels = bool(ax.get_legend_handles_labels()[0])
    if legend is True or (legend is None and has_labels):
        ax.legend()
    elif isinstance(legend, dict):
        ax.legend(**legend)
    return ax


# Generic alias for spines
type Side = Literal["left", "top", "right", "bottom"]

def despine(
    axes: Iterable[Axes] | Axes,
    sides: Iterable[Side] | Side,
):
    """Hide the given spines on one or several axes"""
    if isinstance(sides, str):
        sides = (sides,)
    elif isinstance(sides, Mapping) or not isinstance(sides, Iterable):
        raise TypeError("sides must be a side or an iterable of sides (not a dict)")

    if isinstance(axes, Axes):
        axes = (axes,)
    elif isinstance(axes, np.ndarray):
        axes = axes.flat # 2D arrays from plt.subplots(2, 2)
    elif isinstance(axes, Mapping) or not isinstance(axes, Iterable):
        raise TypeError("axes must be an Axes or an iterable of Axes (not a dict)")

    sides = tuple(sides)
    for ax in axes:
        for side in sides:
            ax.spines[side].set_visible(False)


def draw(ax, *data, kind="plot", **kwargs):
    deco = {k: kwargs.pop(k) for k in list(kwargs) if k in _DECO}
    artist = getattr(ax, kind)(*data, **kwargs)
    decorate(ax, **deco)
    return artist

_DECO = {"xlabel", "ylabel", "title", "xlim", "ylim", "vlines", "hlines",
         "ref", "legend", "xscale", "yscale", "aspect"}


def merged_legend(ax, *others, **kw):
    """One legend for several (twin) axes."""
    pairs = [a.get_legend_handles_labels() for a in (ax, *others)]
    return ax.legend(sum((h for h, _ in pairs), []), sum((l for _, l in pairs), []), **kw)

# -------------------------------------------------------------------------
# Base configuration (applied to all plots)
# -------------------------------------------------------------------------

BASE_RC = {
    # Fonts & Math (STIX provides Times-compatible math symbols)
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    # Figure & Canvas Defaults
    "figure.dpi": 100,
    "figure.titlesize": 20,
    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # Lines & Markers
    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    # Ticks & Axes framing
    "xtick.bottom" : False,
    # "xtick.direction": "out",
    # "ytick.direction": "out",
    # "xtick.major.size": 4.0,
    # "ytick.major.size": 4.0,
    # "xtick.minor.size": 2.0,
    # "ytick.minor.size": 2.0,
    # "axes.linewidth": 0.8,

    # Grid
    # "axes.grid": True,
    # "grid.color": "#D9D9D9",
    # "grid.linewidth": 0.6,

    # Legend baseline
    "legend.frameon": False,
    "legend.borderpad": 0.4,
}


# -------------------------------------------------------------------------
# Generic template definitions
# -------------------------------------------------------------------------

TEMPLATES = {
    # Minimalist / paper style: despine top and right by default
    "clean": {
        # "axes.spines.top": False,
        # "axes.spines.right": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
    },

    # Engineering / measurement style: subtle grid for numeric readout
    "grid": {
        "axes.grid": True,
        "grid.color": "#B0B0B0",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,  # Keep grid behind curves
    },

    # Compact multi-panel style (subplots sharing axes or tight grid arrays)
    "compact": {
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },

    # Presentation / screen style: scaled for visibility from a distance
    "presentation": {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
        "axes.linewidth": 1.2,
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
}

def init_base_style(**overrides):
    """
    Apply BASE_RC globally across matplotlib. Run once at startup.
    """
    plt.rcParams.update(BASE_RC)
    if overrides:
        plt.rcParams.update(overrides)


def apply_style(template=None, **overrides):
    """
    Globally apply a preset template over the existing rcParams.
    """
    if template is not None:
        if template not in TEMPLATES:

            raise KeyError(f"Template '{template}' not found. Available: {list(TEMPLATES.keys())}")
        plt.rcParams.update(TEMPLATES[template])

    if overrides:
        plt.rcParams.update(overrides)


@contextmanager
def style_context(template=None, **overrides):
    """
    Locally apply a template and/or overrides using a context manager.
    Resets to previous state upon exit.
    """
    merged = {}
    if template is not None:
        if template not in TEMPLATES:
            raise KeyError(f"Template '{template}' not found. Available: {list(TEMPLATES.keys())}")
        merged.update(TEMPLATES[template])

    merged.update(overrides)

    with plt.rc_context(merged):
        yield


init_base_style()
