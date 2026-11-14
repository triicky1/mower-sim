#!/usr/bin/env python3

from functools import partial

from .config import Params
from .types import EulerSim

W = 61  # number of characters between ║ and ║


# ── Reusable helpers ────────────────────────────────────────────────────────

def paint(code: str, s: str, color: bool = True) -> str:
    """Color text with ANSI code, e.g. '1;32'. Unchanged text if color=False."""
    return f"\033[{code}m{s}\033[0m" if color else s


def row(text: str = "", style: str = "0", width: int = W,
        color: bool = True, align: str = "^") -> str:
    """A row in the box: ║ text ║.  align: '^' centered, '<' left, '>' right.

    Color the entire row via `style`, never parts of the text: ANSI codes count as
    characters and break padding. Text longer than `width` breaks the frame.
    """
    return (paint("36", "║", color)
            + paint(style, f"{text:{align}{width}}", color)
            + paint("36", "║", color))


def edge(top: bool = True, width: int = W, color: bool = True) -> str:
    """Top or bottom edge of the box."""
    left, right = ("╔", "╗") if top else ("╚", "╝")
    return paint("36", left + "═" * width + right, color)


# ── Banners ──────────────────────────────────────────────────────────────────

def print_welcome(settings, controller=None, sensor=None, art=None, color=True):
    """Prints a welcome banner. art = optional string with ASCII/braille art."""
    r = partial(row, color=color)
    rule = "─" * 44

    # The subtitle line is built from the parts actually included in the run
    parts = ["DC motor"]
    if controller:
        parts.append("PI controller")
    if sensor:
        parts.append("AMT102-V")

    lines = [
        edge(True, color=color),
        r(),
        r("V E H I C L E   S I M U L A T I O N", "1;97"),
        r("Euler's step method", "97"),
        r(),
        r("  ·  ".join(parts), "36"),
        r(),
        r(rule, "90"),
        r(f"t = {settings.t0:g} → {settings.tf:g} s      dt = {settings.dt:g} s", "97"),
    ]
    if controller:
        lines.append(r(f"Kp = {settings.Kp:g}   Ki = {settings.Ki:g}   Kd = {settings.Kd:g}   n_ctrl = {settings.n_ctrl}", "97"))
    else:
        lines.append(r("Control: open-loop", "97"))
    if sensor is not None and hasattr(sensor, "PPR"):
        lines.append(r(f"Encoder: {sensor.PPR} PPR  ({sensor.CPR} counts/rev)", "97"))
    lines += [
        r(rule, "90"),
        r(),
        r("Starting simulation ...", "1;32"),
        r(),
        edge(False, color=color),
    ]

    if art:
        print(art)
    print("\n".join(lines))


def summarize(res: EulerSim, name: str = "", params: Params = Params(), color: bool = True):
    """Summary of a run in the same box as the welcome banner.
    Green = within limit, red = limit exceeded."""
    r = partial(row, color=color)
    rule = "─" * 56

    e = res.v - res.vr
    up, down = res.α > 0, res.α < 0
    i_max, u_max = abs(res.Ia).max(), abs(res.Ua).max()
    dip = f"{e[up].min():.2f} m/s" if up.any() else "–"
    over = f"{e[down].max():.2f} m/s" if down.any() else "–"

    def ok(good: bool) -> str:
        return "92" if good else "91"

    def metric(label: str, value: str, extra: str = "", style: str = "97") -> str:
        return r(f"  {label:<24}{value:>12}   {extra}", style, align="<")

    lines = [
        edge(True, color=color),
        r(),
        r("R E S U L T S", "1;97"),
    ]
    if name:
        lines.append(r(name, "36"))
    lines += [
        r(),
        r(rule, "90"),
        metric("Max current |I|", f"{i_max:.1f} A", f"(I_max = {params.I_max})", ok(i_max <= params.I_max)),
        metric("Max voltage |U|", f"{u_max:.1f} V", f"(U_nom = {params.U_nom})", ok(u_max <= params.U_nom)),
        metric("Uphill speed dip", dip),
        metric("Downhill overshoot", over),
        metric("Distance", f"{res.s[-1]:.0f} m"),
        r(rule, "90"),
        r(),
        edge(False, color=color),
    ]
    print("\n".join(lines))
