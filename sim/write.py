#!/usr/bin/env python3

from .config import Params
from .types import EulerSim


def print_welcome(settings, sensor=None, art=None, color=True):
    """Skriver ut en välkomstbanner. art = valfri sträng med ASCII/braille-konst."""
    W = 61
    paint = (lambda code, s: f"\033[{code}m{s}\033[0m") if color else (lambda code, s: s)

    def row(text="", style="0"):
        return paint("36", "║") + paint(style, text.center(W)) + paint("36", "║")

    rule = "─" * 44
    lines = [
        paint("36", "╔" + "═" * W + "╗"),
        row(),
        row("F O R D O N S S I M U L E R I N G", "1;97"),
        row("Eulers stegmetod", "97"),
        row(),
        row("Likströmsmotor  ·  PI-regulator  ·  AMT102-V", "36"),
        row(),
        row(rule, "90"),
        row(f"t = {settings.t0:g} → {settings.tf:g} s      dt = {settings.dt:g} s", "97"),
        row(f"Kp = {settings.Kp:g}   Ki = {settings.Ki:g}   Kd = {settings.Kd:g}   n_ctrl = {settings.n_ctrl}", "97"),
    ]
    if sensor is not None and hasattr(sensor, "PPR"):
        lines.append(row(f"Encoder: {sensor.PPR} PPR  ({sensor.CPR} counts/varv)", "97"))
    lines += [
        row(rule, "90"),
        row(),
        row("Startar simulering ...", "1;32"),
        row(),
        paint("36", "╚" + "═" * W + "╝"),
    ]

    if art:
        print(art)
    print("\n".join(lines))
