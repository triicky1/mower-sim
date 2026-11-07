#!/usr/bin/env python3
# sim/types.py

from typing import NamedTuple
import numpy as np


class EulerSim(NamedTuple):
    """Simuleringsresultat för vanlig simulering"""
    t: np.ndarray
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    α: np.ndarray
    v: np.ndarray
    vm: np.ndarray
    vr: np.ndarray
    a: np.ndarray
    Ia: np.ndarray
    Ua: np.ndarray
    Fd: np.ndarray

    @property
    def alpha(self) -> np.ndarray:
        return self.α


class EulerStep(NamedTuple):
    """Simuleringsresultat för live simulering"""
    t: float
    s: float
    x: float
    y: float
    α: float
    v: float
    vm: float
    vr: float
    a: float
    Ia: float
    Ua: float
    Fd: float

    @property
    def alpha(self) -> float:
        return self.α


class LiveCommand(NamedTuple):
    """Styrkommando som skickas in till step_method_forever vid live-körning."""
    vr: float | None = None
    u_manual: float | None = None
    α: float | None = None


type RoadSegment = tuple[float, float] | tuple[float, float, float]
