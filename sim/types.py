#!/usr/bin/env python3
# sim/types.py

from typing import NamedTuple
import numpy as np


class EulerSim(NamedTuple):
    """Simulation result for standard simulation"""
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
    """Simulation result for live simulation"""
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
    """Control command sent to step_method_forever during live execution."""
    vr: float | None = None
    u_manual: float | None = None
    α: float | None = None


type RoadSegment = tuple[float, float] | tuple[float, float, float]
