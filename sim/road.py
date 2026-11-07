#!/usr/bin/env python3
# sim/road.py

import numpy as np
from .types import RoadSegment


class RoadProfile:
    """Simulationskompatibel färdbana i 2D.

    RoadProfile((length_m, slope_deg[, vref_mps]), ...)
    """

    def __init__(self, *segments: RoadSegment):
        lengths = [float(seg[0]) for seg in segments]
        degrees = [float(seg[1]) for seg in segments]
        self.vrefs: list[float | None] = [
            float(seg[2]) if len(seg) > 2 else None for seg in segments
        ]

        self.lengths = np.asarray(lengths, dtype=float)
        self.edges = np.cumsum(self.lengths)
        self.starts = np.concatenate(([0.0], self.edges[:-1]))
        self.alpha = np.radians(degrees)
        self.dy = np.sin(self.alpha)
        self.y_starts = np.concatenate(([0.0], np.cumsum(self.lengths * self.dy)[:-1]))

    def _index(self, s: float) -> int:
        if s <= 0.0:
            return 0
        k = int(np.searchsorted(self.edges, s, side="right"))
        return min(k, len(self.alpha) - 1)

    def slope(self, s: float) -> float:
        return float(self.alpha[self._index(s)])

    def vref(self, s: float) -> float | None:
        return self.vrefs[self._index(s)]

    def elevation(self, s: float, y0: float = 0.0) -> float:
        """Beräknar vägbanans höjd y vid tillryggalagd sträcka s."""
        if s < 0.0:
            return float(y0 + s * self.dy[0])
        k = self._index(s)
        ds = s - self.starts[k]
        return float(y0 + self.y_starts[k] + ds * self.dy[k])
