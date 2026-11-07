#!/usr/bin/env python3
# dynamics.py

from typing import Optional
import numpy as np
from .config import Params

class VehicleDynamics:
    def __init__(self, params: Optional[Params] = None):
        self.params = params or Params()

    def _vehicle_acceleration(self, v: float, α: float, Fd: float) -> float:
        r"""Beräknar fordonets acceleration längs färdriktningen.

        Newtons andra lag bestämmer totalkraften som summan av alla krafter
        längs rörelseriktningen, `F = Fd - F_roll - F_drag - F_grav = ma`, där
        motståndskrafterna `F_roll` och  `F_drag` verkar mot rörelseriktningen.
        """
        veh = self.params

        A = veh.Af if v >= 0 else veh.Ab             # Tvärsnittsarea
        F_drag = 0.5 * veh.Cd * veh.ρ * A * v**2     # Luftmotstånd
        F_roll = veh.Cr * veh.m * veh.g* np.cos(α) # Rullmotstånd
        F_grav = veh.m * veh.g * np.sin(α)           # Tyngdkraft
        Ff = veh.μ * veh.m * veh.g * np.cos(α)     # Friktionskraft

        # Statisk friktion.
        if abs(v) < veh.ϵ and abs(Fd - F_grav) <= F_roll:
            return 0.0

        # Motståndskrafter verkar mot fordonets rörelseriktning
        sgn = np.sign(v) or np.sign(Fd - F_grav)
        return 1/veh.m * (Fd - (F_roll + F_drag) * sgn - F_grav)


    def dynamics_current(self, v: float, i: float, α: float = 0.0):
        """Fordonsvehl med ankarström som insignal (strömstyrning).

        Motorn matas med ström `Ia`. vehlen beräknar ankarspänningen
        `Ua = Ra · Ia + Kt · ω` (där `ω = i2or · v / r`), motorns drivkraft
        `Fd = (i2or · η / r) · (Kt · Ia - Tf · sgn(ω))` och accelerationen
        `dv/dt = f(v, α, Fd)` via `_vehicle_acceleration`.

        Note
        ----
        Spänningen är här en utsignal som regulatorn måste hålla inom |U_nom|.
        """
        veh = self.params

        ω = veh.i2or * v / veh.r      # Motorns vinkelfrekvens
        Ua = veh.Ra * i + veh.Kt * ω  # Ankarspänningen §$U_a(v)$§

        # Motorns inre friktionskrafter verkar mot vinkelfrekvensen.
        # Om varvtalet är noll motverkar friktionen det drivande momentet.
        sgn = np.sign(ω) or np.sign(veh.Kt*i)

        # Motorns effektiva drivkraft.
        Fd = (veh.i2or * veh.η / veh.r) * (veh.Kt * i - veh.Tf * sgn)
        a = self._vehicle_acceleration(v, α, Fd)

        return a, i, Ua, Fd


    def dynamics_voltage(self, v: float, u: float, α: float = 0.0):
        """Fordonsvehl med ankarspänning som insignal (spänningsstyrning).

        Motorn matas med spänning `Ua`. vehlen beräknar ankarströmmen
        `Ia = (Ua - Kt · ω) / Ra` (där `ω = i2or · v / r`), motorns drivkraft
        `Fd = (i2or · η / r) · (Kt · Ia - Tf · sgn(ω))` och accelerationen
        `dv/dt = f(v, α, Fd)` via `_vehicle_acceleration`.
        Strömmen är här en utsignal som regulatorn måste hålla inom ±I_max.
        """
        veh = self.params

        ω = veh.i2or * v / veh.r                    # Motorns vinkelfrekvens
        Ia = (u - veh.Kt * ω) / veh.Ra              # Ankarspänningen §$U_a(v)$§

        # Motorns inre friktionskrafter verkar mot vinkelfrekvensen.
        # Om varvtalet är noll motverkar friktionen det drivande momentet.
        sgn = np.sign(ω) or np.sign(veh.Kt*Ia)

        # Motorns effektiva drivkraft.
        Fd = (veh.i2or * veh.η / veh.r) * (veh.Kt * Ia - veh.Tf * sgn)
        a = self._vehicle_acceleration(v, α, Fd)

        return a, Ia, u, Fd


    def ramp_ref(self, t: float, vref: float | None = None) -> float:
        """Linjär börvärdesramp uppåt.

        Börvärdet ökar linjärt från 0 med lutningen a_max, räknat från
        starttiden t0, och begränsas uppåt av arbetshastigheten v_work.
        """
        veh = self.params
        vref = veh.v_work if vref is None else vref
        return (
            min(veh.a_max * (t - veh.t0), vref)
            if vref > 0.0 else
            max(-veh.a_max * (t - veh.t0), vref)
        )


    def deramp_ref(self, t: float, vref: float | None = None) -> float:
        """Linjär börvärdesramp nedåt.

        Börvärdet minskar linjärt med lutningen a_max, räknat från t0,
        och begränsas nedåt av 0 så att det aldrig blir negativt.
        """
        veh = self.params
        vref = veh.v_work if vref is None else vref
        return max(-veh.a_max * (t - veh.t0), 0)
