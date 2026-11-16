#!/usr/bin/env python3
# dynamics.py

from typing import Optional
import numpy as np
from .config import Params

class VehicleDynamics:
    def __init__(self, params: Optional[Params] = None):
        self.params = params or Params()

    def _vehicle_acceleration(self, v: float, α: float, Fd: float) -> float:
        r"""Calculates the vehicle's acceleration along the direction of travel.

        Newton's second law determines the total force as the sum of all forces
        along the direction of motion, `F = Fd - F_roll - F_drag - F_grav = ma`, where
        the resistive forces `F_roll` and  `F_drag` act against the direction of motion.
        """
        veh = self.params

        A = veh.Af if v >= 0 else veh.Ab             # Cross-sectional area
        F_drag = 0.5 * veh.Cd * veh.ρ * A * v**2     # Aerodynamic drag
        F_roll = veh.Cr * veh.m * veh.g* np.cos(α)   # Rolling resistance
        F_grav = veh.m * veh.g * np.sin(α)           # Gravitational force
        Ff = veh.μ * veh.m * veh.g * np.cos(α)       # Friction force
        # m_eff = veh.m + veh.η * veh.Jm * (veh.i2or / veh.r)**2  # Effective mass

        # Static friction.
        if abs(v) < veh.ϵ and abs(Fd - F_grav) <= F_roll:
            return 0.0

        # Resistive forces act against the vehicle's direction of motion
        sgn = np.sign(v) or np.sign(Fd - F_grav)
        return 1/veh.m * (Fd - (F_roll + F_drag) * sgn - F_grav)


    def dynamics_current(self, v: float, i: float, α: float = 0.0):
        """Vehicle model with armature current as input (current control).

        The motor is supplied with current `Ia`. The model calculates the armature voltage
        `Ua = Ra · Ia + Kt · ω` (where `ω = i2or · v / r`), the motor's driving force
        `Fd = (i2or · η / r) · (Kt · Ia - Tf · sgn(ω))` and the acceleration
        `dv/dt = f(v, α, Fd)` via `_vehicle_acceleration`.

        Note
        ----
        Here, the voltage is an output that the controller must keep within |U_nom|.
        """
        veh = self.params

        ω = veh.i2or * v / veh.r      # Motor angular velocity
        Ua = veh.Ra * i + veh.Kt * ω  # Armature voltage §$U_a(v)$§

        # The motor's internal friction forces act against the angular velocity.
        # If the rotational speed is zero, friction opposes the driving torque.
        sgn = np.sign(ω) or np.sign(veh.Kt*i)

        # The motor's effective driving force.
        Fd = (veh.i2or * veh.η / veh.r) * (veh.Kt * i - veh.Tf * sgn)
        a = self._vehicle_acceleration(v, α, Fd)

        return a, i, Ua, Fd


    def dynamics_voltage(self, v: float, u: float, α: float = 0.0):
        """Vehicle model with armature voltage as input (voltage control).

        The motor is supplied with voltage `Ua`. The model calculates the armature current
        `Ia = (Ua - Kt · ω) / Ra` (where `ω = i2or · v / r`), the motor's driving force
        `Fd = (i2or · η / r) · (Kt · Ia - Tf · sgn(ω))` and the acceleration
        `dv/dt = f(v, α, Fd)` via `_vehicle_acceleration`.
        Here, the current is an output that the controller must keep within ±I_max.
        """
        veh = self.params

        ω = veh.i2or * v / veh.r                    # Motor angular velocity
        Ia = (u - veh.Kt * ω) / veh.Ra              # Armature voltage §$U_a(v)$§

        # The motor's internal friction forces act against the angular velocity.
        # If the rotational speed is zero, friction opposes the driving torque.
        sgn = np.sign(ω) or np.sign(veh.Kt*Ia)

        # The motor's effective driving force.
        Fd = (veh.i2or * veh.η / veh.r) * (veh.Kt * Ia - veh.Tf * sgn)
        a = self._vehicle_acceleration(v, α, Fd)

        return a, Ia, u, Fd


    def ramp_ref(self, vr_prev: float, dt: float, vref: float | None = None) -> float:
        """Rate-limited setpoint.

        Moves the previous setpoint toward the target (the segment limit `vref`,
        or v_work if None) by at most a_max*dt per step, so the target can change
        at any time without the setpoint jumping.
        """
        veh = self.params
        target = veh.v_work if vref is None else vref
        step = veh.a_max * dt
        return vr_prev + max(-step, min(step, target - vr_prev))


    def deramp_ref(self, t: float, t_break: float, vref: float | None = None) -> float:
        """Linear downward setpoint ramp.

        The setpoint decreases linearly toward 0 with slope a_max, starting from
        t_break, and is clamped at 0 so that it does not overshoot.
        """
        veh = self.params
        vref = veh.v_work if vref is None else vref
        return (
            max(vref - veh.a_max * (t - t_break), 0.0)
            if vref >= 0.0 else
            min(vref + veh.a_max * (t - t_break), 0.0)
        )
