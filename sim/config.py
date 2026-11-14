#!/usr/bin/env python3
# sim/config.py

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Self
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

type DynamicsModel = Callable[[float, float, float], tuple[float, float, float, float]]
type ControlMode = Literal["voltage", "current"]

# TODO:
# class SensorSettings(_BaseSchema): ...
# class ControllerSettings(_BaseSchema): ...
# class PhysicsalConstants(): ...


class _BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def keys(self):
        return type(self).model_fields.keys()

    def __getitem__(self, key):
        return getattr(self, key)


class VehicleModel(_BaseSchema):
    """Vehicle mass and physical constants."""
    m: float = Field(default=250.0, gt=0, description="Vehicle mass [kg]")
    r: float = Field(default=0.2, gt=0, description="Vehicle wheel radius [m]")
    Af: float = Field(default=0.4, gt=0, description="Vehicle cross-sectional area, front [m²]")
    Ab: float = Field(default=0.4, gt=0, description="Vehicle cross-sectional area, rear [m²]")
    ρ: float = Field(default=1.225, gt=0, validation_alias=AliasChoices("ρ", "rho"), description="Air density at ground level [kg/m³]")
    Cd: float = Field(default=0.8, gt=0, description="Vehicle drag coefficient [-]")
    Cr: float = Field(default=0.05, gt=0, description="Vehicle rolling resistance coefficient [-]")
    μ: float = Field(default=0.6, gt=0, le=1, validation_alias=AliasChoices("μ", "mu"), description="Coefficient of friction between wheel and surface [-]")
    g: float = Field(default=9.82, gt=0, description="Gravitational acceleration [m/s²]")
    ϵ: float = Field(default=1e-9, gt=0, validation_alias=AliasChoices("ϵ", "epsilon"), description="Numerical tolerance, e.g. for determining standstill [-]")


class MotorSettings(_BaseSchema):
    """DC motor and gearing specifications."""
    ω_nom: float = Field(default=168.2, gt=0, validation_alias=AliasChoices("ω_nom", "omega_nom", "w_nom"), description="Nominal rotational speed [rad/s]")
    T_nom: float = Field(default=100.0, gt=0, description="Nominal torque [N·m]")
    Kt: float = Field(default=1.542, gt=0, description="Torque constant = voltage constant [N·m/A = V·s/rad]")
    Ra: float = Field(default=0.083, gt=0, description="Motor internal resistance [Ω]")
    Tf: float = Field(default=1.6818, ge=0, description="Motor friction torque [N·m]")
    i2or: float = Field(default=3.0, gt=0, description="Gear ratio, motor shaft to wheel (input-to-output ratio) [-]")
    η: float = Field(default=0.8, gt=0, le=1, validation_alias=AliasChoices("η", "eta"), description="Gearbox efficiency [-]")
    U_nom: float = Field(default=264.8, gt=0, description="Nominal voltage [V]")
    I_max: float = Field(default=65.9, gt=0, description="Maximum continuous positive current [A]")
    I_min: float = Field(default=-65.9, lt=0, description="Maximum continuous negative current [A]")


class OperatingLimits(_BaseSchema):
    """Operating range."""
    F_max: float = Field(default=936.0, gt=0, description="Vehicle maximum positive force within operating range [N]")
    F_min: float = Field(default=-922.0, lt=0, description="Vehicle maximum negative force within operating range [N]")
    v_work: float = Field(default=4.4, gt=0, description="Vehicle operating speed [m/s]")
    a_max: float = Field(default=1.5, gt=0, description="Maximum acceleration [m/s²]")
    a_min: float = Field(default=-1.5, lt=0, description="Maximum deceleration [m/s²]")
    α_max: float = Field(default=10.0, gt=0, validation_alias=AliasChoices("α_max", "alpha_max"), description="Maximum incline in operating range [°]")
    α_min: float = Field(default=-10.0, lt=0, validation_alias=AliasChoices("α_min", "alpha_min"), description="Minimum incline in operating range [°]")


class InitialConditions(_BaseSchema):
    """Initial simulation states."""
    t0: float = Field(default=0.0, ge=0, description="Initial time [s]")
    x0: float = Field(default=0.0, description="Initial position [m]")
    y0: float = Field(default=0.0, description="Initial height [m]")
    α0: float = Field(default=0.0, validation_alias=AliasChoices("α0", "alpha0"), description="Initial angle [°]")
    v0: float = Field(default=0.0, description="Initial velocity [m/s]")
    a0: float = Field(default=0.0, description="Initial acceleration [m/s²]")


class Params(
    VehicleModel,
    MotorSettings,
    OperatingLimits,
    InitialConditions,
):
    """All physical, motor, limit, and initial parameters."""


class Settings(Params):
    """Simulator and controller settings for `eulers_step_method`."""
    tf: float = Field(..., gt=0, description="End time [s]")
    dt: float = Field(..., gt=0, description="Step size for Euler's method [s]")
    Kp: float = Field(default=0, ge=0, description="Controller proportional gain")
    Ki: float = Field(default=0, ge=0, description="Controller integral gain")
    Kd: float = Field(default=0, ge=0, description="Controller derivative gain")
    u_frac: float = Field(default=1, gt=0, le=1, description="Allowed fraction of nominal voltage U_nom [-]")
    i_frac: float = Field(default=1, gt=0, le=1, description="Allowed fraction of maximum current I_max [-]")
    n_ctrl: int = Field(default=1, gt=0, description="Number of simulation steps per controller sample, `Ts = n_ctrl·dt`)")
    open_loop: bool = Field(default=False, description="Run in open-loop mode")
    vref: float = Field(default=4.4, gt=0, description="Vehicle reference speed, default=4.4 (operating speed) [m/s]")

    @model_validator(mode="after")
    def check_time(self) -> Self:
        if self.dt <= 0 or self.tf <= self.dt:
            raise ValueError(f"Requires 0 < dt < tf (got 0 < {self.dt} < {self.tf})")
        if self.tf <= self.t0:
            raise ValueError(f"Requires tf > t0 (got tf={self.tf}, t0={self.t0})")
        return self

    def get_control_limits(self, mode: ControlMode, v: float = 0.0) -> tuple[float, float]:
        ω = self.i2or * v / self.r
        u_max = self.U_nom * self.u_frac
        i_min = self.I_min * self.i_frac
        i_max = self.I_max * self.i_frac

        if self.open_loop and mode == "voltage":
            return -u_max, u_max

        if self.open_loop and mode == "current":
            return -u_max, u_max


        match mode:
            case "current":
                lo = max(i_min, (-u_max - self.Kt * ω) / self.Ra)
                hi = min(i_max, (u_max - self.Kt * ω) / self.Ra)
                return lo, hi
            case "voltage":
                lo = max(-u_max, self.Kt * ω + self.Ra * i_min)
                hi = min(u_max, self.Kt * ω + self.Ra * i_max)
                return lo, hi
            case _:
                raise ValueError(f"Unknown control mode ¯\\_(ツ)_/¯: {mode}")
