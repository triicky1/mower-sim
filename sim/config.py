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

# TODO: Skriva om alla descriptions på engelska


class _BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def keys(self):
        return type(self).model_fields.keys()

    def __getitem__(self, key):
        return getattr(self, key)


class VehicleModel(_BaseSchema):
    """Fordonets massa och fysikaliska konstanter."""
    m: float = Field(default=250.0, gt=0, description="Fordonets massa [kg]")
    r: float = Field(default=0.2, gt=0, description="Fordonets hjulradie [m]")
    Af: float = Field(default=0.4, gt=0, description="Fordonets tvärsnittsarea, framifrån [m²]")
    Ab: float = Field(default=0.4, gt=0, description="Fordonets tvärsnittsarea, bakifrån [m²]")
    ρ: float = Field(default=1.225, gt=0, validation_alias=AliasChoices("ρ", "rho"), description="Luftens densitet vid marken [kg/m³]")
    Cd: float = Field(default=0.8, gt=0, description="Fordonets dragkoefficient [-]")
    Cr: float = Field(default=0.05, gt=0, description="Fordonets rullmotståndskoefficient [-]")
    μ: float = Field(default=0.6, gt=0, le=1, validation_alias=AliasChoices("μ", "mu"), description="Friktionskoefficient mellan hjul och underlag [-]")
    g: float = Field(default=9.82, gt=0, description="Tyngdaccelerationen [m/s²]")
    ϵ: float = Field(default=1e-9, gt=0, validation_alias=AliasChoices("ϵ", "epsilon"), description="Numerisk tolerans, t.ex. för att avgöra stillastående [-]")


class MotorSettings(_BaseSchema):
    """Likströmsmotor och utväxling, specifikationer."""
    ω_nom: float = Field(default=168.2, gt=0, validation_alias=AliasChoices("ω_nom", "omega_nom", "w_nom"), description="Nominellt varvtal [rad/s]")
    T_nom: float = Field(default=100.0, gt=0, description="Nominellt vridmoment [N·m]")
    Kt: float = Field(default=1.542, gt=0, description="Momentkonstant = spänningskonstant [N·m/A = V·s/rad]")
    Ra: float = Field(default=0.083, gt=0, description="Motorns inre resistans [Ω]")
    Tf: float = Field(default=1.6818, ge=0, description="Motorns friktionsmoment [N·m]")
    i2or: float = Field(default=3.0, gt=0, description="Utväxling, motoraxel till hjul (input-to-output ratio) [-]")
    η: float = Field(default=0.8, gt=0, le=1, validation_alias=AliasChoices("η", "eta"), description="Växellådans verkningsgrad [-]")
    U_nom: float = Field(default=264.8, gt=0, description="Nominell spänning [V]")
    I_max: float = Field(default=65.9, gt=0, description="Maximal kontinuerlig positiv ström [A]")
    I_min: float = Field(default=-65.9, lt=0, description="Maximal kontinuerlig negativ ström [A]")


class OperatingLimits(_BaseSchema):
    """Arbetsområde."""
    F_max: float = Field(default=936.0, gt=0, description="Fordonets maximala positiva kraft inom arbetsområdet [N]")
    F_min: float = Field(default=-922.0, lt=0, description="Fordonets maximala negativa kraft inom arbetsområdet [N]")
    v_work: float = Field(default=4.4, gt=0, description="Fordonets arbetshastighet [m/s]")
    a_max: float = Field(default=1.5, gt=0, description="Maximal acceleration [m/s²]")
    a_min: float = Field(default=-1.5, lt=0, description="Maximal retardation [m/s²]")
    α_max: float = Field(default=10.0, gt=0, validation_alias=AliasChoices("α_max", "alpha_max"), description="Högsta lutning i arbetsområdet [°]")
    α_min: float = Field(default=-10.0, lt=0, validation_alias=AliasChoices("α_min", "alpha_min"), description="Minsta lutning i arbetsområdet [°]")


class InitialConditions(_BaseSchema):
    """Initial simulation states."""
    t0: float = Field(default=0.0, ge=0, description="Starttid [s]")
    x0: float = Field(default=0.0, description="Startläge [m]")
    y0: float = Field(default=0.0, description="Starthöjd [m]")
    α0: float = Field(default=0.0, validation_alias=AliasChoices("α0", "alpha0"), description="Startvinkel [°]")
    v0: float = Field(default=0.0, description="Starthastighet [m/s]")
    a0: float = Field(default=0.0, description="Startacceleration [m/s²]")


class Params(
    VehicleModel,
    MotorSettings,
    OperatingLimits,
    InitialConditions,
):
    """Alla fysikaliska-, motor-, gräns-, och begynnelseparametrar."""


class Settings(Params):
    """Inställningar för simulator och regulator till `eulers_step_method`."""
    tf: float = Field(..., gt=0, description="Sluttid [s]")
    dt: float = Field(..., gt=0, description="Steglängd för Eulers stegmetod [s]")
    Kp: float = Field(default=0, ge=0, description="Regulatorns proportionalförstärkning")
    Ki: float = Field(default=0, ge=0, description="Regulatorns integralförstärkning")
    Kd: float = Field(default=0, ge=0, description="Regulatorns derivataförstärkning")
    u_frac: float = Field(default=1, gt=0, le=1, description="Tillåten andel av nominell spänning U_nom [-]")
    i_frac: float = Field(default=1, gt=0, le=1, description="Tillåten andel av maximal ström I_max [-]")
    n_ctrl: int = Field(default=1, gt=0, description="Antal simuleringssteg per regulatorsampel, `Ts = n_ctrl·dt`)")
    # vref: float = Field(default=4.4, gt=0, description="Fordonets referenshastighet, default=4.4 (arbetshastighet) [m/s]")

    @model_validator(mode="after")
    def check_time(self) -> Self:
        if self.dt <= 0 or self.tf <= self.dt:
            raise ValueError(f"Kräver 0 < dt < tf (fick 0 < {self.dt} < {self.tf})")
        if self.tf <= self.t0:
            raise ValueError(f"Kräver tf > t0 (fick tf={self.tf}, t0={self.t0})")
        return self

    def get_control_limits(self, mode: ControlMode, v: float = 0.0) -> tuple[float, float]:
        ω = self.i2or * v / self.r
        u_max = self.U_nom * self.u_frac
        i_min = self.I_min * self.i_frac
        i_max = self.I_max * self.i_frac

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
                raise ValueError(f"Okänt kontrolläge ¯\\_(ツ)_/¯: {mode}")
