#!/usr/bin/env python3
# sim/euler.py

import numpy as np
import itertools
from collections.abc import Generator
from .config import ControlMode, DynamicsModel, Settings
from .controller import Controller
from .dynamics import VehicleDynamics
from .road import RoadProfile
from .sensor import AMT102VSensor
from .types import EulerSim, EulerStep, LiveCommand
from .write import print_welcome

type StepInput = float | LiveCommand | None


def step_method(
    road: RoadProfile,
    f: DynamicsModel,
    mode: ControlMode,
    settings: Settings,
    controller: Controller | bool = True,
    sensor: AMT102VSensor | bool = True,
) -> EulerSim:
    """Euler stegmetod för fordon med likströmsmotor."""
    ART = open("figures/.wojak", encoding="utf-8").read()
    print_welcome(settings, sensor if isinstance(sensor, AMT102VSensor) else None, art=ART)

    t0, tf, dt = settings.t0, settings.tf, settings.dt
    x0, y0, α0, v0, a0 = settings.x0, settings.y0, settings.α0, settings.v0, settings.a0
    Kp, Ki, Kd, n_ctrl = settings.Kp, settings.Ki, settings.Kd, settings.n_ctrl
    r, i2or = settings.r, settings.i2or
    lo_nom, hi_nom = settings.get_control_limits(mode, 0)
    ramp_ref = VehicleDynamics(settings).ramp_ref

    # TODO: Make controller optional...
    # TODO: Make sensor optional... (pass both explicitly)

    N = round((tf - t0)/dt)             # Antal steg
    Ts = n_ctrl * dt                    # Regulatorns sampeltid

    if controller is True:
        controller = Controller(Kp, Ki, Kd, Ts)

    # https://www.digikey.se/sv/products/detail/same-sky-formerly-cui-devices/AMT102-V/827015
    if sensor is True:
        # Initialisera default sensor
        sensor = AMT102VSensor(
            PPR=2048,
            wheel_radius=r,
            gear_ratio=i2or,
            sample_rate_hz=1 / dt,
        )
    elif isinstance(sensor, AMT102VSensor):
        # Nollställ given sensor
        sensor.reset()

    # Initialisera tomma vektorer att spara stegen i
    t = np.linspace(t0, tf, N+1)
    s, x, y, α, v, vm, vr, a, Ia, Ua, Fd = np.zeros((11, N + 1))
    x[0], y[0], α[0], v[0], a[0] = x0, y0, α0, v0, a0

    cmd = 0.0

    # Påbörja Eulers steg metod
    for k in range(N + 1):
        # TODO: Lägg till önskade hastigheter i RoadProfile segmenten
        α[k] = road.slope(s[k])
        vr[k] = ramp_ref(t[k])

        # Sensorn avläses "kontinuerligt" (varje steg) om inte satt till False
        vm[k] = sensor.measure_velocity(v[k], t[k]) if sensor else v[k]

        lo, hi = settings.get_control_limits(mode, v[k])

        if controller:
            # Nytt regulator sampel
            if k % n_ctrl == 0:
                cmd = controller.update(vr[k], vm[k], lo, hi)
        else:
            # Eller styrning med open-loop
            cmd = lo if v[k] <= 0.0 else hi

        # Gränserna gäller även mellan sampel
        u_k = float(np.clip(cmd, lo, hi))


        a[k], Ia[k], Ua[k], Fd[k] = f(v[k], u_k, α[k])

        if k == N: break

        v[k+1] = v[k] + a[k] * dt
        s[k+1] = s[k] + v[k] * dt
        x[k+1] = x[k] + v[k] * np.cos(α[k]) * dt
        y[k+1] = y[k] + v[k] * np.sin(α[k]) * dt

    return EulerSim(t, s, x, y, α, v, vm, vr, a, Ia, Ua, Fd)


def step_method_forever(
    road: RoadProfile,
    f: DynamicsModel,
    mode: ControlMode,
    settings: Settings,
    controller: Controller | bool = True,
    sensor: AMT102VSensor | bool = True,
) -> Generator[EulerStep, StepInput, EulerSim]:
    """Euler stegmetod som generator för oändlig realtidssimulering."""
    t0, dt = settings.t0, settings.dt
    s_k = 0.0
    x_k, y_k, α_k, v_k, a_k = (
        settings.x0,
        settings.y0,
        settings.α0,
        settings.v0,
        settings.a0,
    )
    Kp, Ki, Kd, n_ctrl = settings.Kp, settings.Ki, settings.Kd, settings.n_ctrl
    r, i2or = settings.r, settings.i2or
    ramp_ref = VehicleDynamics(settings).ramp_ref

    Ts = n_ctrl * dt

    if controller is True:
        controller = Controller(Kp, Ki, Kd, Ts)

    if sensor is True:
        sensor = AMT102VSensor(
            PPR=2048,
            wheel_radius=r,
            gear_ratio=i2or,
            sample_rate_hz=1 / dt,
        )
    elif isinstance(sensor, AMT102VSensor):
        sensor.reset()

    history: list[EulerStep] = []
    u_cmd = 0.0
    cmd: StepInput = None

    try:
        for k in itertools.count():
            t_k = t0 + k * dt

            if isinstance(cmd, LiveCommand):
                α_k = cmd.α if cmd.α is not None else road.slope(s_k)
                vr_k = cmd.vr if cmd.vr is not None else ramp_ref(t_k, road.vref(s_k))
                u_manual = cmd.u_manual
            elif isinstance(cmd, (int, float)):
                α_k = road.slope(s_k)
                vr_k = float(cmd)
                u_manual = None
            else:
                α_k = road.slope(s_k)
                vr_k = ramp_ref(t_k, road.vref(s_k))
                u_manual = None

            vm_k = sensor.measure_velocity(v_k, t_k) if sensor else v_k

            # Beräkna hastighetsberoende styrgränser från Settings
            lo, hi = settings.get_control_limits(mode, v_k)

            if u_manual is not None:
                u_cmd = float(u_manual)
                if controller:
                    controller.previous_error = 0.0
            elif controller:
                if k % n_ctrl == 0:
                    u_cmd = float(controller.update(vr_k, vm_k, lo, hi))
            else:
                u_cmd = lo if v_k <= 0.0 else hi

            u_k = float(np.clip(u_cmd, lo, hi))
            a_k, Ia_k, Ua_k, Fd_k = f(v_k, u_k, α_k)

            step = EulerStep(
                t=t_k,
                s=s_k,
                x=x_k,
                y=y_k,
                α=α_k,
                v=v_k,
                vm=vm_k,
                vr=vr_k,
                a=a_k,
                Ia=Ia_k,
                Ua=Ua_k,
                Fd=Fd_k,
            )
            history.append(step)

            cmd = yield step

            v_k = v_k + a_k * dt
            s_k = s_k + v_k * dt
            x_k = x_k + v[k] * np.cos(α_k) * dt if False else x_k + v_k * np.cos(α_k) * dt
            y_k = y_k + v_k * np.sin(α_k) * dt

    except (GeneratorExit, StopIteration):
        pass

    if not history:
        empty = np.empty(0, dtype=float)
        return EulerSim(*(empty for _ in range(12)))

    return EulerSim(*(np.asarray(col, dtype=float) for col in zip(*history)))
