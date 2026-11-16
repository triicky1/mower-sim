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

def _get_mode(f: DynamicsModel) -> ControlMode:
    match f.__name__:
        case "dynamics_current":
            mode = "current"
        case "dynamics_voltage":
            mode = "voltage"
        case _:
            raise ValueError(f"Dynamics model '{f.__name__}' not supported")
    return mode


def step_method(
    road: RoadProfile,
    f: DynamicsModel,
    settings: Settings,
    controller: Controller | bool = True,
    sensor: AMT102VSensor | bool = True,
    stop_at_end: bool = False,
    silent: bool = False
) -> EulerSim:
    """Euler step method for vehicle with DC motor."""

    mode = _get_mode(f)

    t0, tf, dt = settings.t0, settings.tf, settings.dt
    x0, y0, α0, v0, a0 = settings.x0, settings.y0, settings.α0, settings.v0, settings.a0
    Kp, Ki, Kd, n_ctrl = settings.Kp, settings.Ki, settings.Kd, settings.n_ctrl
    r, i2or = settings.r, settings.i2or
    lo_nom, hi_nom = settings.get_control_limits(mode, 0)
    vref = settings.vref or settings.v_work
    deramp_ref = VehicleDynamics(settings).deramp_ref
    ramp_ref = VehicleDynamics(settings).ramp_ref

    N = round((tf - t0)/dt)             # Number of steps
    Ts = n_ctrl * dt                    # Controller sample time

    manual = True if sensor is False and controller is False else False

    # controller: True -> Default, False -> None
    if controller is True:
        controller = Controller(Kp, Ki, Kd, Ts)
        controller.reset()
    elif isinstance(controller, Controller):
        controller.reset()

    # Sensor: True -> Default, False -> Disable quantization
    if isinstance(sensor, AMT102VSensor):
        sensor.reset()
    else:
        # sensor is always enabled, simulations where its set to false turns of quantization.
        sensor = AMT102VSensor(wheel_radius=r, gear_ratio=i2or, enable_quantization=sensor)

    # Initialize empty vectors to store the steps in
    t = np.linspace(t0, tf, N+1)
    s, x, y, α, v, vm, vr, a, Ia, Ua, Fd = np.zeros((11, N + 1))
    x[0], y[0], α[0], v[0], a[0] = x0, y0, α0, v0, a0

    cmd = 0.0
    t_break: float | None = None
    v_break: float | None = None

    if not silent:
        print_welcome(settings, controller or None, sensor or None)

    # Start Euler's step method
    for k in range(N + 1):
        α[k] = road.slope(s[k])

        # Inside the simulation loop
        if stop_at_end and t[k] >= t0 + 0.92 * (tf - t0):
            t_break = t_break or t[k]
            v_break = v_break or vr[k - 1] if k > 0 else ramp_ref(t[0])
            vr[k] = deramp_ref(t[k], t_break, v_break) # type: ignore
        else:
            vr[k] = ramp_ref(t[k]) if not manual else vref

        # The sensor is read "continuously" (every step) unless set to False
        vm[k] = sensor.measure_velocity(v[k], t[k])

        lo, hi = settings.get_control_limits(mode, v[k])

        if not controller:
            # Open-loop control
            cmd = lo if v[k] < 0.0 else hi

        elif k % n_ctrl == 0:
            # New controller sample
            cmd = controller.update(vr[k], vm[k], lo, hi)

        # Limits also apply between samples
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
    settings: Settings,
    controller: Controller | bool = True,
    sensor: AMT102VSensor | bool = True,
) -> Generator[EulerStep, StepInput, EulerSim]:
    """Euler step method as a generator for infinite real-time simulation."""
    mode = _get_mode(f)

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

            # Calculate speed-dependent control limits from Settings
            lo, hi = settings.get_control_limits(mode, v_k)

            if u_manual is not None:
                # Manual control
                u_cmd = float(u_manual)
                if controller:
                    controller.previous_error = 0.0
            elif controller:
                # New controller sample
                if k % n_ctrl == 0:
                    u_cmd = float(controller.update(vr_k, vm_k, lo, hi))
            else:
                # Open-loop control
                u_cmd = lo if v_k <= 0.0 else hi

            u_k = float(np.clip(u_cmd, lo, hi))
            a_k, Ia_k, Ua_k, Fd_k = f(v_k, u_k, α_k)

            step = EulerStep(
                t=t_k, s=s_k, x=x_k, y=y_k, α=α_k, v=v_k, vm=vm_k,
                vr=vr_k, a=a_k, Ia=Ia_k, Ua=Ua_k, Fd=Fd_k,
            )
            history.append(step)

            cmd = yield step

            # Explicit Euler: positions use v_k from this step, then v is updated
            s_k = s_k + v_k * dt
            x_k = x_k + v_k * np.cos(α_k) * dt
            y_k = y_k + v_k * np.sin(α_k) * dt
            v_k = v_k + a_k * dt

    except (GeneratorExit, StopIteration):
        pass

    if not history:
        empty = np.empty(0, dtype=float)
        return EulerSim(*(empty for _ in range(12)))

    return EulerSim(*(np.asarray(col, dtype=float) for col in zip(*history)))
