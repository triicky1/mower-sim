#!/usr/bin/env python3
# tests/test_sim.py

import numpy as np
import pytest # pytest-9.2.0

from sim import (
    AMT102VSensor,
    Controller,
    Params,
    RoadProfile,
    Settings,
    VehicleDynamics,
    step_method,
)

ROAD = RoadProfile(
    (20, 0),
    (10, 1),
    (10, 2),
    (10, 3),
    (10, 4),
    (10, 5),
    (50, 0),
    (10, -5),
    (10, -4),
    (10, -3),
    (10, -2),
    (10, -1),
    (20, 0),
)


def test_road_profile():
    """Kontrollerar att lutningen byter vid rätt kant och behåller sista lutningen."""
    road = RoadProfile((10.0, 0.0), (20.0, 5.0), (15.0, -3.0))
    assert np.isclose(road.slope(0.0), 0.0)
    assert np.isclose(road.slope(9.99), 0.0)
    assert np.isclose(road.slope(10.01), np.radians(5.0))
    assert np.isclose(road.slope(29.99), np.radians(5.0))
    assert np.isclose(road.slope(30.01), np.radians(-3.0))
    assert np.isclose(road.slope(500.0), np.radians(-3.0))


def test_static_friction_hold():
    """Står fordonet still och |Fd - F_grav| <= F_roll ska accelerationen vara exakt 0."""
    p = Params()
    dyn = VehicleDynamics(p)
    F_roll = p.Cr * p.m * p.g
    # Drivkraft mindre än rullmotstånd på plan mark vid v = 0
    assert dyn._vehicle_acceleration(v=0.0, α=0.0, Fd=0.5 * F_roll) == 0.0
    # Drivkraft större än rullmotstånd ska ge positiv acceleration
    assert dyn._vehicle_acceleration(v=0.0, α=0.0, Fd=1.5 * F_roll) > 0.0


def test_dynamics_limits_consistency():
    """Verifierar att dynamikmodellerna returnerar fysikaliska gränser i rätt enhet."""
    s = Settings(tf=1, dt=0.1)
    dyn = VehicleDynamics(s)
    # Vid v = 0 (omega = 0)
    lo_i, hi_i = s.get_control_limits('current', v=0)
    assert np.isclose(lo_i, -s.I_max)
    assert np.isclose(hi_i, s.I_max)

    lo_u, hi_u = s.get_control_limits('voltage', v=0)
    # Vid omega = 0 begränsas spänningen av Ra * I_max (~5.47 V) som är lägre än U_nom
    assert hi_u <= s.U_nom
    assert np.isclose(hi_u, min(s.U_nom, s.Ra * s.I_max))


def test_controller_anti_windup_and_derivative():
    """Verifierar att integralen inte skenar vid mättnad och att derivatan uppdateras."""
    pid = Controller(Kp=10.0, Ki=50.0, Kd=1.0, Ts=0.01, enable_history=True)

    # Kör med stort fel mot taket hi = 10.0
    for _ in range(100):
        u = pid.update(setpoint=10.0, measurement=0.0, lo=-10.0, hi=10.0)
        assert u == 10.0

    # Eftersom u_demand > u_command och error > 0 ska anti-windup hålla integral == 0
    assert np.isclose(pid.integral, 0.0)
    assert np.isclose(pid.previous_error, 10.0)
    assert len(pid.timestamps) == 100


def test_sensor_amt102v_standstill():
    sensor = AMT102VSensor(
        PPR=2048,
        wheel_radius=0.2,
        gear_ratio=3.0,
        sample_rate_hz=100.0,
        enable_quantization=True,
    )
    t_vec = np.arange(0.0, 0.5, 0.001)
    v_out = [sensor.measure_velocity(v_true=0.0, t=t) for t in t_vec]
    assert np.all(np.array(v_out) == 0.0)


def test_sensor_amt102v_constant_velocity():
    sensor = AMT102VSensor(
        PPR=2048,
        wheel_radius=0.2,
        gear_ratio=3.0,
        sample_rate_hz=100.0,
        enable_quantization=True,
    )
    v_target = 1.0
    v_meas = []
    for t in np.arange(0.0, 1.0, 0.001):
        v_m = sensor.measure_velocity(v_true=v_target, t=t)
        v_meas.append(v_m)

    # Medelvärdet efter transienten ska matcha target
    v_steady = np.array(v_meas[50:])
    assert abs(np.mean(v_steady) - v_target) < 0.01


def test_sensor_amt102v_disabled_quantization():
    sensor = AMT102VSensor(
        PPR=2048,
        wheel_radius=0.2,
        gear_ratio=3.0,
        sample_rate_hz=100.0,
        enable_quantization=False,
    )
    for t in np.arange(0.0, 0.05, 0.001):
        v = sensor.measure_velocity(v_true=2.5, t=t)
    assert abs(v - 2.5) < 1e-9 # type: ignore


def test_euler_simulation_current_and_voltage():
    """Kör hela simuleringen för både ström- och spänningsstyrning och kontrollerar gränser."""
    cfg = Settings(tf=35.0, dt=0.01, Kp=50.0, Ki=50.0, n_ctrl=1)
    dyn = VehicleDynamics(cfg)

    sim_i = step_method(ROAD, dyn.dynamics_current, cfg)
    sim_u = step_method(ROAD, dyn.dynamics_voltage, cfg)

    for mode_name, sim in [("current", sim_i), ("voltage", sim_u)]:
        # Inga NaN eller Inf
        assert np.all(np.isfinite(sim.v)), f"{mode_name}: v innehåller NaN/Inf"
        assert np.all(np.isfinite(sim.Ia)), f"{mode_name}: Ia innehåller NaN/Inf"
        assert np.all(np.isfinite(sim.Ua)), f"{mode_name}: Ua innehåller NaN/Inf"

        # Ström och spänning måste hållas inom fysiska märkdata
        assert np.max(np.abs(sim.Ia)) <= cfg.I_max + 1e-6, f"{mode_name}: Överskred I_max"
        assert np.max(np.abs(sim.Ua)) <= cfg.U_nom + 1e-6, f"{mode_name}: Överskred U_nom"

        # Fordonet ska nå arbetshastigheten v_work (~4.4 m/s) och ta sig över hela banan (>120 m)
        assert np.isclose(sim.v[-1], cfg.v_work, atol=0.25), f"{mode_name}: Nådde ej v_work"
        assert sim.s[-1] > 120.0, f"{mode_name}: Körde inte hela banan"
        assert sim.x[-1] > 100.0, f"{mode_name}: x-koordinat uppdateras felaktigt"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
