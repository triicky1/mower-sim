#!/usr/bin/env python3
# examples/template.py
"""Template for your own simulations, with every option shown.

Run from the repository root with:  python3 template.py
"""

import matplotlib.pyplot as plt

from hmi import plot_dashboard, run_live
from sim import (
    AMT102VSensor,
    Controller,
    RoadProfile,
    Settings,
    VehicleDynamics,
    step_method,
    summarize,
)


def main():
    # Road segments: (length_m, slope_deg) or (length_m, slope_deg, vref_mps)
    road = RoadProfile(
        (20.0, 0.0),         # 20 m flat start
        (15.0, 6.0),         # 15 m uphill at 6 degrees
        (25.0, 0.0, 3.5),    # 25 m flat with a 3.5 m/s speed limit
        (15.0, -5.0),        # 15 m downhill at -5 degrees
        (20.0, 0.0),         # 20 m flat finish
    )

    # Solver and physical settings
    cfg = Settings(
        tf=30.0,             # Simulation duration [s]
        dt=0.001,            # Euler integration step size [s]
        n_ctrl=10,           # Controller runs every 10 steps (Ts = 10 ms)
        vref=4.4,            # Default reference speed [m/s]
    )

    # Standalone controller, it does not depend on the rest of the package.
    # Ts must match the controller sample time: n_ctrl * dt.
    # If `controller=` is left out of `step_method`, the gains in `Settings` are used instead.
    ctrl = Controller(
        Kp=100.0,
        Ki=300.0,
        Kd=0.0,
        Ts=cfg.n_ctrl * cfg.dt,
    )

    # Vehicle dynamics and encoder model
    dyn = VehicleDynamics(cfg)
    encoder = AMT102VSensor(
        wheel_radius=cfg.r,
        gear_ratio=cfg.i2or,
        PPR=2048,
        sample_rate_hz=100.0,
    )

    # Run simulation ("current" or "voltage" control mode)
    sim = step_method(
        road=road,
        f=dyn.dynamics_current,  # or dyn.dynamics_voltage, with mode="voltage"
        settings=cfg,
        controller=ctrl,         # Controller | bool. True: PI(D) parameters from `cfg`, False: open-loop
        sensor=encoder,          # AMT102VSensor | bool. True: default sensor, False: no quantization
        # stop_at_end=False,     # bool. True: brake when the simulation reaches 92% of its duration
        # silent=False,          # bool. True: suppress simulation information
    )


    # Print terminal metrics and plot results
    summarize(sim, "Custom Course", cfg)
    plot_dashboard(sim, params=cfg, title="Custom Course Simulation")

    # Optional: drive the same road yourself in the real-time simulator (opens a Pygame window)
    # res = run_live(road, dyn.dynamics_current, cfg)
    # plot_dashboard(res, params=cfg, title="Live control")

    # Optional: list all settings with their descriptions
    # print_default_settings()
    # print_active_settings(cfg)


def print_default_settings():
    """Print the defaults and descriptions straight from the class."""
    for name, field in Settings.model_fields.items():
        print(f"{name:<8} = {field.default!r:<18} # {field.description}")


def print_active_settings(cfg: Settings):
    """Print the active values and descriptions of an instance."""
    for name, field in Settings.model_fields.items():
        print(f"{name:<8} = {getattr(cfg, name)!r:<10} # {field.description}")


if __name__ == "__main__":
    main()
    plt.show()
