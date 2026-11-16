#!/usr/bin/env python3
# examples/live.py
"""Real-time simulation over a custom course."""

import matplotlib.pyplot as plt

import plotkit as plk
from hmi import plot_dashboard, run_live
from sim import RoadProfile, Settings, VehicleDynamics


def live(funmode=False, Kp=54.0, Ki=40.0, v_max=float('inf'), dt=0.001):
    plk.apply_style("clean")
    plk.apply_style("grid")

    settings = Settings(dt=dt, tf=float('inf'), Kp=Kp, Ki=Ki, v_work=v_max)
    vehicle = VehicleDynamics(settings)
    road = RoadProfile(
        (30.0,   0.0),                                      # straight start
        (20.0,   8.0), (20.0,  -8.0),                       # small hump
        (15.0,  20.0), (15.0,  35.0), (10.0,  50.0),        # looping ascent, getting steeper
        ( 5.0,   0.0),                                      # the top
        (10.0, -50.0), (15.0, -35.0), (15.0, -20.0),        # looping descent
        (20.0,   0.0),                                      # breather
        (25.0,  12.0), (25.0, -12.0),
        (25.0,  12.0), (25.0, -12.0),                       # roller coaster
        (10.0,  40.0), (10.0, -40.0),
        (10.0,  40.0), (10.0, -40.0),                       # sharp jagged bumps
        (30.0, -10.0), (30.0,  10.0),                       # valley
        (20.0,   0.0),                                      # finish
    )

    res = run_live(road, vehicle.dynamics_current, settings, funmode=funmode)

    plot_dashboard(
        res,
        params=settings,
        title="Live control",
        save_path="livecontrol",
    )
    plt.show()


if __name__ == "__main__":
    live()
