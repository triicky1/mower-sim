#!/usr/bin/env python3
# examples/batch.py
"""Batch simulations: open-loop step responses, current/voltage control and sensor analysis."""

import matplotlib.pyplot as plt

import plotkit as plk
from hmi import plot_comparison, plot_dashboard, plot_sensor_effect, plot_transients, plot_sensor_fft
from sim import RoadProfile, Settings, VehicleDynamics, step_method, summarize


def main():
    plk.apply_style("clean")
    plk.apply_style("grid")

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

    cfg_open_loop_voltage = Settings(tf=0.5, dt=0.001, u_frac=0.5, open_loop=True)
    dyn_open_loop_voltage = VehicleDynamics(cfg_open_loop_voltage)
    sim_open_loop_voltage = step_method(
        RoadProfile((10,0)), # short road at 10 m for complete step response saturation without limits
        dyn_open_loop_voltage.dynamics_voltage,
        cfg_open_loop_voltage,
        controller=False,
        sensor=False
    )
    summarize(sim_open_loop_voltage, "Open-loop")

    cfg_open_loop_current = Settings(tf=100, dt=0.001, i_frac=0.5, open_loop=True)
    dyn_open_loop_current = VehicleDynamics(cfg_open_loop_current)
    sim_open_loop_current = step_method(
        RoadProfile((5000,0)), # long straight road at 5 km for complete step response saturation without limits
        dyn_open_loop_current.dynamics_current,
        cfg_open_loop_current,
        controller=False,
        sensor=False
    )
    summarize(sim_open_loop_current, "Open-loop")

    # Plot the transients for task 5.
    plot_transients(
        sim_open_loop_voltage,
        params=cfg_open_loop_voltage,
        title=r"Open-loop voltage step ($u_a = 0.5\,U_{\mathrm{nom}}$)",
        save_path="dashboard_open_loop_5a",
    )
    plot_transients(
        sim_open_loop_current,
        params=cfg_open_loop_current,
        title=r"Open-loop current step ($i_a = 0.5\,I_{\mathrm{max}}$)",
        save_path="dashboard_open_loop_5b",
    )

    cfg_current = Settings(tf=45.0, dt=0.001, Kp=100, Ki=300)
    dyn_current = VehicleDynamics(cfg_current)
    sim_current = step_method(
        ROAD,
        dyn_current.dynamics_current,
        cfg_current,
        stop_at_end=True
    )
    summarize(sim_current, "Current control", cfg_current)

    cfg_voltage = Settings(tf=45.0, dt=0.001, Kp=5.6, Ki=116)
    dyn_voltage = VehicleDynamics(cfg_voltage)
    sim_voltage = step_method(
        ROAD,
        dyn_voltage.dynamics_voltage,
        cfg_voltage,
        stop_at_end=True
    )
    summarize(sim_voltage, "Voltage control", cfg_voltage)


    # Plot 3x2 engineering panels
    plot_dashboard(
        sim_current,
        params=cfg_current,
        title="Vehicle simulation: Current control with AMT102-V (2048 PPR)",
        save_path="dashboard_current",
    )
    plot_dashboard(
        sim_voltage,
        params=cfg_voltage,
        title="Vehicle simulation: Voltage control with AMT102-V (2048 PPR)",
        save_path="dashboard_voltage",
    )

    # Direct comparison between current and voltage control
    plot_comparison(
        sim_current,
        sim_voltage,
        params=cfg_current,
        save_path="comparison_modes",
    )

    # Visualize the impact from the sensor (ideal sensor vs AMT102-V with quantization)
    cfg_compare = Settings(tf=45.0, dt=0.001, Kp=100, Ki=300)
    dyn_compare = VehicleDynamics(cfg_compare)

    # no breaking in the end to get accurate simulations
    sim_compare = step_method(
        ROAD,
        dyn_compare.dynamics_current,
        cfg_compare,
        sensor=True,
        silent=True
    )
    sim_ideal = step_method(
        ROAD,
        dyn_compare.dynamics_current,
        cfg_compare,
        sensor=False
    )
    summarize(sim_ideal, "Ideal sensor with current control", cfg_compare)

    plot_sensor_effect(
        sim_ideal,
        sim_compare,
        params=cfg_compare,
        save_path="sensor_effect",
    )

    plot_sensor_fft(sim_ideal, sim_compare, params=cfg_compare, sensor_fs=100.0, save_path="sensor_fft.pdf")


if __name__ == "__main__":
    main()
    plt.show()
