#!/usr/bin/env python3
# main.py

import matplotlib.pyplot as plt

import plotkit as plk
from sim import Params, RoadProfile, Settings, EulerSim, VehicleDynamics, step_method
from hmi import plot_comparison, plot_dashboard, plot_sensor_effect, run_live


plk.mkdir('figures/history')
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

def summarize(res: EulerSim, name: str = "", params: Params = Params()):
    e = res.v - res.vr
    up, down = res.α > 0, res.α < 0
    print(f"{name}: max|I| = {abs(res.Ia).max():.1f} A (I_max = {params.I_max}), "
          f"max|U| = {abs(res.Ua).max():.1f} V (U_nom = {params.U_nom}), "
          f"dip uphill = {e[up].min():.2f} m/s, overshoot downhill = {e[down].max():.2f} m/s, "
          f"distance = {res.s[-1]:.0f} m")

def sim():
    cfg_current = Settings(tf=45.0, dt=0.01, Kp=54.0, Ki=54.0)
    dyn_current = VehicleDynamics(cfg_current)
    sim_current = step_method(ROAD, dyn_current.dynamics_current, "current", cfg_current)

    cfg_voltage = Settings(tf=45.0, dt=0.01, Kp=5.6, Ki=116)
    dyn_voltage = VehicleDynamics(cfg_voltage)
    sim_voltage = step_method(ROAD, dyn_voltage.dynamics_voltage, "voltage", cfg_voltage)

    summarize(sim_current, "Strömstyrning", cfg_current)
    summarize(sim_voltage, "Spänningsstyrning", cfg_voltage)

    # Skapa kompletta 3x2 ingenjörspaneler
    plot_dashboard(
        sim_current,
        params=cfg_current,
        title="Fordonssimulering: Strömstyrning med AMT102-V (2048 PPR)",
        save_path="dashboard_current",
    )
    plot_dashboard(
        sim_voltage,
        params=cfg_voltage,
        title="Fordonssimulering: Spänningsstyrning med AMT102-V (2048 PPR)",
        save_path="dashboard_voltage",
    )

    # Direkt jämförelse mellan ström- och spänningsstyrning
    plot_comparison(
        sim_current,
        sim_voltage,
        params=cfg_current,
        save_path="comparison_modes",
    )

    # Studie av sensorns inverkan (Ideell sensor vs AMT102-V 0.25 deg)
    cfg_ideal = Settings(tf=45.0, dt=0.01, Kp=50.0, Ki=40.0, n_ctrl=1)
    dyn_ideal = VehicleDynamics(cfg_ideal)
    sim_ideal = step_method(ROAD, dyn_ideal.dynamics_current, "current", cfg_ideal)

    plot_sensor_effect(
        sim_ideal,
        sim_current,
        params=cfg_current,
        save_path="sensor_effect",
    )

    plt.show()

def live():
    settings = Settings(dt=0.001, tf=float('inf'), Kp=54.0, Ki=40.0)
    vehicle = VehicleDynamics(settings)
    # road = RoadProfile((40.0, 0.0), (30.0, 6.0), (40.0, -4.0), (50.0, 0.0))
    road = RoadProfile(
        (30.0,   0.0),                                      # startrak
        (20.0,   8.0), (20.0,  -8.0),                       # liten puckel
        (15.0,  20.0), (15.0,  35.0), (10.0,  50.0),        # "loop"-uppfart, allt brantare
        ( 5.0,   0.0),                                      # toppen
        (10.0, -50.0), (15.0, -35.0), (15.0, -20.0),        # "loop"-utfart
        (20.0,   0.0),                                      # andningspaus
        (25.0,  12.0), (25.0, -12.0),
        (25.0,  12.0), (25.0, -12.0),                       # berg-och-dalbana
        (10.0,  40.0), (10.0, -40.0),
        (10.0,  40.0), (10.0, -40.0),                       # vassa hack
        (30.0, -10.0), (30.0,  10.0),                       # dalgång
        (20.0,   0.0),                                      # målrak
    )

    res = run_live(road, vehicle.dynamics_current, "current", settings)

def main():
    live()


if __name__ == "__main__":
    main()
