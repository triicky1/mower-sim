# Mower Simulator

2D longitudinal vehicle dynamics simulator for a DC-motor-driven mower using forward Euler integration, discrete PID control, and a CUI AMT102-V quadrature encoder model.

## Installation

```bash
git clone [https://github.com/triicky1/mower-sim](https://github.com/triicky1/mower-sim)
cd mower-sim
pip install -r requirements.txt
```

## Verify that everything works

```bash
pytest
```

## Running the Simulations

```bash
python3 main.py [plots | live]
```

* **`python3 main.py`**: Runs the pre-configured batch simulations (open-loop step response, closed-loop current control, closed-loop voltage control, and sensor quantization/FFT analysis) and saves the figures.
* **`python3 main.py plots`**: Runs the batch simulations and opens the interactive Matplotlib windows.
* **`python3 main.py live`**: Launches the real-time Pygame simulator where you can drive the vehicle over a custom course and switch control modes on the fly.

## Live Mode Controls

| Key | Action |
| :--- | :--- |
| `W` / `S` or `Up` / `Down` | Adjust target speed $v_{\text{ref}}$ (PID) or motor input (Open-Loop) |
| `A` / `D` or `Left` / `Right` | Manually override road slope $\alpha$ within $[\alpha_{\min}, \alpha_{\max}]$ |
| `Space` | Brake (set target speed or manual input to `0`) |
| `M` | Toggle between closed-loop PID and manual Open-Loop control |
| `R` | Reset manual overrides back to automatic road profile and reference ramp |
| `P` | Pause or resume simulation |
| `1` - `4` | Set simulation speed (`0.25x`, `1x`, `2x`, `5x`) |
| `Q` or `Esc` | Quit live mode and display the summary dashboard for the run |

## Custom Simulation Guide

You can build custom courses, tune PID gains, or swap sensor resolutions in Python by combining `RoadProfile`, `Settings`, `VehicleDynamics`, and `step_method`:

```python
import matplotlib.pyplot as plt
from hmi import plot_dashboard
from sim import (
    AMT102VSensor,
    RoadProfile,
    Settings,
    VehicleDynamics,
    step_method,
    summarize,
)

# Define road segments as (length_m, slope_deg) or (length_m, slope_deg, vref_mps)
road = RoadProfile(
    (20.0, 0.0),         # 20 m flat start
    (15.0, 6.0),         # 15 m uphill at 6 degrees
    (25.0, 0.0, 3.5),    # 25 m flat with a 3.5 m/s speed limit
    (15.0, -5.0),        # 15 m downhill at -5 degrees
    (20.0, 0.0),         # 20 m flat finish
)

# Configure solver, controller gains, and physical limits
cfg = Settings(
    tf=30.0,             # Simulation duration [s]
    dt=0.001,            # Euler integration step size [s]
    Kp=100.0,            # Proportional gain
    Ki=300.0,            # Integral gain
    Kd=0.0,              # Derivative gain
    n_ctrl=10,           # Controller runs every 10 steps (Ts = 10 ms)
    vref=4.4,            # Default reference speed [m/s]
)

# Initialize vehicle dynamics and encoder model
dyn = VehicleDynamics(cfg)
encoder = AMT102VSensor(
    wheel_radius=cfg.r,
    gear_ratio=cfg.i2or,
    PPR=2048,            # Supported: 48..2048 (see sim/sensor.py)
    sample_rate_hz=100.0,
)

# Run simulation ("current" or "voltage" control mode)
sim = step_method(road, dyn.dynamics_current, "current", cfg, sensor=encoder)

# Print terminal metrics and plot results
summarize(sim, "Custom Course", cfg)
plot_dashboard(sim, params=cfg, title="Custom Course Simulation")
plt.show()
```

> [!NOTE]
> To see all tunable settings and their descriptions, add
>
> ```python
> # From an initialized instance (shows active values + descriptions)
> for name, field in Settings.model_fields.items():
>     print(f"{name:<8} = {getattr(cfg, name)!r:<10} # {field.description}")
>
> # Directly from the class without instantiating (shows defaults + descriptions)
> for name, field in Settings.model_fields.items():
>     print(f"{name:<8} = {field.default!r:<18} # {field.description}")
> ```

### Control Modes & Options

* **Current vs. Voltage Control**: Pass `dyn.dynamics_current` with `"current"` to regulate armature current $I_a$ (bounded by $\pm I_{\max}$ and back-EMF voltage limits), or `dyn.dynamics_voltage` with `"voltage"` to regulate armature voltage $U_a$ (bounded by $\pm U_{\text{nom}}$ and current limits).
* **Ideal vs. Quantized Sensor**: Pass `sensor=False` to `step_method` to bypass encoder quantization and use exact vehicle velocity feedback.
* **Open-Loop Mode**: Set `open_loop=True` in `Settings` and pass `controller=False` to `step_method` to apply a constant fraction (`u_frac` or `i_frac`) of nominal voltage or current.
