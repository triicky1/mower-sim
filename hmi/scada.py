# hmi/scada.py

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

import plotkit as plk
from sim import Params, EulerSim


COL = dict(x="#0072B2", v="#009E73", a="#D55E00", i="#CC79A7", u="#555555", f="#800815")
LIMIT = dict(color="crimson", ls="--", lw=1.0)


def _find_slope_transitions(res: EulerSim) -> list[float]:
    """Finds the times when the road slope changes."""
    idx = np.where(np.diff(res.α) != 0)[0]
    return [float(res.t[i + 1]) for i in idx]


def _shade_slopes(ax: Axes, res: EulerSim):
    """Subtly shades uphill (warm tone) and downhill (cool tone) in the background."""
    t = res.t
    up = res.α > 1e-4
    down = res.α < -1e-4
    ax.fill_between(t, 0, 1, where=up, color="#D55E00", alpha=0.07, # type: ignore
                    transform=ax.get_xaxis_transform(), label="_nolegend_")
    ax.fill_between(t, 0, 1, where=down, color="#0072B2", alpha=0.07, # type: ignore
                    transform=ax.get_xaxis_transform(), label="_nolegend_")


def plot_transients(res: EulerSim, *, params: Params = Params(), title: str = "", save_path: str | None = None):
    """Compact 2x2 engineering panel for analyzing initial transients."""
    I_max, U_nom, a_max, a_min = params.I_max, params.U_nom, params.a_max, params.a_min
    t = res.t
    t_trans = _find_slope_transitions(res)
    trans_style = dict(color="0.65", ls=":", lw=0.9)

    with plk.style_context("compact"):
        fig, axes = plt.subplots(2, 2, sharex="col", figsize=(11, 7.0), constrained_layout=True)
        ax_v, ax_fa, ax_u, ax_i = axes.ravel()

        # Velocity (actual and reference) & Horizontal position (twin axis)
        ax_v.plot(t, res.vr, "k--", lw=1.1, zorder=2, label=r"Reference $v_\mathrm{ref} (v_\mathrm{work})$")

        ax_x = ax_v.twinx()
        ax_x.grid(False)
        ax_x.spines["top"].set_visible(False)

        plk.draw(
            ax_v, t, res.v, color=COL["v"], lw=1.8, zorder=3,
            label=r"Actual $v$", ylabel="Velocity (m/s)",
            title="Velocity & Position", vlines=t_trans, ref=trans_style,
            legend=False,
        )
        ax_x.plot(t, res.x, color=COL["x"], lw=1.2, ls="-.", zorder=3, label=r"Position $x$")
        ax_x.set_ylabel("Horizontal position $x$ (m)", color=COL["x"])
        ax_x.tick_params(axis="y", labelcolor=COL["x"])
        _shade_slopes(ax_v, res)
        plk.merged_legend(ax_v, ax_x, loc="lower right", ncol=3)

        # Driving force Fd & Acceleration a (twin axis)
        ax_f = ax_fa.twinx()
        ax_f.grid(False)
        ax_f.spines["top"].set_visible(False)

        plk.draw(
            ax_fa, t, res.a, color=COL["a"], lw=1.4, label=r"Acceleration $a$",
            ylabel="Acceleration (m/s²)",
            title="Driving force & Acceleration",
            hlines={a_max: r"$\pm a_\mathrm{max}$", a_min: None}, ref=LIMIT,
            vlines=t_trans, legend=False,
        )
        ax_f.plot(t, res.Fd, color=COL["f"], lw=1.1, ls="-.", label=r"Driving force $F_d$")
        ax_f.set_ylabel("Driving force $F_d$ (N)", color=COL["f"])
        ax_f.tick_params(axis="y", labelcolor=COL["f"])
        plk.merged_legend(ax_fa, ax_f, loc="upper right", ncol=3)
        _shade_slopes(ax_fa, res)

        # Armature voltage Ua with limits
        plk.draw(
            ax_u, t, res.Ua, color=COL["u"], lw=1.3,
            xlabel="Time $t$ (s)", ylabel="Armature voltage $u_a$ (V)",
            title="Motor voltage",
            hlines={U_nom: r"$\pm U_\mathrm{nom}$", -U_nom: None}, ref=LIMIT,
            vlines=t_trans, legend=dict(bbox_to_anchor=(1,0.95)),
        )
        _shade_slopes(ax_u, res)

        # Armature current Ia with limits
        plk.draw(
            ax_i, t, res.Ia, color=COL["i"], lw=1.2,
            xlabel="Time $t$ (s)", ylabel="Armature current $i_a$ (A)",
            title="Motor current",
            hlines={I_max: r"$\pm I_\mathrm{max}$", -I_max: None}, ref=LIMIT,
            vlines=t_trans, legend=dict(bbox_to_anchor=(1,0.95)),
        )
        _shade_slopes(ax_i, res)

        # Apply time axis limits to the shared columns
        ax_u.set_xlim(t[0], t[-1])
        ax_i.set_xlim(t[0], t[-1])

        for ax in (ax_v, ax_fa, ax_u, ax_i):
            ax.tick_params(bottom=True)

        if title:
            fig.suptitle(title, fontsize=16)

        plk.savefig(save_path)
        return fig, axes


def plot_dashboard(res: EulerSim, *, params: Params = Params(), title: str = "", save_path: str | None = None):
    """Complete 3x2 engineering panel adapted for plotkit."""
    I_max, U_nom, a_max, a_min = params.I_max, params.U_nom, params.a_max, params.a_min
    t = res.t
    t_trans = _find_slope_transitions(res)
    trans_style = dict(color="0.65", ls=":", lw=0.9)

    with plk.style_context("compact"):
        fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), constrained_layout=True)
        ax_v, ax_e, ax_i, ax_u, ax_af, ax_road = axes.ravel()

        # 1. Velocity (measured, reference, actual)
        ax_v.plot(t, res.vm, color="0.78", lw=0.6, zorder=1, label=r"Sensor $v_m$")
        ax_v.plot(t, res.vr, "k--", lw=1.1, zorder=2, label=r"Reference $v_\mathrm{ref}$")
        plk.draw(
            ax_v, t, res.v, color=COL["v"], lw=1.8, zorder=3,
            label=r"Actual $v$", ylabel="Velocity (m/s)",
            title="Velocity control", vlines=t_trans, ref=trans_style,
            legend=dict(loc="lower center", ncol=3),
        )
        _shade_slopes(ax_v, res)

        # 2. Control error v - v_ref
        e = res.v - res.vr
        plk.draw(
            ax_e, t, e, color=COL["a"], lw=1.4,
            ylabel=r"Control error $v - v_\mathrm{ref}$ (m/s)",
            title="Velocity deviation", hlines=0.0,
            vlines=t_trans, ref=trans_style,
        )
        _shade_slopes(ax_e, res)

        # Annotate largest dip uphill and overshoot downhill
        if np.any(res.α > 0):
            k_min = int(np.argmin(e))
            ax_e.annotate(
                f"Min: {e[k_min]:+.2f} m/s",
                xy=(t[k_min], e[k_min]), xytext=(t[k_min] + 2.0, e[k_min]),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8), fontsize=8.5,
            )

        # 3. Armature current Ia
        plk.draw(
            ax_i, t, res.Ia, color=COL["i"], lw=1.2,
            ylabel="Armature current $i_a$ (A)", title="Motor current",
            hlines={I_max: r"$\pm I_\mathrm{max}$", -I_max: None}, ref=LIMIT,
            legend=dict(loc="upper right"),
        )
        _shade_slopes(ax_i, res)

        # 4. Armature voltage Ua
        plk.draw(
            ax_u, t, res.Ua, color=COL["u"], lw=1.2,
            ylabel="Armature voltage $u_a$ (V)", title="Motor voltage",
            hlines={U_nom: r"$\pm U_\mathrm{nom}$", -U_nom: None}, ref=LIMIT,
            legend=dict(loc="lower right"),
        )
        _shade_slopes(ax_u, res)

        # 5. Acceleration & Driving force (twin axis with merged_legend)
        ax_f = ax_af.twinx()
        ax_f.grid(False)
        ax_f.spines["top"].set_visible(False)

        plk.draw(
            ax_af, t, res.a, color=COL["a"], lw=1.4, label=r"Acceleration $a$",
            xlabel="Time $t$ (s)", ylabel="Acceleration (m/s²)",
            title="Vehicle dynamics",
            hlines={a_max: r"$\pm a_\mathrm{max}$", a_min: None}, ref=LIMIT,
            legend=False,
        )
        ax_f.plot(t, res.Fd, color=COL["f"], lw=1.1, ls="-.", label=r"Driving force $F_d$")
        ax_f.set_ylabel("Driving force $F_d$ (N)", color=COL["f"])
        ax_f.tick_params(axis="y", labelcolor=COL["f"])
        plk.merged_legend(ax_af, ax_f, loc="lower center", bbox_to_anchor=(0.5, 0.87), ncol=3)
        _shade_slopes(ax_af, res)

        # 6. Spatial path profile y(x) and slope alpha(s)
        ax_alpha = ax_road.twinx()
        ax_alpha.grid(False)
        ax_alpha.spines["top"].set_visible(False)

        ax_road.fill_between(res.s, res.y, res.y.min() - 0.3, color=COL["x"], alpha=0.12)
        plk.draw(
            ax_road, res.s, res.y, color=COL["x"], lw=2.0, label=r"Elevation $y(s)$",
            xlabel="Distance $s$ (m)", ylabel="Elevation $y$ (m)",
            title="Travel distance along road profile", legend=False,
        )
        ax_alpha.step(res.s, np.degrees(res.α), color="0.35", lw=1.2, ls="--",
                      where="post", label=r"Slope $\alpha(s)$")
        ax_alpha.set_ylabel("Slope $\\alpha$ (°)", color="0.35")
        plk.merged_legend(ax_road, ax_alpha, loc="upper right")

        for ax in (ax_v, ax_e, ax_i, ax_u, ax_af):
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(bottom=True)
        ax_road.set_xlim(res.s[0], res.s[-1])
        ax_road.tick_params(bottom=True)

        if title:
            fig.suptitle(title, fontsize=18)

        plk.savefig(save_path)
        return fig, axes


def plot_comparison(
    sim_i: EulerSim,
    sim_u: EulerSim,
    *,
    params: Params = Params(),
    title: str = "Comparison: Current control vs. Voltage control",
    save_path: str | None = None,
):
    """Compares current control and voltage control in 4 shared panels."""
    I_max, U_nom = params.I_max, params.U_nom
    t = sim_i.t
    t_trans = _find_slope_transitions(sim_i)
    trans_style = dict(color="0.65", ls=":", lw=0.9)

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9.5), sharex=True, constrained_layout=True)

    # 1. Velocity
    axes[0].plot(t, sim_i.vr, "k--", lw=1.1, label=r"Reference $v_\mathrm{ref}$")
    axes[0].plot(t, sim_i.v, color=COL["v"], lw=1.8, label="Current control")
    plk.draw(
        axes[0], t, sim_u.v, color=COL["x"], lw=1.6, ls="-.", label="Voltage control",
        ylabel="Velocity (m/s)", title=title, vlines=t_trans, ref=trans_style,
        legend=dict(loc="lower right", ncol=3),
    )
    _shade_slopes(axes[0], sim_i)

    # 2. Control error
    axes[1].plot(t, sim_i.v - sim_i.vr, color=COL["v"], lw=1.5, label="Current control")
    plk.draw(
        axes[1], t, sim_u.v - sim_u.vr, color=COL["x"], lw=1.5, ls="-.", label="Voltage control",
        ylabel=r"Error $v - v_\mathrm{ref}$ (m/s)", hlines=0.0,
        legend=dict(loc="lower right", ncol=2),
    )
    _shade_slopes(axes[1], sim_i)

    # 3. Armature current
    axes[2].plot(t, sim_i.Ia, color=COL["v"], lw=1.2, alpha=0.85, label="Current control")
    plk.draw(
        axes[2], t, sim_u.Ia, color=COL["x"], lw=1.2, alpha=0.85, label="Voltage control",
        ylabel="Current $i_a$ (A)",
        hlines={I_max: r"$\pm I_\mathrm{max}$", -I_max: None}, ref=LIMIT,
        legend=dict(loc="upper right", ncol=3),
    )
    _shade_slopes(axes[2], sim_i)

    # 4. Armature voltage
    axes[3].plot(t, sim_i.Ua, color=COL["v"], lw=1.2, alpha=0.85, label="Current control")
    plk.draw(
        axes[3], t, sim_u.Ua, color=COL["x"], lw=1.2, alpha=0.85, label="Voltage control",
        xlabel="Time $t$ (s)", ylabel="Voltage $u_a$ (V)",
        hlines={U_nom: r"$\pm U_\mathrm{nom}$", -U_nom: None}, ref=LIMIT,
        legend=dict(loc="lower right", ncol=3),
    )
    _shade_slopes(axes[3], sim_i)

    for ax in axes:
        ax.margins(x=0)
        ax.tick_params(bottom=True)

    plk.savefig(save_path)
    return fig, axes


def plot_sensor_effect(
    sim_ideal: EulerSim,
    sim_noisy: EulerSim,
    *,
    params: Params = Params(),
    title: str = "Effect of AMT102-V sensor error and quantization",
    save_path: str | None = None,
):
    """Shows how encoder inaccuracy (0.25°) propagates to the control signal."""
    I_max = params.I_max
    t = sim_noisy.t
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)

    # 1. Measured vs. true velocity
    axes[0].plot(t, sim_noisy.vm, color="0.5", lw=0.7, label=r"Measured $v_m$ (AMT102-V)")
    axes[0].plot(t, sim_ideal.v, "k--", lw=1.2, label=r"Actual $v$ (ideal sensor)")
    plk.draw(
        axes[0], t, sim_noisy.v, color=COL["v"], lw=1.6, label=r"Actual $v$ (with sensor noise)",
        ylabel="Velocity (m/s)", title=title,
        legend=dict(loc="lower center", ncol=3),
    )

    # 2. Measurement error vm - v
    plk.draw(
        axes[1], t, sim_noisy.vm - sim_noisy.v, color=COL["a"], lw=0.7,
        ylabel=r"Measurement error $v_m - v$ (m/s)", hlines=0.0,
    )

    # 3. Resulting current ripple
    axes[2].plot(t, sim_noisy.Ia, color=COL["i"], lw=0.9, alpha=0.8, label="With AMT102-V noise")
    plk.draw(
        axes[2], t, sim_ideal.Ia, color="k", lw=1.4, label="Ideal sensor ($0^\\circ$ error)",
        xlabel="Time $t$ (s)", ylabel="Armature current $i_a$ (A)",
        hlines={I_max: r"$\pm I_\mathrm{max}$", -I_max: None}, ref=LIMIT,
        legend=dict(loc="upper right", ncol=2),
    )

    for ax in axes:
        ax.margins(x=0)
        ax.tick_params(bottom=True)

    plk.savefig(save_path)
    return fig, axes


def _amplitude_spectrum(t: np.ndarray, x: np.ndarray, t_min: float):
    """Single-sided amplitude spectrum (Hann window) for t >= t_min.

    Assumes a fixed time step (Euler). The amplitude has the same unit as x.
    """
    m = t >= t_min
    t, x = t[m], x[m]
    x = x - x.mean()
    w = np.hanning(x.size)
    amp = 2.0 * np.abs(np.fft.rfft(x * w)) / w.sum()
    f = np.fft.rfftfreq(x.size, float(np.mean(np.diff(t))))
    return f, amp


def plot_sensor_fft(
    sim_ideal: EulerSim,
    sim_noisy: EulerSim,
    *,
    params: Params = Params(),
    sensor_fs: float = 100.0,
    t_min: float = 5.0,
    title: str = "Frequency analysis of AMT102-V measurement error",
    save_path: str | None = None,
):
    """Amplitude spectrum for the measurement error and for the armature current.

    sensor_fs : sensor sampling frequency (Hz), sets the x-axis limit (Nyquist)
    t_min     : start time for the analysis window (s), after the acceleration transient
    """
    t = sim_noisy.t
    f_max = sensor_fs / 2.0

    # Shaft frequency at mean velocity in the analysis window: f = v * i2or / (2π r)
    v_cruise = float(np.mean(np.abs(sim_noisy.v[t >= t_min])))
    f_shaft = v_cruise * params.i2or / (2.0 * np.pi * params.r)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), constrained_layout=True)

    # 1. Measurement error vm - v
    f, A = _amplitude_spectrum(t, sim_noisy.vm - sim_noisy.v, t_min)
    plk.draw(
        axes[0], f, A, color=COL["a"], lw=0.9,
        ylabel=r"Amplitude $|v_m - v|$ (m/s)", title=title, legend=False,
    )
    sel = (f > 0) & (f <= f_max)
    k = int(np.argmax(np.where(sel, A, 0.0)))
    axes[0].annotate(
        f"{f[k]:.2f} Hz", xy=(f[k], A[k]), xytext=(f[k] + 0.06 * f_max, A[k]),
        arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8), fontsize=8.5,
    )

    # 2. Armature current with and without sensor error
    f, A_n = _amplitude_spectrum(t, sim_noisy.Ia, t_min)
    _, A_i = _amplitude_spectrum(sim_ideal.t, sim_ideal.Ia, t_min)
    axes[1].plot(f, A_n, color=COL["i"], lw=0.9, alpha=0.85, label="With AMT102-V noise")
    plk.draw(
        axes[1], f, A_i, color="k", lw=1.2, label="Ideal sensor",
        xlabel="Frequency $f$ (Hz)", ylabel=r"Amplitude $|i_a|$ (A)",
        legend=dict(loc="upper right", ncol=2),
    )

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlim(0, f_max)
        ax.tick_params(bottom=True)
        ax.axvline(f_shaft, color="0.5", ls=":", lw=1.0)

    plk.savefig(save_path)
    return fig, axes
