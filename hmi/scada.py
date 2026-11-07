# viz/scada.py

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

import plotkit as plk
from sim import Params, EulerSim


COL = dict(x="#0072B2", v="#009E73", a="#D55E00", i="#CC79A7", u="#555555", f="#800815")
LIMIT = dict(color="crimson", ls="--", lw=1.0)


def _find_slope_transitions(res: EulerSim) -> list[float]:
    """Hittar tidpunkterna då väglutningen ändras."""
    idx = np.where(np.diff(res.α) != 0)[0]
    return [float(res.t[i + 1]) for i in idx]


def _shade_slopes(ax: Axes, res: EulerSim):
    """Skuggar uppförsbacke (varm ton) och nedförsbacke (kall ton) subtilt i bakgrunden."""
    t = res.t
    up = res.α > 1e-4
    down = res.α < -1e-4
    ax.fill_between(t, 0, 1, where=up, color="#D55E00", alpha=0.07, # type: ignore
                    transform=ax.get_xaxis_transform(), label="_nolegend_")
    ax.fill_between(t, 0, 1, where=down, color="#0072B2", alpha=0.07, # type: ignore
                    transform=ax.get_xaxis_transform(), label="_nolegend_")


def plot_dashboard(res: EulerSim, *, params: Params = Params(), title: str = "", save_path: str | None = None):
    """Komplett 3x2 ingenjörspanel anpassad för plotkit."""
    I_max, U_nom, a_max, a_min = params.I_max, params.U_nom, params.a_max, params.a_min
    t = res.t
    t_trans = _find_slope_transitions(res)
    trans_style = dict(color="0.65", ls=":", lw=0.9)

    with plk.style_context("compact"):
        fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), constrained_layout=True)
        ax_v, ax_e, ax_i, ax_u, ax_af, ax_road = axes.ravel()

        # 1. Hastighet (uppmätt, referens, verklig)
        ax_v.plot(t, res.vm, color="0.78", lw=0.6, zorder=1, label=r"Sensor $v_m$")
        ax_v.plot(t, res.vr, "k--", lw=1.1, zorder=2, label=r"Referens $v_\mathrm{ref}$")
        plk.draw(
            ax_v, t, res.v, color=COL["v"], lw=1.8, zorder=3,
            label=r"Verklig $v$", ylabel="Hastighet (m/s)",
            title="Hastighetsreglering", vlines=t_trans, ref=trans_style,
            legend=dict(loc="lower right", ncol=3),
        )
        _shade_slopes(ax_v, res)

        # 2. Reglerfel v - v_ref
        e = res.v - res.vr
        plk.draw(
            ax_e, t, e, color=COL["a"], lw=1.4,
            ylabel=r"Reglerfel $v - v_\mathrm{ref}$ (m/s)",
            title="Hastighetsavvikelse", hlines=0.0,
            vlines=t_trans, ref=trans_style,
        )
        _shade_slopes(ax_e, res)

        # Annotera största dipp i uppförsbacke och översläng i nedförsbacke
        if np.any(res.α > 0):
            k_min = int(np.argmin(e))
            ax_e.annotate(
                f"Min: {e[k_min]:+.2f} m/s",
                xy=(t[k_min], e[k_min]), xytext=(t[k_min] + 2.0, e[k_min]),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8), fontsize=8.5,
            )

        # 3. Ankarström Ia
        plk.draw(
            ax_i, t, res.Ia, color=COL["i"], lw=1.2,
            ylabel="Ankarström $i_a$ (A)", title="Motorström",
            hlines={I_max: r"$\pm I_\mathrm{max}$", -I_max: None}, ref=LIMIT,
            legend=dict(loc="upper right"),
        )
        _shade_slopes(ax_i, res)

        # 4. Ankarspänning Ua
        plk.draw(
            ax_u, t, res.Ua, color=COL["u"], lw=1.2,
            ylabel="Ankarspänning $u_a$ (V)", title="Motorspänning",
            hlines={U_nom: r"$\pm U_\mathrm{nom}$", -U_nom: None}, ref=LIMIT,
            legend=dict(loc="lower right"),
        )
        _shade_slopes(ax_u, res)

        # 5. Acceleration & Drivkraft (tvillingaxel med merged_legend)
        ax_f = ax_af.twinx()
        ax_f.grid(False)
        ax_f.spines["top"].set_visible(False)

        plk.draw(
            ax_af, t, res.a, color=COL["a"], lw=1.4, label=r"Acceleration $a$",
            xlabel="Tid $t$ (s)", ylabel="Acceleration (m/s²)",
            title="Fordonsdynamik",
            hlines={a_max: r"$\pm a_\mathrm{max}$", a_min: None}, ref=LIMIT,
            legend=False,
        )
        ax_f.plot(t, res.Fd, color=COL["f"], lw=1.1, ls="-.", label=r"Drivkraft $F_d$")
        ax_f.set_ylabel("Drivkraft $F_d$ (N)", color=COL["f"])
        ax_f.tick_params(axis="y", labelcolor=COL["f"])
        plk.merged_legend(ax_af, ax_f, loc="upper right", ncol=2)
        _shade_slopes(ax_af, res)

        # 6. Rumslig banprofil y(x) och lutning alpha(s)
        ax_alpha = ax_road.twinx()
        ax_alpha.grid(False)
        ax_alpha.spines["top"].set_visible(False)

        ax_road.fill_between(res.s, res.y, res.y.min() - 0.3, color=COL["x"], alpha=0.12)
        plk.draw(
            ax_road, res.s, res.y, color=COL["x"], lw=2.0, label=r"Höjd $y(s)$",
            xlabel="Tillryggalagd sträcka $s$ (m)", ylabel="Höjd $y$ (m)",
            title="Vägprofil i rumskoordinater", legend=False,
        )
        ax_alpha.step(res.s, np.degrees(res.α), color="0.35", lw=1.2, ls="--",
                      where="post", label=r"Lutning $\alpha(s)$")
        ax_alpha.set_ylabel("Lutning $\\alpha$ (°)", color="0.35")
        plk.merged_legend(ax_road, ax_alpha, loc="upper right")

        for ax in (ax_v, ax_e, ax_i, ax_u, ax_af):
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(bottom=True)
        ax_road.set_xlim(res.s[0], res.s[-1])
        ax_road.tick_params(bottom=True)

        if title:
            fig.suptitle(title, fontsize=14)

        plk.savefig(save_path)
        return fig, axes


def plot_comparison(
    sim_i: EulerSim,
    sim_u: EulerSim,
    *,
    params: Params = Params(),
    title: str = "Jämförelse: Strömstyrning vs. Spänningsstyrning",
    save_path: str | None = None,
):
    """Jämför strömstyrning och spänningsstyrning i 4 delade paneler."""
    I_max, U_nom = params.I_max, params.U_nom
    t = sim_i.t
    t_trans = _find_slope_transitions(sim_i)
    trans_style = dict(color="0.65", ls=":", lw=0.9)

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9.5), sharex=True, constrained_layout=True)

    # 1. Hastighet
    axes[0].plot(t, sim_i.vr, "k--", lw=1.1, label=r"Referens $v_\mathrm{ref}$")
    axes[0].plot(t, sim_i.v, color=COL["v"], lw=1.8, label="Strömstyrning")
    plk.draw(
        axes[0], t, sim_u.v, color=COL["x"], lw=1.6, ls="-.", label="Spänningsstyrning",
        ylabel="Hastighet (m/s)", title=title, vlines=t_trans, ref=trans_style,
        legend=dict(loc="lower right", ncol=3),
    )
    _shade_slopes(axes[0], sim_i)

    # 2. Reglerfel
    axes[1].plot(t, sim_i.v - sim_i.vr, color=COL["v"], lw=1.5, label="Strömstyrning")
    plk.draw(
        axes[1], t, sim_u.v - sim_u.vr, color=COL["x"], lw=1.5, ls="-.", label="Spänningsstyrning",
        ylabel=r"Fel $v - v_\mathrm{ref}$ (m/s)", hlines=0.0,
        legend=dict(loc="lower right", ncol=2),
    )
    _shade_slopes(axes[1], sim_i)

    # 3. Ankarström
    axes[2].plot(t, sim_i.Ia, color=COL["v"], lw=1.2, alpha=0.85, label="Strömstyrning")
    plk.draw(
        axes[2], t, sim_u.Ia, color=COL["x"], lw=1.2, alpha=0.85, label="Spänningsstyrning",
        ylabel="Ström $i_a$ (A)",
        hlines={I_max: r"$\pm I_\mathrm{max}$", -I_max: None}, ref=LIMIT,
        legend=dict(loc="upper right", ncol=3),
    )
    _shade_slopes(axes[2], sim_i)

    # 4. Ankarspänning
    axes[3].plot(t, sim_i.Ua, color=COL["v"], lw=1.2, alpha=0.85, label="Strömstyrning")
    plk.draw(
        axes[3], t, sim_u.Ua, color=COL["x"], lw=1.2, alpha=0.85, label="Spänningsstyrning",
        xlabel="Tid $t$ (s)", ylabel="Spänning $u_a$ (V)",
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
    title: str = "Effekt av AMT102-V givarfel och kvantisering",
    save_path: str | None = None,
):
    """Visar hur pulsgivarens onoggrannhet (0.25°) fortplantas till styrsignalen."""
    I_max = params.I_max
    t = sim_noisy.t
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)

    # 1. Uppmätt vs sann hastighet
    axes[0].plot(t, sim_noisy.vm, color="0.75", lw=0.7, label=r"Uppmätt $v_m$ (AMT102-V)")
    axes[0].plot(t, sim_ideal.v, "k--", lw=1.2, label=r"Verklig $v$ (ideell givare)")
    plk.draw(
        axes[0], t, sim_noisy.v, color=COL["v"], lw=1.6, label=r"Verklig $v$ (med givarbrus)",
        ylabel="Hastighet (m/s)", title=title,
        legend=dict(loc="lower right", ncol=3),
    )

    # 2. Mätfel vm - v
    plk.draw(
        axes[1], t, sim_noisy.vm - sim_noisy.v, color=COL["a"], lw=0.7,
        ylabel=r"Mätfel $v_m - v$ (m/s)", hlines=0.0,
    )

    # 3. Resulterande strömrippel
    axes[2].plot(t, sim_noisy.Ia, color=COL["i"], lw=0.9, alpha=0.8, label="Med AMT102-V brus")
    plk.draw(
        axes[2], t, sim_ideal.Ia, color="k", lw=1.4, label="Ideell givare ($0^\\circ$ fel)",
        xlabel="Tid $t$ (s)", ylabel="Ankarström $i_a$ (A)",
        hlines={I_max: r"$\pm I_\mathrm{max}$", -I_max: None}, ref=LIMIT,
        legend=dict(loc="upper right", ncol=2),
    )

    for ax in axes:
        ax.margins(x=0)
        ax.tick_params(bottom=True)

    plk.savefig(save_path)
    return fig, axes
