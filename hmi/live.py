# hmi/live.py

from collections import deque
import math
import numpy as np
import pygame

from sim import (
    DynamicsModel,
    EulerSim,
    EulerStep,
    LiveCommand,
    RoadProfile,
    Settings,
    step_method_forever,
)
from sim.euler import _get_mode

BG_COLOR = (22, 26, 29)
PANEL_BG = (30, 35, 38)
GRID_COLOR = (48, 56, 60)
TEXT_COLOR = (211, 198, 170)
MUTED_TEXT = (133, 146, 137)
ACCENT_GREEN = (167, 192, 128)
ACCENT_BLUE = (127, 187, 179)
ACCENT_ORANGE = (230, 152, 117)
ACCENT_RED = (230, 126, 128)
ACCENT_PURPLE = (214, 153, 182)


def _draw_strip_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    t_buf: deque[float],
    series: list[tuple[deque[float], tuple[int, int, int], int]],
    y_min: float,
    y_max: float,
    title: str,
    unit: str,
    font: pygame.font.Font,
    limit_lines: list[float] | None = None,
) -> None:
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=4)
    pygame.draw.rect(surface, GRID_COLOR, rect, width=1, border_radius=4)

    if y_max <= y_min:
        y_max = y_min + 1.0

    def val_to_y(val: float) -> int:
        norm = (val - y_min) / (y_max - y_min)
        norm = max(0.0, min(1.0, norm))
        return int(rect.bottom - 6 - norm * (rect.height - 26))

    if y_min < 0.0 < y_max:
        y0 = val_to_y(0.0)
        pygame.draw.line(surface, GRID_COLOR, (rect.left + 4, y0), (rect.right - 4, y0), 1)

    if limit_lines:
        for lim in limit_lines:
            if y_min <= lim <= y_max:
                ylim = val_to_y(lim)
                pygame.draw.line(
                    surface, (110, 65, 68), (rect.left + 4, ylim), (rect.right - 4, ylim), 1
                )

    if len(t_buf) >= 2:
        t_start = t_buf[0]
        t_span = max(1e-6, t_buf[-1] - t_start)
        w = rect.width - 12

        for data_buf, color, width in series:
            if len(data_buf) < 2:
                continue
            pts = [
                (
                    int(rect.left + 6 + ((t_val - t_start) / t_span) * w),
                    val_to_y(val),
                )
                for t_val, val in zip(t_buf, data_buf)
            ]
            pygame.draw.lines(surface, color, False, pts, width)

    header = f"{title} [{unit}]"
    if series and len(series[-1][0]) > 0:
        header += f": {series[-1][0][-1]:+.2f}"
    lbl = font.render(header, True, TEXT_COLOR)
    surface.blit(lbl, (rect.left + 8, rect.top + 4))


def _draw_world(
    surface: pygame.Surface,
    rect: pygame.Rect,
    road: RoadProfile,
    step: EulerStep,
    settings: Settings,
    wheel_angle: float,
    manual_alpha: float | None,
    font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=4)
    surface.set_clip(rect)

    px_per_m = 28.0
    cam_s = step.s
    origin_x = rect.left + int(rect.width * 0.30)
    origin_y = rect.top + int(rect.height * 0.62)

    # Moving background grid in world coordinates (every 5 meters)
    grid_step_px = int(5.0 * px_per_m)
    x_offset = int((-step.x * px_per_m) % grid_step_px)
    y_offset = int((step.y * px_per_m) % grid_step_px)
    for gx in range(rect.left + x_offset, rect.right, grid_step_px):
        pygame.draw.line(surface, (35, 41, 45), (gx, rect.top), (gx, rect.bottom), 1)
    for gy in range(rect.top + y_offset, rect.bottom, grid_step_px):
        pygame.draw.line(surface, (35, 41, 45), (rect.left, gy), (rect.right, gy), 1)

    s_left = cam_s - (origin_x - rect.left) / px_per_m - 2.0
    s_right = cam_s + (rect.right - origin_x) / px_per_m + 2.0
    s_samples = np.linspace(s_left, s_right, 160)

    road_y_at_car = road.elevation(cam_s, settings.y0)

    def get_rel_y(s_val: float) -> float:
        if manual_alpha is not None:
            return (s_val - cam_s) * math.sin(manual_alpha)
        return road.elevation(s_val, settings.y0) - road_y_at_car

    road_pts: list[tuple[int, int]] = []
    for s_val in s_samples:
        rel_y = get_rel_y(float(s_val))
        sx = int(origin_x + (s_val - cam_s) * px_per_m)
        sy = int(origin_y - rel_y * px_per_m)
        road_pts.append((sx, sy))

    if len(road_pts) >= 2:
        poly_pts = road_pts + [(rect.right, rect.bottom), (rect.left, rect.bottom)]
        pygame.draw.polygon(surface, (39, 46, 51), poly_pts)
        pygame.draw.lines(surface, ACCENT_GREEN, False, road_pts, 3)

    # Sleepers/road markers every 2 meters and text every 10 meters
    first_tick = int(math.floor(s_left / 2.0) * 2)
    last_tick = int(math.ceil(s_right / 2.0) * 2)
    for s_tick in range(first_tick, last_tick + 1, 2):
        rel_y = get_rel_y(float(s_tick))
        mx = int(origin_x + (s_tick - cam_s) * px_per_m)
        my = int(origin_y - rel_y * px_per_m)
        if rect.left <= mx <= rect.right and rect.top <= my <= rect.bottom:
            if s_tick % 10 == 0:
                pygame.draw.line(surface, TEXT_COLOR, (mx, my), (mx, my + 12), 2)
                m_lbl = font.render(f"{s_tick}m", True, MUTED_TEXT)
                surface.blit(m_lbl, (mx - 14, my + 14))
            else:
                pygame.draw.line(surface, (85, 95, 90), (mx, my), (mx, my + 6), 2)

    # Draw the vehicle tilted at angle α
    ca = math.cos(step.α)
    sa = math.sin(step.α)
    r_px = max(8, int(settings.r * px_per_m * 1.4))
    half_wb = r_px * 2.2
    ch_h = r_px * 1.6

    def to_screen(local_x: float, local_y: float) -> tuple[int, int]:
        rx = local_x * ca - local_y * sa
        ry = local_x * sa + local_y * ca
        return int(origin_x + rx), int(origin_y - ry)

    chassis = [
        to_screen(-half_wb * 1.35, r_px * 0.7),
        to_screen(half_wb * 1.35, r_px * 0.7),
        to_screen(half_wb * 1.15, r_px * 0.7 + ch_h),
        to_screen(-half_wb * 1.25, r_px * 0.7 + ch_h),
    ]
    pygame.draw.polygon(surface, ACCENT_BLUE, chassis)
    pygame.draw.polygon(surface, TEXT_COLOR, chassis, 2)

    for wx_local in (-half_wb, half_wb):
        w_center = to_screen(wx_local, r_px)
        pygame.draw.circle(surface, (25, 28, 30), w_center, r_px)
        pygame.draw.circle(surface, TEXT_COLOR, w_center, r_px, 2)
        spoke_dx = int((r_px - 2) * math.cos(wheel_angle))
        spoke_dy = int((r_px - 2) * math.sin(wheel_angle))
        pygame.draw.line(
            surface,
            ACCENT_ORANGE,
            (w_center[0] - spoke_dx, w_center[1] + spoke_dy),
            (w_center[0] + spoke_dx, w_center[1] - spoke_dy),
            2,
        )

    surface.set_clip(None)
    pygame.draw.rect(surface, GRID_COLOR, rect, width=1, border_radius=4)


def run_live(
    road: RoadProfile,
    f: DynamicsModel,
    settings: Settings,
    window_size: tuple[int, int] = (1200, 760),
) -> EulerSim:
    """Runs interactive real-time simulation in Pygame and returns EulerSim."""
    mode = _get_mode(f)

    pygame.init()
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption(f"Vehicle Simulator Live ({mode})")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 14)
    font_bold = pygame.font.SysFont("monospace", 15, bold=True)

    sim = step_method_forever(road, f, settings)
    step = next(sim)

    dt = settings.dt

    vr_manual: float | None = None
    alpha_manual: float | None = None
    u_manual: float | None = None
    open_loop = False
    paused = False
    time_scale = 1.0
    accumulator = 0.0
    wheel_angle = 0.0

    max_pts = 400
    plot_stride = max(1, round(0.02 / dt))
    step_counter = 0

    t_buf: deque[float] = deque(maxlen=max_pts)
    v_buf: deque[float] = deque(maxlen=max_pts)
    vr_buf: deque[float] = deque(maxlen=max_pts)
    vm_buf: deque[float] = deque(maxlen=max_pts)
    i_buf: deque[float] = deque(maxlen=max_pts)
    u_buf: deque[float] = deque(maxlen=max_pts)
    a_buf: deque[float] = deque(maxlen=max_pts)

    running = True
    while running:
        real_dt = min(clock.tick(60) / 1000.0, 0.1)
        lo_cur, hi_cur = settings.get_control_limits(mode, step.v)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_SPACE:
                    if open_loop:
                        u_manual = 0.0
                    else:
                        vr_manual = 0.0
                elif event.key == pygame.K_m:
                    open_loop = not open_loop
                    if open_loop:
                        u_manual = step.Ia if mode == "current" else step.Ua
                    else:
                        u_manual = None
                        vr_manual = step.v
                elif event.key == pygame.K_r:
                    vr_manual = None
                    alpha_manual = None
                    u_manual = None
                    open_loop = False
                elif event.key == pygame.K_1:
                    time_scale = 0.25
                elif event.key == pygame.K_2:
                    time_scale = 1.0
                elif event.key == pygame.K_3:
                    time_scale = 2.0
                elif event.key == pygame.K_4:
                    time_scale = 5.0

        keys = pygame.key.get_pressed()
        if open_loop:
            span = settings.I_max - settings.I_min if mode == "current" else 2.0 * settings.U_nom
            u_rate = 0.35 * span * real_dt
            cur_u = u_manual if u_manual is not None else 0.0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                u_manual = float(np.clip(cur_u + u_rate, lo_cur, hi_cur))
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                u_manual = float(np.clip(cur_u - u_rate, lo_cur, hi_cur))
            else:
                u_manual = float(np.clip(cur_u, lo_cur, hi_cur))
        else:
            v_rate = 3.0 * real_dt
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                base_vr = vr_manual if vr_manual is not None else step.vr
                vr_manual = base_vr + v_rate
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                base_vr = vr_manual if vr_manual is not None else step.vr
                vr_manual = base_vr - v_rate

        alpha_rate = math.radians(8.0) * real_dt
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            base_a = alpha_manual if alpha_manual is not None else step.α
            alpha_manual = max(math.radians(settings.α_min), base_a - alpha_rate)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            base_a = alpha_manual if alpha_manual is not None else step.α
            alpha_manual = min(math.radians(settings.α_max), base_a + alpha_rate)

        if not paused:
            accumulator += real_dt * time_scale
            cmd = LiveCommand(vr=vr_manual, u_manual=u_manual, α=alpha_manual)

            while accumulator >= dt:
                step = sim.send(cmd)
                accumulator -= dt
                wheel_angle += (step.v / settings.r) * dt
                step_counter += 1

                if step_counter % plot_stride == 0:
                    t_buf.append(step.t)
                    v_buf.append(step.v)
                    vr_buf.append(step.vr)
                    vm_buf.append(step.vm)
                    i_buf.append(step.Ia)
                    u_buf.append(step.Ua)
                    a_buf.append(step.a)

        w, h = screen.get_size()
        screen.fill(BG_COLOR)

        world_rect = pygame.Rect(16, 16, w - 32, int(h * 0.42))
        _draw_world(
            screen, world_rect, road, step, settings, wheel_angle, alpha_manual, font
        )

        ctrl_state = (
            "OPEN-LOOP (Manual)"
            if open_loop
            else ("PID (Manual v_ref)" if vr_manual is not None else "PID (Auto ramp/road)")
        )
        slope_state = "Manual" if alpha_manual is not None else "RoadProfile"
        status_line = (
            f"t={step.t:6.2f}s ({time_scale:.2g}x) | "
            f"s={step.s:6.1f}m | "
            f"v={step.v:+5.2f} m/s (v_ref={step.vr:+5.2f}, v_m={step.vm:+5.2f}) | "
            f"α={math.degrees(step.α):+5.1f}° ({slope_state}) | "
            f"Ia={step.Ia:+6.1f}A | Ua={step.Ua:+6.1f}V | "
            f"Mode: {ctrl_state}"
        )
        help_line = (
            "[W/S or UP/DOWN]: Control v_ref / input  "
            "[A/D or LEFT/RIGHT]: Slope α  "
            "[SPACE]: Brake (0)  "
            "[M]: PID/Open-Loop  "
            "[R]: Auto  "
            "[P]: Pause  "
            "[1-4]: Time scale  "
            "[Q/ESC]: Quit & Save"
        )
        screen.blit(font_bold.render(status_line, True, TEXT_COLOR), (20, world_rect.bottom + 10))
        screen.blit(font.render(help_line, True, MUTED_TEXT), (20, world_rect.bottom + 32))

        charts_top = world_rect.bottom + 58
        charts_h = h - charts_top - 16
        half_w = (w - 44) // 2
        half_h = (charts_h - 12) // 2

        r_v = pygame.Rect(16, charts_top, half_w, half_h)
        r_a = pygame.Rect(28 + half_w, charts_top, half_w, half_h)
        r_i = pygame.Rect(16, charts_top + half_h + 12, half_w, half_h)
        r_u = pygame.Rect(28 + half_w, charts_top + half_h + 12, half_w, half_h)

        v_cap = settings.v_work if math.isfinite(settings.v_work) else 0.0
        v_span = max(v_cap * 1.4, max((abs(x) for x in v_buf), default=5.0) * 1.2, 1.0)
        _draw_strip_chart(
            screen,
            r_v,
            t_buf,
            [(vm_buf, MUTED_TEXT, 1), (vr_buf, ACCENT_ORANGE, 2), (v_buf, ACCENT_GREEN, 2)],
            -0.5 * v_span,
            v_span,
            "Velocity v (green), v_ref (orange), v_m (gray)",
            "m/s",
            font,
        )
        _draw_strip_chart(
            screen,
            r_a,
            t_buf,
            [(a_buf, ACCENT_ORANGE, 2)],
            settings.a_min * 1.5,
            settings.a_max * 1.5,
            "Acceleration a",
            "m/s²",
            font,
            limit_lines=[settings.a_min, settings.a_max],
        )
        _draw_strip_chart(
            screen,
            r_i,
            t_buf,
            [(i_buf, ACCENT_PURPLE, 2)],
            settings.I_min * 1.15,
            settings.I_max * 1.15,
            "Armature current Ia",
            "A",
            font,
            limit_lines=[settings.I_min, settings.I_max],
        )
        _draw_strip_chart(
            screen,
            r_u,
            t_buf,
            [(u_buf, ACCENT_BLUE, 2)],
            -settings.U_nom * 1.15,
            settings.U_nom * 1.15,
            "Armature voltage Ua",
            "V",
            font,
            limit_lines=[-settings.U_nom, settings.U_nom],
        )
        pygame.display.flip()

    pygame.quit()

    try:
        sim.throw(StopIteration)
    except StopIteration as e:
        return e.value

    raise RuntimeError("The generator did not return an EulerSim object.")
