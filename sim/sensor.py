#!/usr/bin/env python3

import math
import warnings
import numpy as np

TWO_PI = 2.0 * np.pi

PPR_SIGMA_1 = [256, 512, 1024, 2048]
PPR_SIGMA_2 = [48, 96, 100, 125, 192, 200, 250, 384, 400, 500]
PPR_SIGMA_3 = [800, 1000]
VALID_PPR = {*PPR_SIGMA_1, *PPR_SIGMA_2, *PPR_SIGMA_3}

# Quadrature duty cycle tolerance (min, max offset from 50%) by resolution
DUTY_CYCLE_TOL_BY_PPR = {
    **{ppr: (-0.01, 0.01) for ppr in PPR_SIGMA_1},  # 49 - 51 %
    **{ppr: (-0.03, 0.03) for ppr in PPR_SIGMA_2},  # 47 - 53 %
    **{ppr: (-0.07, 0.06) for ppr in PPR_SIGMA_3},  # 43 - 56 %
}

# Resolutions limited to 7500 RPM; all other resolutions allow 15000 RPM
PPR_MAX_7500_RPM = {192, 384, 400, 500, 800, 1000, 1024, 2048}
FLANK_EDGE_PULSE_TRIGGER_TIME = 30e-9  # CMOS rise/fall time (s)


class AMT102VSensor:
    def __init__(
        self,
        *,
        wheel_radius,
        gear_ratio,
        PPR=2048,
        sample_rate_hz=100.0,
        enable_quantization=True,
    ):
        """Initialize a discrete AMT102-V sensor model.

        Parameters
        ----------
        wheel_radius : float
            Vehicle wheel radius in meters.
        gear_ratio : float
            Gear ratio from motor to wheel.
        PPR : int, default=1024
            Pulses per revolution.
        sample_rate_hz : float, default=100.0
            Sensor sampling frequency in Hz.
        enable_quantization : bool, default=True
            Simulate quantization.
        """
        if PPR not in VALID_PPR:
            raise ValueError(
                f"Invalid PPR ({PPR}), choose from available {VALID_PPR}"
            )

        # Hardware settings
        self.PPR = PPR
        self.CPR = 4 * PPR  # PPR x 4 after quadrature decoding
        max_rpm = 7500.0 if PPR in PPR_MAX_7500_RPM else 15000.0
        self.max_omega = (max_rpm / 60.0) * TWO_PI
        self.accuracy = np.radians(0.25)
        self.duty_cycle_tol = DUTY_CYCLE_TOL_BY_PPR[PPR]
        self.rise_fall_time = FLANK_EDGE_PULSE_TRIGGER_TIME

        # Simulation parameters
        self.sample_interval = 1.0 / sample_rate_hz
        self.enable_quantization = enable_quantization
        self.rng = np.random.default_rng(seed=42)

        self.omega_to_v = wheel_radius / gear_ratio
        self.v_to_omega = 1.0 / self.omega_to_v
        self.counts_to_rad = TWO_PI / self.CPR

        self.reset()

    def get_counts(self, omega):
        """Integer pulse count at the current shaft angle."""
        # Half the 30 ns rise/fall time for logic-threshold crossing plus
        # 1.5 us hardcoded MCU interrupt/timer capture jitter.
        capture_delay = (
            0.5 * self.rise_fall_time + self.rng.normal(0.0, 1.5e-6)
        )
        sensed_angle = (
            self.theta + self.theta0  # True shaft angle
            - omega * capture_delay   # Threshold + capture delay
        )
        accuracy_error = self.accuracy_amplitude * math.sin(
            sensed_angle + self.accuracy_phase
        )

        # Convert angle to quadrature counts and apply falling-edge offsets
        continuous_counts = (
            (sensed_angle - accuracy_error) / self.counts_to_rad
        )
        pulses = math.floor(continuous_counts / 4.0)
        state_pos = continuous_counts - 4.0 * pulses

        # [A_rise, B_rise, A_fall, B_fall]
        return (
            4 * pulses
            + (state_pos >= 1.0)
            + (state_pos >= 2.0 + self.duty_offset_A)
            + (state_pos >= 3.0 + self.duty_offset_B)
        )

    def reset(self):
        """Reset the sensor before a new simulation."""
        # Unknown starting position (missing Z-index)
        self.theta0 = self.rng.uniform(0.0, TWO_PI)

        # Once-per-revolution accuracy error.
        # The datasheet lists the accuracy as typical, with no distribution
        # given. Assume it is the mean amplitude of a uniform [0, 2x typical]
        # draw. The phase is unknown relative to the disc.
        self.accuracy_amplitude = self.rng.uniform(0.0, 2.0 * self.accuracy)
        self.accuracy_phase = self.rng.uniform(0.0, TWO_PI)

        # Tolerance deviation in the quadrature duty cycle (%)
        # Randomized within the limits for the selected PPR
        self.duty_offset_A = 4.0 * self.rng.uniform(*self.duty_cycle_tol)
        self.duty_offset_B = 4.0 * self.rng.uniform(*self.duty_cycle_tol)

        self.t_prev = None
        self.t_sample_prev = 0.0
        self.t_next = 0.0
        self.theta = 0.0
        self.prev_counts = 0
        self.omega_prev = 0.0
        self.last_measured_v = 0.0

    def measure_velocity(self, v_true, t) -> float:
        if not self.enable_quantization:
            return v_true

        if abs(omega := v_true * self.v_to_omega) > self.max_omega:
            warnings.warn("AMT102-V exceeded max RPM")

        if self.t_prev is None:
            self.t_prev = t
            self.t_sample_prev = t
            self.t_next = t + self.sample_interval
            self.prev_counts = self.get_counts(omega)
            self.omega_prev = omega
            return 0.0

        if (dt := t - self.t_prev) < 0.0:
            raise ValueError("Reset sensor before new simulation")
        self.t_prev = t

        # Integrate shaft angle as trapezoid
        self.theta += 0.5 * (self.omega_prev + omega) * dt
        self.omega_prev = omega

        # Wait for next sample
        if t < self.t_next:
            return self.last_measured_v

        dt_sample = t - self.t_sample_prev
        self.t_sample_prev = t

        # Skip missed samples if step is longer than interval
        while self.t_next <= t:
            self.t_next += self.sample_interval

        counts = self.get_counts(omega)
        count_diff = counts - self.prev_counts
        self.prev_counts = counts

        omega_measured = (count_diff * self.counts_to_rad) / dt_sample
        self.last_measured_v = omega_measured * self.omega_to_v
        return self.last_measured_v
