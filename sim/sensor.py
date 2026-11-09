#!/usr/bin/env python3

import warnings
import numpy as np

TWO_PI = 2.0 * np.pi
ACCURACY = 0.25
VALID_PPR = {
    48, 96, 100, 125, 192, 200, 250, 256,
    384, 400, 500, 512, 800, 1000, 1024, 2048,
}
DUTY_CYCLE_TOL_BY_PPR = {
    **{ppr: (-0.01, 0.01) for ppr in (256, 512, 1024, 2048)},
    **{ppr: (-0.03, 0.03) for ppr in (48, 96, 100, 125, 192, 200, 250, 384, 400, 500)},
    **{ppr: (-0.07, 0.06) for ppr in (800, 1000)},
}
FLANK_EDGE_PULSE_TRIGGER_TIME = 30e-9

class AMT102VSensor:
    def __init__(
        self,
        PPR=2048,
        wheel_radius=0.2,
        gear_ratio=3.0,
        sample_rate_hz=100.0,
        enable_quantization=True,
        seed=1,
    ):
        """Initierar AMT102-V sensormodellen.

        Parameters
        ----------
        PPR : int
            Pulser per varv (Pulses Per Revolution). Standard: 2048 PPR.
        wheel_radius : float
            Fordonets hjulradie i meter [m]. Standard: 0.2 m.
        gear_ratio : float
            Utväxling från motor till hjul. Standard: 3.0.
        sample_rate_hz : float
            Sensorns samplingfrekvens i Hz.
        enable_quantization : bool, default=True
            Simulera diskretiseringsfel (kvantiseringsfel från pulser).
        seed :
            Seed till rng
        """
        if PPR not in VALID_PPR:
            raise ValueError(f"Ogiltig PPR ({PPR}). Tillåtna värden {VALID_PPR}")

        # Hårdvaraspecifikationer och inställningar
        self.PPR = PPR
        self.CPR = 4 * PPR  # PPR x 4 efter kvadraturavkodning
        self.max_rpm = 7500.0 if PPR >= 1000 else 15000.0
        self.max_omega_shaft = (self.max_rpm / 60.0) * TWO_PI
        self.max_omega_shaft_range = (-self.max_omega_shaft, self.max_omega_shaft)
        self.accuracy_rad = np.radians(ACCURACY)
        self.duty_cycle_tol = DUTY_CYCLE_TOL_BY_PPR[PPR]
        self.machine_precision = 1e-15

        # Simulering
        self.r = wheel_radius
        self.gear_ratio = gear_ratio
        self.sample_interval = 1.0 / sample_rate_hz
        self.enable_quantization = enable_quantization
        self.seed = seed

        # Konverteringsfaktorer
        self.omega_to_v = wheel_radius / gear_ratio
        self.rad_to_counts = self.CPR / TWO_PI
        self.counts_to_rad = 1.0 / self.rad_to_counts

        self.reset()

    def _build_wheel_edges(self):
        """Beräknar fysiska flankvinklar för enheten baserat på duty cycle och vinkelnoggrannhet."""
        pulse_period_rad = TWO_PI / self.PPR # pulse period
        duty_cycle = 0.50 + self.duty_cycle_asymmetry

        # Flankpositioner inom en elektrisk cykel (A upp, B upp, A ner, B ner)
        base_pulse_edges = np.array([0.0,  0.25, duty_cycle, duty_cycle+0.25])
        pulse_starts = np.repeat(np.arange(self.PPR) * pulse_period_rad, 4)
        edge_offsets = np.tile(base_pulse_edges * pulse_period_rad, self.PPR)
        ideal_edge_angles = pulse_starts + edge_offsets
        edge_error_rad = self.accuracy_rad * np.sin(
            ideal_edge_angles + self.eccentricity_phase_offset
        )
        distorted_edge_angles = ideal_edge_angles + edge_error_rad
        return np.sort(np.mod(distorted_edge_angles, TWO_PI))

    def _compute_hardware_counts(self, omega_shaft):
        """Beräknar heltalsvärdet i hårdvarans pulsräknare vid nuvarande axelvinkel."""
        edge_delay = omega_shaft * FLANK_EDGE_PULSE_TRIGGER_TIME
        delayed_shaft_angle_rad = (
            self.theta
            + self.initial_shaft_angle_offset
            - edge_delay
        )
        full_revolutions = int(np.floor(delayed_shaft_angle_rad / TWO_PI))
        revolution_angle_rad = delayed_shaft_angle_rad - full_revolutions * TWO_PI

        passed_edges_in_revolution = int(
            np.searchsorted(self.wheel_edge_angles_rad, revolution_angle_rad, side="right")
        )

        return full_revolutions * self.CPR + passed_edges_in_revolution

    def reset(self):
        """Återställ/nollställ sensorn inför en ny simulering."""
        self.rng = np.random.default_rng(self.seed)

        self.started = False
        self._warned_overspeed = False

        self.t_next = 0.0
        self.t_sample_prev = 0.0
        self.t_prev = 0.0
        self.theta = 0.0
        self.prev_total_counts = 0
        self.last_measured_v = 0.0

        # Slumpas automatiskt inom databladets gränser för vald PPR
        self.initial_shaft_angle_offset = self.rng.uniform(0.0, TWO_PI)
        self.eccentricity_phase_offset = self.rng.uniform(0.0, TWO_PI)
        self.duty_cycle_asymmetry = self.rng.uniform(*self.duty_cycle_tol)

        self.wheel_edge_angles_rad = self._build_wheel_edges()

    def first_measure(self, t):
        self.t_prev = t
        self.t_sample_prev = t
        self.t_next = t + self.sample_interval
        self.prev_total_counts = self._compute_hardware_counts(omega_shaft=0.0)
        self.started = True
        return 0.0

    def measure_velocity(self, v_true, t) -> float:
        omega_wheel = v_true / self.r
        omega_shaft_demand = omega_wheel * self.gear_ratio

        if abs(omega_shaft_demand) > self.max_omega_shaft and not self._warned_overspeed:
            warnings.warn("AMT102-V användes över max RPM: utsignal utanför specs")
            self._warned_overspeed = True

        omega_shaft = np.clip(omega_shaft_demand, *self.max_omega_shaft_range)

        if not self.started:
            return self.first_measure(t)

        # Integrera motoraxelns vinkel
        dt = t - self.t_prev
        self.t_prev = t
        if dt > 0.0:
            self.theta += omega_shaft * dt

        # Vänta tills nästa samplingsögonblick
        if t < self.t_next:
            return self.last_measured_v

        dt_sample = t - self.t_sample_prev
        self.t_sample_prev = t
        while self.t_next <= t:
            self.t_next += self.sample_interval

        # Uppmätt värde om kvantisering är avstängd
        v_measured = omega_shaft * self.omega_to_v

        if self.enable_quantization:
            quantized_total_counts = self._compute_hardware_counts(omega_shaft)

            sampled_count_diff = quantized_total_counts - self.prev_total_counts
            self.prev_total_counts = quantized_total_counts

            omega_measured = (sampled_count_diff * self.counts_to_rad) / dt_sample
            v_measured = omega_measured * self.omega_to_v

        self.last_measured_v = float(v_measured)
        return self.last_measured_v
