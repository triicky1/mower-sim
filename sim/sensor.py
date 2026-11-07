#!/usr/bin/env python3

import numpy as np
import warnings

r"""Vald sensor: Same Sky (CUI Devices) AMT102-V Inkrementell Kapacitiv Pulsgivare

Kontinuerlig vinkel vid varje steg $\Delta t = t - t_\text{prev}$
$$\theta(t) = \theta(t_\text{prev}) + \left(\frac{v_\text{true}}{r_\text{wheel} \cdot i_\text{gear}\right) \Delta t$$

Avläst rotorvinkel vid varje sampel $t_k$ inkl. noggrannhetsfel
$$\tilde{\theta}(t_k) = \theta(t_k) + e_{\text{acc}}$$

Kvantiseringsräknare
$$C_k = \left\lfloor \frac{\tilde{\theta}(t_k)}{2\pi} \cdot \text{CPR} \right\rfloor$$

Uppmätt hastighet med kvantiseringsfel
$$v_\text{meas} = \frac{C_k}{\mathrm{CPR} \cdot (t_k - t_{k-1} + \delta{t_jitter}$$
"""

VALID_PPR = {48, 96, 100, 125, 192, 200, 250, 256, 384, 400, 500, 512, 800, 1000, 1024, 2048}
TWO_PI = 2.0 * np.pi

ACCURACY = 0.25  # Upplösningsfel ()

DUTY_CYCLE_TOL_BY_PPR = {
    **{ppr: 0.01 for ppr in (256, 512, 1024, 2048)},
    **{ppr: 0.03 for ppr in (48, 96, 100, 125, 192, 200, 250, 384, 400, 500)},
    **{ppr: 0.065 for ppr in (800, 1000)}
}

FLANK_TRIGGER_TIME = 30e-9 # 30 ns

class AMT102VSensor:
    def __init__(
        self,
        PPR=2048,
        wheel_radius=0.2,
        gear_ratio=3.0,
        sample_rate_hz=100.0,
        enable_quantization=True,
        seed=1
    ):
        """Initierar AMT102-V sensormodellen.

        Parametrar:
        -----------
        PPR : int
            Pulser per varv (Pulses Per Revolution). Standard: 2048 PPR (AMT102-V inställning).
        wheel_radius : float
            Fordonets hjulradie i meter [m]. Standard: 0.2 m.
        gear_ratio : float
            Utväxling från motor till hjul (i_gear). Standard: 3.0.
        sample_rate_hz : float
            Sensorns samplingfrekvens i Hz.
        enable_quantization : bool
            Om True simuleras diskretiseringsfel (kvantiseringsfel från pulser).
        """
        if PPR not in VALID_PPR:
            raise ValueError(f"Ogiltig PPR ({PPR}). Tillåtna värden {VALID_PPR}")

        # Hårdvaraspecifikationer och inställningar
        self.PPR = PPR
        self.CPR = 4 * PPR # PPR x 4 efter kvadraturavkodning
        self.max_rpm = 7500.0 if PPR >= 1000 else 15000.0
        self.max_omega_shaft = (self.max_rpm / 60.0) * TWO_PI
        self.accuracy_rad = np.radians(ACCURACY)
        self.duty_cycle_tol = DUTY_CYCLE_TOL_BY_PPR[PPR]

        # Simulering
        self.r = wheel_radius
        self.i_gear = gear_ratio
        self.sample_interval = 1.0 / sample_rate_hz
        self.enable_quantization = enable_quantization
        self.seed = seed

        # Konstanter / koverteringskonstanter
        self.epsilon = 1e-6
        self.omega_to_v = wheel_radius / gear_ratio
        self.rad_to_counts = self.CPR / (2.0 * np.pi)
        self.counts_to_rad = 1.0 / self.rad_to_counts

        self.reset()


    def reset(self):
        """Återställ/nollställ sensorn inför en ny simulering."""
        self.rng = np.random.default_rng(self.seed)
        self.started = False
        self.t_next = 0.0            # nästa samplingsögonblick
        self.t_sample_prev = 0.0     # tid för föregående sampling
        self.t_prev = 0.0            # tid för föregående anrop
        self.theta = 0.0             # integrerad motorvinkel [rad]
        self.counts_prev = 0         # pulsräknare vid föregående sampling
        self.last_measured_v = 0.0   # senast uppmätta hastigheten

        # TODO:
        # self.eccentricity_phase = self.rng.uniform(0.0, 2.0 * np.pi)
        # self.duty_asymmetry = self.rng.uniform(-self.duty_cycle_tol, self.duty_cycle_tol)


    def first_measure(self, t):
        self.t_prev = t
        self.t_sample_prev = t
        self.t_next = t + self.sample_interval
        self.started = True
        return 0.0


    def measure_velocity(self, v_true, t) -> float:
        if not self.started:
            return self.first_measure(t)

        omega_wheel = v_true / self.r
        omega_shaft = np.clip(omega_wheel * self.i_gear, -self.max_omega_shaft, self.max_omega_shaft)

        # Integrera motorns vinkel
        dt = t - self.t_prev
        self.t_prev = t

        if dt > 0.0:
            self.theta += omega_shaft * dt

        # Vänta tills nästa samplingsögonblick
        if t + FLANK_TRIGGER_TIME < self.t_next:
            return self.last_measured_v

        dt_sample = t - self.t_sample_prev
        self.t_sample_prev = t
        while self.t_next <= t + FLANK_TRIGGER_TIME:
            self.t_next += self.sample_interval

        # Uppmätt värde om quantization avstängt
        v_measured = omega_shaft * self.omega_to_v

        if self.enable_quantization:
            # Mätfel +- 0.25 grader
            angle_error = (
                self.rng.uniform(-self.accuracy_rad, self.accuracy_rad)
                if abs(omega_shaft) > self.epsilon
                else 0.0
            )

            # Efter
            theta_measured = self.theta + angle_error * np.sin(self.theta % TWO_PI)
            # theta_measured = self.theta + angle_error

            # Heltal pulser i räknaren (counts)
            current_counts = np.trunc(theta_measured * self.rad_to_counts)
            delta_counts = current_counts - self.counts_prev
            self.counts_prev = current_counts

            # Räkna tillbaka från pulsdifferens till linjär hastighet
            omega_shaft_measured = (delta_counts * self.counts_to_rad) / dt_sample
            v_measured = (omega_shaft_measured / self.i_gear) * self.r

        self.last_measured_v = float(v_measured)
        return self.last_measured_v



# class AMT102VSensor:
#     def __init__(
#         self,
#         PPR=2048,
#         wheel_radius=0.2,
#         gear_ratio=3.0,
#         sample_rate_hz=100.0,
#         enable_quantization=True,
#         seed=1,                 # seed = en specifik fysisk encoder-enhet
#         ecc_fraction=0.9,       # andel av accuracy-budgeten som är excentricitet (antagande)
#         mount_ecc_deg=0.0,      # extra monteringsexcentricitet hos er (utanför datasheet)
#         phase_err_deg=0.0,      # fasfel A/B i elektriska grader (datasheet anger ingen tolerans)
#         clock_ppm=0.0,          # MCU-klockfel (egenskap hos avläsningen, inte sensorn)
#         jitter_std=0.0,         # avläsningsjitter [s] (egenskap hos avläsningen)
#     ):
#         if PPR not in VALID_PPR:
#             raise ValueError(f"Ogiltig PPR ({PPR}). Tillåtna värden {VALID_PPR}")
#
#         self.PPR = PPR
#         self.CPR = 4 * PPR
#         max_rpm = 7500.0 if PPR >= 1000 else 15000.0
#         self.max_omega_shaft = max_rpm / 60.0 * TWO_PI
#         self.duty_range = DUTY_RANGE_BY_PPR[PPR]
#         self.accuracy_rad = np.radians(ACCURACY_DEG)
#         self.ecc_fraction = ecc_fraction
#         self.mount_ecc_rad = np.radians(mount_ecc_deg)
#         self.phase_err = phase_err_deg / 360.0      # andel av T
#         self.clock_scale = 1.0 + clock_ppm * 1e-6
#         self.jitter_std = jitter_std
#
#         self.r = wheel_radius
#         self.i_gear = gear_ratio
#         self.sample_interval = 1.0 / sample_rate_hz
#         self.enable_quantization = enable_quantization
#         self.seed = seed
#         self.counts_to_rad = TWO_PI / self.CPR
#
#         self.reset()
#
#     # ------------------------------------------------------------------
#     def _build_edges(self):
#         """Faktiska kantvinklar [rad] för just den här enheten (CPR st, sorterade)."""
#         T = TWO_PI / self.PPR
#         d, pe = self.duty, self.phase_err
#         base = np.array([0.0, 0.25 + pe, d, d + 0.25 + pe])        # A↑ B↑ A↓ B↓
#         ideal = ((np.arange(self.PPR)[:, None] + base[None, :]) * T).ravel()
#
#         acc = self.accuracy_rad
#         sigma = acc * (1.0 - self.ecc_fraction) / 3.0
#         phi = self.rng.uniform(0.0, TWO_PI)                         # excentricitetens fas
#         pitch = np.clip(self.rng.normal(0.0, sigma, ideal.size), -3 * sigma, 3 * sigma)
#
#         err = acc * self.ecc_fraction * np.sin(ideal + phi) + pitch
#         if self.mount_ecc_rad:
#             err += self.mount_ecc_rad * np.sin(ideal + self.rng.uniform(0.0, TWO_PI))
#         return np.sort(np.mod(ideal + err, TWO_PI))
#
#     def _counts(self, theta):
#         """Antal passerade kanter (heltal) vid motoraxelvinkel theta [rad]."""
#         th = theta + self.theta0
#         rev = np.floor(th / TWO_PI)
#         frac = th - rev * TWO_PI
#         return int(rev) * self.CPR + int(np.searchsorted(self.edges, frac, side="right"))
#
#     # ------------------------------------------------------------------
#     def reset(self):
#         """Nollställ inför ny simulering. Samma seed ger samma fysiska enhet."""
#         self.rng = np.random.default_rng(self.seed)
#         self.theta0 = self.rng.uniform(0.0, TWO_PI)    # okänd startvinkel relativt kanterna
#         self.duty = self.rng.uniform(*self.duty_range)
#         self.edges = self._build_edges()
#         self.started = False
#         self.t_next = 0.0
#         self.t_sample_prev = 0.0
#         self.counts_prev = 0
#         self.last_measured_v = 0.0
#         self._warned = False
#
#     def measure_velocity(self, s_true, v_true, t) -> float:
#         theta = s_true / self.r * self.i_gear      # motoraxelns vinkel
#         omega = v_true / self.r * self.i_gear
#
#         if abs(omega) > self.max_omega_shaft and not self._warned:
#             warnings.warn("AMT102-V används över max RPM: utsignalen ligger utanför specen.")
#             self._warned = True
#
#         if not self.started:
#             self.counts_prev = self._counts(theta)
#             self.t_sample_prev = t
#             self.t_next = t + self.sample_interval
#             self.started = True
#             return 0.0
#
#         if t + FLANK_TRIGGER_TIME < self.t_next:
#             return self.last_measured_v
#
#         dt_sample = t - self.t_sample_prev
#         self.t_sample_prev = t
#         while self.t_next <= t + FLANK_TRIGGER_TIME:
#             self.t_next += self.sample_interval
#
#         if not self.enable_quantization:
#             self.last_measured_v = float(v_true)
#             return self.last_measured_v
#
#         # Avläsningsjitter: räknaren läses vid t + δ (vinkeln extrapoleras linjärt)
#         delta_t = self.rng.normal(0.0, self.jitter_std) if self.jitter_std > 0.0 else 0.0
#         counts = self._counts(theta + omega * delta_t)
#         delta_counts = counts - self.counts_prev
#         self.counts_prev = counts
#
#         omega_meas = delta_counts * self.counts_to_rad / (dt_sample * self.clock_scale)
#         self.last_measured_v = float(omega_meas / self.i_gear * self.r)
#         return self.last_measured_v
