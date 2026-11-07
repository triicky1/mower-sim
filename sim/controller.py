#!/usr/bin/env python3

import numpy as np

# https://www.digikey.com/en/maker/tutorials/2024/implementing-a-pid-controller-algorithm-in-python
class Controller:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, Ts=0.0, enable_history=False):
        """Initierar en tidsdiskret PID-regulator

        Parameters
        ----------
        Kp : float, default=0.0
            Proportionalförstärkning.
        Ki : float, default=0.0
            Integralförstärkning.
        Kd : float, default=0.0
            Derivataförstärkning.
        dt : float, default=0.0
            Tidssteg i sekunder.
        enable_history : bool, default=False
            Spara historik av alla börvärden, insignaler och styrsignaler.
        """
        self.Kp, self.Ki, self.Kd, self.Ts = Kp, Ki, Kd, Ts

        self.want_log = enable_history

        self.reset()


    # TODO: Lägg till beskrivning för anti-windup
    # Slutar integreringen vid små förändringar ...
    def update(self, setpoint, measurement, lo, hi):
        """Beräkna nästa styrsignal.

        Parameters
        ----------
        setpoint : float
            Processens börvärde.
        measurerment : float
            Processens mätvärde.
        lo, hi : float
            Utsignalens undre/övre begräsning.
        """
        error = setpoint - measurement
        integral = self.integral + error * self.Ts
        derivative = (error-self.previous_error)/self.Ts

        u_demand = self.Kp * error + self.Ki * integral + self.Kd * derivative
        u_command = np.clip(u_demand, lo, hi)
        self.previous_error = error

        # Mättnadsskydd (villkorlig integrering)
        if error * (u_demand - u_command) <= 0:
            self.integral = integral

        # Uppdatera historik
        if self.want_log:
            self._update_history(setpoint,measurement,u_command)

        return np.clip(u_command, lo, hi)


    def reset(self):
        """Nollställer/initierar regulatorinställningar."""
        self.previous_error = 0.0
        self.integral = 0.0
        if self.want_log:
            self.timestamps = []
            self.y_values = []
            self.u_values = []
            self.r_values = []


    def _update_history(self, r, y, u):
        """Uppdaterar historik."""
        if self.want_log:
            t = self.timestamps[-1] + self.Ts if self.timestamps else 0
            self.timestamps.append(t)
            self.y_values.append(y)
            self.u_values.append(u)
            self.r_values.append(r)
