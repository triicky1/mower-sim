#!/usr/bin/env python3

import numpy as np

class Controller:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, Ts=0.0, enable_history=False):
        """Initializes a discrete-time PID controller

        Parameters
        ----------
        Kp : float, default=0.0
            Proportional gain.
        Ki : float, default=0.0
            Integral gain.
        Kd : float, default=0.0
            Derivative gain.
        dt : float, default=0.0
            Time step in seconds.
        enable_history : bool, default=False
            Save history of all setpoints, input signals, and control signals.
        """
        self.Kp, self.Ki, self.Kd, self.Ts = Kp, Ki, Kd, Ts
        self.want_log = enable_history
        self.reset()


    def update(self, setpoint, measurement, lo, hi):
        """Calculate the next control signal.

        Parameters
        ----------
        setpoint : float
            Process setpoint.
        measurerment : float
            Process measurement.
        lo, hi : float
            Lower/upper limits of the output signal.
        """
        error = setpoint - measurement
        integral = self.integral + error * self.Ts
        derivative = (error-self.previous_error)/self.Ts

        u_demand = self.Kp * error + self.Ki * integral + self.Kd * derivative
        u_command = np.clip(u_demand, lo, hi)
        self.previous_error = error

        # Anti-windup
        if error * (u_demand - u_command) <= 0:
            self.integral = integral

        # Update history
        if self.want_log:
            self._update_history(setpoint,measurement,u_command)

        return u_command


    def reset(self):
        """Resets/initializes controller settings."""
        self.previous_error = 0.0
        self.integral = 0.0
        if self.want_log:
            self.timestamps = []
            self.y_values = []
            self.u_values = []
            self.r_values = []


    def _update_history(self, r, y, u):
        """Updates history."""
        if self.want_log:
            t = self.timestamps[-1] + self.Ts if self.timestamps else 0
            self.timestamps.append(t)
            self.y_values.append(y)
            self.u_values.append(u)
            self.r_values.append(r)
