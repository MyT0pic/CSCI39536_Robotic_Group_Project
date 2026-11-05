"""PID controller."""
from dataclasses import dataclass, field
from typing import Optional

from robot.utils.numeric import clamp, normalize_angle
from robot.utils.types import Limits


@dataclass
class PIDController:
    """PID controller with output limits and anti-windup.

    The derivative is taken of the measurement, not the error, so a setpoint change doesn't kick the output.

    Args:
        kp, ki, kd: gains.
        output_limits: (low, high) for the output.
        integral_limits: (low, high) for the accumulated error.
        wrap_angle: setpoint and measurement are angles (rad); errors take the short way round.
    """
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limits: Limits = (None, None)
    integral_limits: Limits = (None, None)
    wrap_angle: bool = False
    integral: float = field(default=0.0, init=False)
    _previous: Optional[float] = field(default=None, init=False, repr=False)  # last measurement

    def reset(self) -> None:
        """Clear the integral and the last measurement."""
        self.integral = 0.0
        self._previous = None

    def update(self, measurement: float, dt: float, setpoint: float, feedforward: float = 0.0) -> float:
        """Advance one step and return the output.

        Args:
            measurement: current value.
            dt: time since the last update (s).
            setpoint: target value.
            feedforward: added to the output.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        error = self._difference(setpoint, measurement)
        derivative = 0.0 if self._previous is None else -self._difference(measurement, self._previous) / dt
        self._previous = measurement
        previous_integral = self.integral
        self.integral = clamp(self.integral + error * dt, self.integral_limits)
        raw = self.kp * error + self.ki * self.integral + self.kd * derivative + feedforward
        output = clamp(raw, self.output_limits)
        # Anti-windup: while the output is held at a limit, don't let the integral grow toward it
        if output != raw and (self.integral - previous_integral) * (raw - output) > 0:
            self.integral = previous_integral
        return output

    def _difference(self, a: float, b: float) -> float:
        return normalize_angle(a - b) if self.wrap_angle else a - b
