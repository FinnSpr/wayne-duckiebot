# import time

# class PID:
#     def __init__(self, kp, ki, kd):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
        
#         self.reset()

#     def reset(self):
#         self.integral = 0.0
#         self.previous_error = 0.0
#         self.previous_time = None

#     def update(self, error):
#         now = time.time()

#         if self.previous_time is None:
#             self.previous_time = now
#             return self.kp * error

#         dt = now - self.previous_time

#         if dt <= 0:
#             return -self.kp * error

#         # Integral
#         self.integral += error * dt

#         # Derivative
#         derivative = (error - self.previous_error) / dt

#         # PID
#         omega = (
#             self.kp * error
#             + self.ki * self.integral
#             + self.kd * derivative
#         )

#         self.previous_error = error
#         self.previous_time = now

#         return -omega

# pid.py
import time

class PID:
    def __init__(self, kp, ki, kd, integral_limit=None, max_dt=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.max_dt = max_dt
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def update(self, error):
        now = time.time()

        if self.previous_time is None:
            self.previous_time = now
            self.previous_error = error
            return -(self.kp * error)

        dt = now - self.previous_time
        self.previous_time = now  # store the *real* elapsed time, not the clamped one

        if dt <= 0:
            self.previous_error = error
            return -(self.kp * error)

        if self.max_dt is not None:
            dt = min(dt, self.max_dt)

        # Integral, with anti-windup clamp
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        # Derivative
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        omega = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return -omega