class PID:
    def __init__(self, kp, ki, kd, integral_max=500.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error):
        self.integral += error
        if abs(error) < 100:
            self.integral *= 0.35
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        derivative = error - self.last_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0