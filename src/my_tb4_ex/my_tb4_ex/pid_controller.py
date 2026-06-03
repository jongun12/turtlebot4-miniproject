import time

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_error = 100.0
        self.min_error = -100.0
        self.tolerance = 0.05
        self.prev_error = 0.0
        self.integral = 0.0
        self.pre_time = time.time()
    
    def update(self, error):
        current_time = time.time()
        dt = current_time - self.pre_time
        self.pre_time = current_time

        if abs(error) < self.tolerance:
            return 0.0
        
        error = max(min(error, self.max_error), self.min_error)

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        return output