class PIDController:
    def __init__(self, k_p: float, k_i: float, k_d: float):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0

    def _update_error(self, current_error: float):
        self.i_error += current_error
        self.d_error = current_error - self.p_error
        self.p_error = current_error

    def get_result(self, current_error: float, make_up_coefficient=1.0):
        self._update_error(current_error)
        return (-self.k_p * self.p_error - self.k_i * self.i_error - self.k_d * self.d_error) * make_up_coefficient

    def reset(self):
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0


class Target:
    def __init__(self, target_lateral, target_speed):
        self.lateral = target_lateral
        self.speed = target_speed

    def go_right(self):
        self.lateral += 0.25 if self.lateral < 0.625 else 0

    def go_left(self):
        self.lateral -= 0.25 if self.lateral > 0.125 else 0

    def faster(self):
        self.speed += 10

    def slower(self):
        self.speed -= 10
