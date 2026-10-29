import numpy as np


class BicycleModel:
    """
    This model can be used to predict next state
    """
    def __init__(self):
        self.state = dict(x=0, y=0, speed=0, heading_theta=0, velocity_dir=0)

    def reset(self, x, y, speed, heading_theta, velocity_dir):
        """
        heading_theta, velocity_dir in radian
        """
        self.state = dict(x=x, y=y, speed=speed, heading_theta=heading_theta, velocity_dir=velocity_dir)

    def predict(self, dt, control):
        """
        In this model, we formulate the car's dynamic model as Bicycle model
        things need to be finetuned
        mass
        mu*g: a = f/m + \mu * mg
        """
        x = self.state["x"]
        y = self.state["y"]
        v = self.state["speed"]
        phi = self.state["heading_theta"]
        beta = self.state["velocity_dir"]
        pedal, steering = control[0], control[1]
        pedal *= 500
        steering *= 40
        new_beta = np.arctan(0.5 * np.tan(steering / 180 * np.pi))

        if pedal < 0:
            pedal *= 3

        a = pedal / 500 * 3
        af = .5

        new_v = 0
        if v > 1e-5 or a > af:
            new_v = v + (a - af) * dt
            if v * new_v < 0:
                new_v = 0

        new_phi = phi + v / 2 / 2 * np.tan(steering / 180 * np.pi) * dt
        new_x = x + v * np.cos(phi + beta) * dt
        new_y = y + v * np.sin(phi + beta) * dt
        new_state = dict(x=new_x, y=new_y, speed=new_v, heading_theta=new_phi, velocity_dir=new_beta)
        self.state = new_state
        return new_state
