import copy

import numpy as np
from scipy.optimize import minimize


class PhysicSetting:
    """
    Work in progress
    """
    def __init__(self, size, hpr, offset, model_path):
        # doffset is used for 4 wheels.
        self.size = size
        self.hpr = hpr
        self.offset = offset
        self.model_path = model_path

    @classmethod
    def offset_after_skew(cls, ps, dx, dy, index, model_path):
        # this is designed for vehicles' wheels,
        # so you dont have to repeat PhysicSetting.__init__(...) 4 times :)

        # 0 - right front wheel
        # 1 - left front wheel
        # 2 - right rear wheel
        # 3 - left rear wheel
        os = ps.offset
        new_os = os + dx * ((1 & index) * 2 - 1) + dy * (1 if index < 2 else -1)
        new_ps = PhysicSetting(ps.size, ps.hpr, new_os, model_path)
        return new_ps


class ModelPredictiveControl:
    """
    Work in progress
    """
    def __init__(self, horizon, dim, dt):
        self.state = None
        self.target = None
        self.horizon = horizon
        self.dim = dim
        self.dt = dt
        self.mass = 800
        self.len = 1
        self.bounds = []
        self.u = np.zeros(self.dim * self.horizon, dtype=np.float32)
        self.config = {"replan": False}

    def cost(self, u, *args):
        raise NotImplementedError

    def plant_model(self, state, dt, *control):
        raise NotImplementedError

    def solve(self):
        if self.config['replan']:
            self.u = np.zeros(self.dim * self.horizon, dtype=np.float32)
        else:
            for _ in range(self.dim):
                self.u = np.delete(self.u, 0)
            for _ in range(self.dim):
                self.u = np.append(self.u, self.u[-1 * self.dim])


class OpponentModelPredictiveControl(ModelPredictiveControl):
    def __init__(self, landmarks, horizon, dim, repeat, dt):
        super(OpponentModelPredictiveControl, self).__init__(horizon, dim, dt)
        self.landmarks = landmarks
        self.target_index = 0
        self.dt = dt
        self.dim = dim
        self.repeat = repeat

        for i in range(self.horizon):
            for _ in range(1):
                self.bounds += [[-1, 1]]
            for _ in range(1):
                self.bounds += [[-1, 1]]

    def cost(self, u, *args):
        state = args[0]
        vehicles = args[1]
        cost = 0.0
        for i in range(self.horizon - 1):
            head_u = u[self.dim * i:(self.dim) * (i + 1)]
            tail_u = u[self.dim * (i + 1):(self.dim) * (i + 2)]
            for j in range(self.repeat):
                state = self.plant_model(
                    state, self.dt, [head_u[_] + (tail_u[_] - head_u[_] * (j / self.repeat)) for _ in range(self.dim)]
                )
                x, y = state[0], state[1]
                dist = (
                    (x - self.landmarks[self.target_index][0])**2 + (y - self.landmarks[self.target_index][1])**2
                )**.5
                cost += dist
                for vehicle in vehicles:

                    p = np.array(
                        [
                            x - vehicle.position[0] - self.dt * (i + j / self.repeat) * vehicle.velocity[0],
                            y + vehicle.position[1]
                        ]
                    )
                    d = (p[0] - x)**2 + (p[1] + y)**2
                    alpha = -vehicle.heading
                    c, s = math.cos(alpha), math.sin(alpha)
                    ratio = 4
                    # print(alpha)
                    d = p @ np.array(
                        [[c**2 + ratio * s**2, -(ratio - 1) * c * s], [-(ratio - 1) * c * s, c**2 + ratio * s**2]]
                    ) @ p.T
                    if d > 50:
                        cost += 20
                    else:
                        cost += 1000 / d

            if state[2] > 6:
                cost += (state[2] - 6) * 10
            # if state[2] < 0.1:
            #     cost += 10000
            if state[2] > 1:
                cost += np.std(u[1::2]) * 20

        # print('current cost %.4f' % cost)
        # cost += (state[2] - 10) **2
        return cost

    def plant_model(self, state, dt, *control):
        [x, y, v, phi, beta] = state
        pedal, steering = control[0]
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
                v = new_v = 0

        new_phi = phi + v / 2 / 2 * np.tan(steering / 180 * np.pi) * dt
        new_x = x + v * math.cos(phi + beta) * dt
        new_y = y + v * math.sin(phi + beta) * dt
        new_state = [new_x, new_y, new_v, new_phi, new_beta]
        return new_state

    def update_target(self, pos):
        dist = lambda pos, lm, i: ((pos[0] - lm[i][0])**2 + (pos[1] - lm[i][1])**2)**0.5

        if self.target_index == 0:
            while dist(pos, self.landmarks, self.target_index) > dist(pos, self.landmarks, self.target_index + 1):
                self.target_index += 1

        while (self.landmarks[self.target_index][0] - pos[0]) * \
                (self.landmarks[self.target_index + 1][0] - pos[0]) + \
                (self.landmarks[self.target_index][1] - pos[1]) * \
                (self.landmarks[self.target_index + 1][1] - pos[1]) < 0:
            self.target_index += 1
        d = dist(pos, self.landmarks, self.target_index)
        if d < 5:
            self.target_index += 1
        return self.target_index

    def solve(self, ref, vehicles):
        new_u = copy.deepcopy(self.u)
        for _ in range(self.dim):
            new_u = np.delete(new_u, 0)
        for _ in range(self.dim):
            new_u = np.append(new_u, new_u[-1 * self.dim])
        for _ in range(self.dim * self.horizon):
            self.u[_] += (new_u[_] - self.u[_]) / self.repeat
        for _ in range(self.dim * self.horizon):
            self.u[_] = min(max(self.u[_], self.bounds[_][0]), self.bounds[_][1])

        # self.u = np.zeros(self.dim * self.horizon)

        u_optimze = minimize(
            self.cost, self.u, (ref, vehicles), method="SLSQP", bounds=self.bounds, tol=1e-5, options={'disp': False}
        )
        self.u = u_optimze.x
        return self.u[:self.dim]
