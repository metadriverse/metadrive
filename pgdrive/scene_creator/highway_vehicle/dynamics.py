from typing import Tuple, List

import numpy as np

from pgdrive.scene_creator.highway_vehicle.kinematics import Vehicle
from pgdrive.scene_manager.traffic_manager import TrafficManager


class BicycleVehicle(Vehicle):
    """
    This model is based on the following assumptions:

    - the vehicle is moving with a constant longitudinal speed
    - the steering input to front tires and the corresponding slip angles are small

    See https://pdfs.semanticscholar.org/bb9c/d2892e9327ec1ee647c30c320f2089b290c1.pdf, Chapter 3.
    """
    MASS: float = 1  # [kg]
    LENGTH_A: float = Vehicle.LENGTH / 2  # [m]
    LENGTH_B: float = Vehicle.LENGTH / 2  # [m]
    INERTIA_Z: float = 1 / 12 * MASS * (Vehicle.LENGTH**2 + 3 * Vehicle.WIDTH**2)  # [kg.m2]
    FRICTION_FRONT: float = 15.0 * MASS  # [N]
    FRICTION_REAR: float = 15.0 * MASS  # [N]

    MAX_ANGULAR_SPEED: float = 2 * np.pi  # [rad/s]
    MAX_SPEED: float = 15  # [m/s]

    def __init__(
        self,
        scene: TrafficManager,
        position: List,
        heading: float = 0,
        speed: float = 0,
        np_random: np.random.RandomState = None,
    ) -> None:
        super().__init__(scene, position, heading, speed, np_random=np_random)
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()

    @property
    def state(self) -> np.ndarray:
        return np.array(
            [
                [self.position[0]], [self.position[1]], [self.heading], [self.speed], [self.lateral_speed],
                [self.yaw_rate]
            ]
        )

    @property
    def derivative(self) -> np.ndarray:
        """
        See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)

        :return: the state derivative
        """
        delta_f = self.action["steering"]
        delta_r = 0
        theta_vf = np.arctan2(self.lateral_speed + self.LENGTH_A * self.yaw_rate, self.speed)  # (2.27)
        theta_vr = np.arctan2(self.lateral_speed - self.LENGTH_B * self.yaw_rate, self.speed)  # (2.28)
        f_yf = 2 * self.FRICTION_FRONT * (delta_f - theta_vf)  # (2.25)
        f_yr = 2 * self.FRICTION_REAR * (delta_r - theta_vr)  # (2.26)
        if abs(self.speed) < 1:  # Low speed dynamics: damping of lateral speed and yaw rate
            f_yf = -self.MASS * self.lateral_speed - self.INERTIA_Z / self.LENGTH_A * self.yaw_rate
            f_yr = -self.MASS * self.lateral_speed + self.INERTIA_Z / self.LENGTH_A * self.yaw_rate
        d_lateral_speed = 1 / self.MASS * (f_yf + f_yr) - self.yaw_rate * self.speed  # (2.21)
        d_yaw_rate = 1 / self.INERTIA_Z * (self.LENGTH_A * f_yf - self.LENGTH_B * f_yr)  # (2.22)
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array(
            [[speed[0]], [speed[1]], [self.yaw_rate], [self.action['acceleration']], [d_lateral_speed], [d_yaw_rate]]
        )

    @property
    def derivative_linear(self) -> np.ndarray:
        x = np.array([[self.lateral_speed], [self.yaw_rate]])
        u = np.array([[self.action['steering']]])
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()
        dx = self.A_lat @ x + self.B_lat @ u
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]], [speed[1]], [self.yaw_rate], [self.action['acceleration']], dx[0], dx[1]])

    def step(self, dt: float) -> None:
        self.clip_actions()
        derivative = self.derivative
        self._position += derivative[0:2, 0] * dt
        self.heading += self.yaw_rate * dt
        self.speed += self.action['acceleration'] * dt
        self.lateral_speed += derivative[4, 0] * dt
        self.yaw_rate += derivative[5, 0] * dt

        self.on_state_update()

    def clip_actions(self) -> None:
        super().clip_actions()
        # Required because of the linearisation
        self.action["steering"] = np.clip(self.action["steering"], -np.pi / 2, np.pi / 2)
        self.yaw_rate = np.clip(self.yaw_rate, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

    def lateral_lpv_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        State: [lateral speed v, yaw rate r]

        :return: lateral dynamics A0, phi, B such that dx = (A0 + theta^T phi)x + B u
        """
        B = np.array([[2 * self.FRICTION_FRONT / self.MASS], [self.FRICTION_FRONT * self.LENGTH_A / self.INERTIA_Z]])

        speed_body_x = self.speed
        A0 = np.array([[0, -speed_body_x], [0, 0]])

        if abs(speed_body_x) < 1:
            return A0, np.zeros((2, 2, 2)), B * 0

        phi = np.array(
            [
                [
                    [-2 / (self.MASS * speed_body_x), -2 * self.LENGTH_A / (self.MASS * speed_body_x)],
                    [
                        -2 * self.LENGTH_A / (self.INERTIA_Z * speed_body_x),
                        -2 * self.LENGTH_A**2 / (self.INERTIA_Z * speed_body_x)
                    ]
                ],
                [
                    [-2 / (self.MASS * speed_body_x), 2 * self.LENGTH_B / (self.MASS * speed_body_x)],
                    [
                        2 * self.LENGTH_B / (self.INERTIA_Z * speed_body_x),
                        -2 * self.LENGTH_B**2 / (self.INERTIA_Z * speed_body_x)
                    ]
                ],
            ]
        )
        return A0, phi, B

    def lateral_lpv_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: [lateral speed v, yaw rate r]

        :return: lateral dynamics A, B
        """
        A0, phi, B = self.lateral_lpv_structure()
        self.theta = np.array([self.FRICTION_FRONT, self.FRICTION_REAR])
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B

    def full_lateral_lpv_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]

        The system is linearized around psi = 0

        :return: lateral dynamics A, phi, B
        """
        A_lat, phi_lat, B_lat = self.lateral_lpv_structure()

        speed_body_x = self.speed
        A_top = np.array([[0, speed_body_x, 1, 0], [0, 0, 0, 1]])
        A0 = np.concatenate((A_top, np.concatenate((np.zeros((2, 2)), A_lat), axis=1)))
        phi = np.array(
            [
                np.concatenate((np.zeros((2, 4)), np.concatenate((np.zeros((2, 2)), phi_i), axis=1)))
                for phi_i in phi_lat
            ]
        )
        B = np.concatenate((np.zeros((2, 1)), B_lat))
        return A0, phi, B

    def full_lateral_lpv_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]

        The system is linearized around psi = 0

        :return: lateral dynamics A, B
        """
        A0, phi, B = self.full_lateral_lpv_structure()
        self.theta = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B
