import numpy as np

from metadrive.component.vehicle_model.bicycle_model import BicycleModel
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger
from metadrive.utils.math_utils import norm


def predict(current_state, actions, model):
    model.reset(*current_state)
    for action in actions:
        model.predict(0.1, action)
    return model.state


def _test_bicycle_model():
    horizon = 10
    setup_logger(True)
    env = MetaDriveEnv(
        {
            "environment_num": 1,
            "traffic_density": .0,
            "use_render": True,
            # "manual_control": True,
            "map": "CCCC",
            "vehicle_config": {
                "enable_reverse": False,
            }
        }
    )
    bicycle_model = BicycleModel()
    o = env.reset()
    vehicle = env.current_track_vehicle
    v_dir = vehicle.velocity_direction
    bicycle_model.reset(*vehicle.position, vehicle.speed, vehicle.heading_theta, np.arctan2(v_dir[1], v_dir[0]))
    actions = []
    for steering in [1.0, 0.8, 0.6, 0.4, 0.2, 0]:
        for dir in [-1, 1]:
            s = dir * steering
            for throttle in [1.0, 0.8, 0.6, 0.4, 0.2, 0, -0.5]:
                actions += [[s, throttle]] * 20
    predict_states = []
    for s in range(len(actions)):
        vehicle = env.current_track_vehicle
        v_dir = vehicle.velocity_direction
        predict_states.append(
            predict(
                current_state=(
                    *env.current_track_vehicle.position, env.current_track_vehicle.speed,
                    env.current_track_vehicle.heading_theta, np.arctan2(v_dir[1], v_dir[0])
                ),
                actions=[actions[i] for i in range(s, s + horizon)],
                model=bicycle_model
            )
        )
        o, r, d, info = env.step(actions[s])
        index = s - horizon
        if index >= 0:
            state = predict_states[index]
            print(norm(state["x"] - vehicle.position[0], state["y"] - vehicle.position[1]))


if __name__ == "__main__":
    _test_bicycle_model()
