"""
The existing layer names and shapes in numpy file:
(note that the terms with "value" in it are removed to save space).
default_policy/fc_1/kernel (275, 256)
default_policy/fc_1/bias (256,)
default_policy/fc_value_1/kernel (275, 256)
default_policy/fc_value_1/bias (256,)
default_policy/fc_2/kernel (256, 256)
default_policy/fc_2/bias (256,)
default_policy/fc_value_2/kernel (256, 256)
default_policy/fc_value_2/bias (256,)
default_policy/fc_out/kernel (256, 4)
default_policy/fc_out/bias (4,)
default_policy/value_out/kernel (256, 1)
default_policy/value_out/bias (1,)
"""

import os.path as osp

import numpy as np

from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.state_obs import LidarStateObservation

ckpt_path = osp.join(osp.dirname(__file__), "expert_weights.npz")
_expert_weights = None
_expert_observation = None


def expert(vehicle, deterministic=False, need_obs=False):
    global _expert_weights
    global _expert_observation
    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
        v_config = get_global_config()["vehicle_config"].copy()
        v_config.update(
            dict(
                lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
                random_agent_model=False
            )
        )
        _expert_observation = LidarStateObservation(v_config)
        assert _expert_observation.observation_space.shape[0] == 275, "Observation not match"
    obs = _expert_observation.observe(vehicle)
    weights = _expert_weights
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/fc_1/kernel"]) + weights["default_policy/fc_1/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_2/kernel"]) + weights["default_policy/fc_2/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    if deterministic:
        return (mean, obs) if need_obs else mean
    std = np.exp(log_std)
    action = np.random.normal(mean, std)
    ret = action
    # ret = np.clip(ret, -1.0, 1.0) all clip should be implemented in env!
    return (ret, obs) if need_obs else ret


def load_weights(path: str):
    """
    Load NN weights
    :param path: weights file path path
    :return: NN weights object
    """
    try:
        model = np.load(path)
        return model
    except FileNotFoundError:
        print("Can not find {}, didn't load anything".format(path))
        return None


def value(obs, weights):
    """
    Given weights, return the evaluation to one state/obseration
    :param obs: observation
    :param weights: variable weights of NN
    :return: value
    """
    if weights is None:
        return 0
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/fc_value_1/kernel"]) + weights["default_policy/fc_value_1/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_value_2/kernel"]) + weights["default_policy/fc_value_2/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/value_out/kernel"]) + weights["default_policy/value_out/bias"]
    ret = x.reshape(-1)
    return ret


# if __name__ == '__main__':
#     for i in range(100):
#         print("Weights? ", type(_expert_weights))
#         ret = expert(np.clip(np.random.normal(0.5, 1, size=(275,)), 0.0, 1.0))
#         print("Return: ", ret)
