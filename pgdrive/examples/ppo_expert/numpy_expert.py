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

ckpt_path = osp.join(osp.dirname(__file__), "expert_weights.npz")
_expert_weights = None


def expert(obs):
    global _expert_weights
    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
    weights = _expert_weights
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/fc_1/kernel"]) + weights["default_policy/fc_1/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_2/kernel"]) + weights["default_policy/fc_2/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    std = np.exp(log_std)
    action = np.random.normal(mean, std)
    ret = action
    ret = np.clip(ret, -1.0, 1.0)
    return ret


if __name__ == '__main__':
    for i in range(100):
        print("Weights? ", type(_expert_weights))
        ret = expert(np.clip(np.random.normal(0.5, 1, size=(275, )), 0.0, 1.0))
        print("Return: ", ret)
