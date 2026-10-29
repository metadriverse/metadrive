import torch
import numpy as np
import os.path as osp
from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.engine.logger import get_logger

ckpt_path = osp.join(osp.dirname(__file__), "expert_weights.npz")
_expert_weights = None
_expert_observation = None

logger = get_logger()


def obs_correction(obs):
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def numpy_to_torch(weights, device):
    """
    Convert numpy weights to torch tensors and move them to the specified device.
    :params:
        weights: numpy weights
        device: torch device
    :return:
        torch_weights: weights in torch tensor
    """
    torch_weights = {}
    for k in weights.keys():
        torch_weights[k] = torch.from_numpy(weights[k]).to(device)
    return torch_weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_expert(vehicle, deterministic=False, need_obs=False):
    """
    load weights by torch, use ppo actor to predict action
    :params:
        vehicle: vehicle instance
        deterministic: whether to use deterministic policy
        need_obs: whether to return observation
    :return:
        action: action predicted by expert
    """
    global _expert_weights
    global _expert_observation
    expert_obs_cfg = dict(
        lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
        side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
        lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
        random_agent_model=False
    )
    origin_obs_cfg = vehicle.config.copy()
    # TODO: some setting in origin cfg will not be covered, then they may change the obs shape
    with torch.no_grad():  # Disable gradient computation
        if _expert_weights is None:
            _expert_weights = numpy_to_torch(np.load(ckpt_path), device)
            config = get_global_config().copy()
            config["vehicle_config"].update(expert_obs_cfg)
            _expert_observation = LidarStateObservation(config)
            assert _expert_observation.observation_space.shape[0] == 275, "Observation not match"
            logger.info("Use Torch PPO expert.")

        vehicle.config.update(expert_obs_cfg)
        obs = _expert_observation.observe(vehicle)
        vehicle.config.update(origin_obs_cfg)
        obs = obs_correction(obs)
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)  # Convert to tensor and move to device
        weights = _expert_weights
        x = torch.matmul(obs, weights["default_policy/fc_1/kernel"]) + weights["default_policy/fc_1/bias"]
        x = torch.tanh(x)
        x = torch.matmul(x, weights["default_policy/fc_2/kernel"]) + weights["default_policy/fc_2/bias"]
        x = torch.tanh(x)
        x = torch.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
        x = x.squeeze(0).cpu()  # Move back to CPU and remove batch dimension
        mean, log_std = torch.split(x, 2, dim=-1)
        if deterministic:
            return (mean.numpy(), obs.cpu().numpy()) if need_obs else mean.numpy()
        std = torch.exp(log_std)
        action = torch.normal(mean, std).cpu()  # Move back to CPU
        return (action.numpy(), obs.cpu().numpy()) if need_obs else action.numpy()


def torch_value(obs, weights):
    """
    ppo critic to predict value
    :params:
        obs: observation
        weights: weights
    :return:
        value: value predicted by critic
    """
    with torch.no_grad():  # Disable gradient computation
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)  # Convert to tensor and move to device
        weights = _expert_weights
        x = torch.matmul(obs, weights["default_policy/fc_value_1/kernel"]) + weights["default_policy/fc_value_1/bias"]
        x = torch.tanh(x)
        x = torch.matmul(x, weights["default_policy/fc_value_2/kernel"]) + weights["default_policy/fc_value_2/bias"]
        x = torch.tanh(x)
        x = torch.matmul(x, weights["default_policy/value_out/kernel"]) + weights["default_policy/value_out/bias"]
        return x.squeeze(0).cpu().numpy()  # Move back to CPU and remove batch dimension
