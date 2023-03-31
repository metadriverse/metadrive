"""
Note (Zhenghao, Dec 27, 2020)

This is a evaluation scripts to run RLLib agent in MetaDrive.
Please install ray==1.0.0 first and prepare your trained agent.
This script is put here for reference only.
Future replacement for the expert might use this script.
"""
import logging
import os
import time
from typing import Dict

import matplotlib
import numpy as np
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.policy import Policy

from metadrive import GeneralizationRacing
from metadrive.constants import TerminationState

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        **kwargs
    ):
        arrive_dest = episode.last_info_for()[TerminationState.SUCCESS]
        crash_vehicle = episode.last_info_for()[TerminationState.CRASH_VEHICLE]
        out_of_road = episode.last_info_for()[TerminationState.OUT_OF_ROAD]
        max_step_rate = not (arrive_dest or crash_vehicle or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_vehicle_rate"] = float(crash_vehicle)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result[TerminationState.CRASH_VEHICLE] = np.nan
        result["out"] = np.nan
        result[TerminationState.MAX_STEP] = np.nan
        result["length"] = result["episode_len_mean"]
        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result[TerminationState.CRASH_VEHICLE] = result["custom_metrics"]["crash_vehicle_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result[TerminationState.MAX_STEP] = result["custom_metrics"]["max_step_rate_mean"]


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False, **kwargs):
    os.environ['OMP_NUM_THREADS'] = '1'

    if ray.__version__.split(".")[0] == "1":  # 1.0 version Ray
        if "redis_password" in kwargs:
            redis_password = kwargs.pop("redis_password")
            kwargs["_redis_password"] = redis_password

    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        local_mode=local_mode,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        **kwargs
    )
    print("Successfully initialize Ray!")
    try:
        print("Available resources: ", ray.available_resources())
    except Exception:
        pass


def get_trainer(checkpoint_path=None, extra_config=None, num_workers=10):
    config = dict(
        num_gpus=0,
        num_workers=num_workers,
        num_cpus_per_worker=1,
        horizon=1000,
        lr=0.0,
        batch_mode="complete_episodes",
        callbacks=DrivingCallbacks,
        # explore=False,  # Add this line to only use mean for action.

        # Setup the correct environment
        env=GeneralizationRacing,
        env_config=dict(num_scenarios=10000)
    )
    if extra_config:
        config.update(extra_config)
    trainer = PPOTrainer(config=config)
    if checkpoint_path is not None:
        trainer.restore(os.path.expanduser(checkpoint_path))
    return trainer


def evaluate(trainer, num_episodes=20):
    ret_reward = []
    ret_length = []
    ret_success_rate = []
    ret_out_rate = []
    ret_crash_vehicle_rate = []
    start = time.time()
    episode_count = 0
    while episode_count < num_episodes:
        rollouts = ParallelRollouts(trainer.workers, mode="bulk_sync")
        batch = next(rollouts)
        episodes = batch.split_by_episode()

        ret_reward.extend([e["rewards"].sum() for e in episodes])
        ret_length.extend([e.count for e in episodes])
        ret_success_rate.extend([e["infos"][-1][TerminationState.SUCCESS] for e in episodes])
        ret_out_rate.extend([e["infos"][-1][TerminationState.OUT_OF_ROAD] for e in episodes])
        ret_crash_vehicle_rate.extend([e["infos"][-1][TerminationState.CRASH_VEHICLE] for e in episodes])

        episode_count += len(episodes)
        print("Finish {} episodes".format(episode_count))

    ret = dict(
        reward=np.mean(ret_reward),
        length=np.mean(ret_length),
        success_rate=np.mean(ret_success_rate),
        out_rate=np.mean(ret_out_rate),
        crash_vehicle_rate=np.mean(ret_crash_vehicle_rate),
        episode_count=episode_count,
        time=time.time() - start,
    )
    print(
        "We collected {} episodes. Spent: {:.3f} s.\nResult: {}".format(
            episode_count,
            time.time() - start, {k: round(v, 3)
                                  for k, v in ret.items()}
        )
    )
    return ret


if __name__ == '__main__':
    ckpt = "checkpoint_417/checkpoint-417"
    num_episodes = 500

    initialize_ray(test_mode=False)
    trainer = get_trainer(ckpt, num_workers=18)
    ret = evaluate(trainer, num_episodes)
    print("Evaluation result: {}".format(ret))
