import time
import pytest

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.constants import DEFAULT_AGENT
from metadrive.examples import expert, get_terminal_state


def _evaluate(env_config, num_episode, has_traffic=True, need_on_same_lane=True):
    s = time.time()
    np.random.seed(0)
    env_config["random_spawn_lane_index"] = False
    env = MetaDriveEnv(env_config)
    lane_idx_need_to_stay = 0
    try:
        obs, _ = env.reset()
        lidar_success = False
        success_list, reward_list, ep_reward, ep_len, ep_count = [], [], 0, 0, 0
        while ep_count < num_episode:
            action = expert(env.agent, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if need_on_same_lane:
                assert lane_idx_need_to_stay == env.agent.lane_index[-1], "Not one the same lane"
            # double check lidar
            if env.config["use_render"]:
                env.render(text={"lane_index": env.agent.lane_index, "step": env.episode_step})
            lidar = [True if p == 1.0 else False for p in env.observations[DEFAULT_AGENT].cloud_points]
            if not all(lidar):
                lidar_success = True
            ep_reward += reward
            ep_len += 1
            if terminated or truncated:
                ep_count += 1
                success_list.append(1 if get_terminal_state(info) == "Success" else 0)
                reward_list.append(ep_reward)
                ep_reward = 0
                ep_len = 0
                env.config["agent_configs"]["default_agent"]["spawn_lane_index"] = (">", ">>", len(reward_list) % 3)
                lane_idx_need_to_stay = len(reward_list) % 3
                obs, _ = env.reset()
                if has_traffic:
                    assert lidar_success
                lidar_success = False
        env.close()
        t = time.time() - s
        ep_reward_mean = sum(reward_list) / len(reward_list)
        success_rate = sum(success_list) / len(success_list)
        # print(
        #     f"Finish {ep_count} episodes in {t:.3f} s. Episode reward: {ep_reward_mean}, success rate: {success_rate}."
        # )
    finally:
        env.close()
    return ep_reward_mean, success_rate


@pytest.mark.parametrize("plane", [True, False], ids=["plane", "mesh"])
def test_expert_with_traffic(plane, use_render=False):
    ep_reward, success_rate = _evaluate(
        dict(
            num_scenarios=1,
            map="CCC",
            use_mesh_terrain=plane,
            start_seed=2,
            random_traffic=False,
            # debug_static_world=True,
            # debug=True,
            use_render=use_render,
            vehicle_config=dict(show_lidar=True),
        ),
        need_on_same_lane=False,
        num_episode=10
    )

    # We change the ego vehicle dynamics! So the expert is not reliable anymore!
    assert 300 < ep_reward < 350, ep_reward
    # assert success_rate == 1.0, success_rate


def test_expert_without_traffic(render=False):
    ep_reward, success_rate = _evaluate(
        dict(
            num_scenarios=1,
            random_agent_model=False,
            map="CCC",
            use_render=render,
            start_seed=2,
            # debug_static_world=True,
            # debug=True,
            traffic_density=0,
            random_traffic=False,
        ),
        num_episode=10,
        has_traffic=False
    )
    # We change the ego vehicle dynamics! So the expert is not reliable anymore!
    assert success_rate == 1.0, success_rate

    assert 300 <= ep_reward <= 350, ep_reward


def test_expert_in_intersection(render=False):
    ep_reward, success_rate = _evaluate(
        dict(
            num_scenarios=1,
            random_agent_model=False,
            map="XTXTS",
            use_render=render,
            start_seed=2,
            # debug_static_world=True,
            # debug=True,
            traffic_density=0,
            random_traffic=False,
        ),
        num_episode=10,
        has_traffic=False
    )
    # We change the ego vehicle dynamics! So the expert is not reliable anymore!
    assert success_rate == 1.0, success_rate

    assert ep_reward > 400, ep_reward


if __name__ == '__main__':
    """
    LQY: I fixed a action bug in StateObservation, which may harm expert performance!
    """
    # test_expert_in_intersection(True)
    # test_expert_without_traffic(True)
    test_expert_with_traffic(plane=False, use_render=False)
