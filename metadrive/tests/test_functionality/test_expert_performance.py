import time

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.constants import DEFAULT_AGENT
from metadrive.examples import expert, get_terminal_state


def _evaluate(env_config, num_episode, has_traffic=True):
    s = time.time()
    np.random.seed(0)
    env = MetaDriveEnv(env_config)
    try:
        obs = env.reset()
        lidar_success = False
        success_list, reward_list, ep_reward, ep_len, ep_count = [], [], 0, 0, 0
        while ep_count < num_episode:
            action = expert(env.vehicle, deterministic=True)
            obs, reward, done, info = env.step(action)
            # double check lidar
            lidar = [True if p == 1.0 else False for p in env.observations[DEFAULT_AGENT].cloud_points]
            if not all(lidar):
                lidar_success = True
            ep_reward += reward
            ep_len += 1
            if done:
                ep_count += 1
                success_list.append(1 if get_terminal_state(info) == "Success" else 0)
                reward_list.append(ep_reward)
                ep_reward = 0
                ep_len = 0
                obs = env.reset()
                if has_traffic:
                    assert lidar_success
                lidar_success = False
        env.close()
        t = time.time() - s
        ep_reward_mean = sum(reward_list) / len(reward_list)
        success_rate = sum(success_list) / len(success_list)
        print(
            f"Finish {ep_count} episodes in {t:.3f} s. Episode reward: {ep_reward_mean}, success rate: {success_rate}."
        )
    finally:
        env.close()
    return ep_reward_mean, success_rate


def test_expert_with_traffic(use_render=False):
    ep_reward, success_rate = _evaluate(
        dict(
            environment_num=1,
            map="CCC",
            start_seed=2,
            random_traffic=False,
            use_render=use_render,
            vehicle_config=dict(show_lidar=True),
        ),
        num_episode=10
    )

    # We change the ego vehicle dynamics! So the expert is not reliable anymore!
    assert 300 < ep_reward < 350, ep_reward
    # assert success_rate == 1.0, success_rate


def test_expert_without_traffic():
    ep_reward, success_rate = _evaluate(
        dict(
            environment_num=1,
            random_agent_model=False,
            map="CCC",
            use_render=False,
            start_seed=2,
            traffic_density=0,
            random_traffic=False,
        ),
        num_episode=10,
        has_traffic=False
    )
    assert 300 <= ep_reward <= 350, ep_reward

    # We change the ego vehicle dynamics! So the expert is not reliable anymore!
    assert success_rate == 1.0, success_rate


if __name__ == '__main__':
    test_expert_without_traffic()
    # test_expert_with_traffic(use_render=False)
