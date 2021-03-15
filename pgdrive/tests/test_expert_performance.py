import time

import numpy as np
from pgdrive import PGDriveEnv
from pgdrive.examples import expert, get_terminal_state


def _evaluate(env_config, num_episode):
    s = time.time()
    np.random.seed(0)
    env = PGDriveEnv(env_config)
    obs = env.reset()
    success_list, reward_list, ep_reward, ep_len, ep_count = [], [], 0, 0, 0
    while ep_count < num_episode:
        action = expert(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        ep_len += 1
        if done:
            ep_count += 1
            success_list.append(1 if get_terminal_state(info) == "Success" else 0)
            reward_list.append(ep_reward)
            ep_reward = 0
            ep_len = 0
            obs = env.reset()
    env.close()
    t = time.time() - s
    ep_reward_mean = sum(reward_list) / len(reward_list)
    success_rate = sum(success_list) / len(success_list)
    print(f"Finish {ep_count} episodes in {t:.3f} s. Episode reward: {ep_reward_mean}, success rate: {success_rate}.")
    return ep_reward_mean, success_rate


def test_expert_with_traffic(use_render=False):
    ep_reward, success_rate = _evaluate(
        dict(environment_num=1, start_seed=3, load_map_from_json=False, random_traffic=False, use_render=use_render),
        num_episode=3
    )
    assert 315 < ep_reward < 330, ep_reward
    assert success_rate == 1.0, success_rate


def test_expert_without_traffic():
    ep_reward, success_rate = _evaluate(
        dict(environment_num=1, start_seed=0, traffic_density=0, load_map_from_json=False, random_traffic=False),
        num_episode=3
    )
    assert 250 <= ep_reward <= 260, ep_reward
    assert success_rate == 1.0, success_rate


if __name__ == '__main__':
    test_expert_without_traffic()
    test_expert_with_traffic()
