from collections import defaultdict

import pytest
from gymnasium.spaces import Box, Dict

from metadrive.constants import TerminationState
from metadrive.envs.marl_envs.marl_racing_env_complex import MultiAgentRacingEnv
from metadrive.policy.idm_policy import IDMPolicy


def _check_spaces_before_reset(env):
    a = set(env.config["target_vehicle_configs"].keys())
    b = set(env.observation_space.spaces.keys())
    c = set(env.action_space.spaces.keys())
    assert a == b == c
    _check_space(env)


def _check_spaces_after_reset(env, obs=None):
    a = set(env.config["target_vehicle_configs"].keys())
    b = set(env.observation_space.spaces.keys())
    assert a == b
    _check_shape(env)

    if obs:
        assert isinstance(obs, dict)
        assert set(obs.keys()) == a


def _check_shape(env):
    b = set(env.observation_space.spaces.keys())
    c = set(env.action_space.spaces.keys())
    d = set(env.vehicles.keys())
    e = set(env.engine.agents.keys())
    f = set([k for k in env.observation_space.spaces.keys() if not env.dones[k]])
    assert d == e == f, (b, c, d, e, f)
    assert c.issuperset(d)
    _check_space(env)


def _check_space(env):
    assert isinstance(env.action_space, Dict)
    assert isinstance(env.observation_space, Dict)
    o_shape = None
    for k, s in env.observation_space.spaces.items():
        assert isinstance(s, Box)
        if o_shape is None:
            o_shape = s.shape
        assert s.shape == o_shape
    a_shape = None
    for k, s in env.action_space.spaces.items():
        assert isinstance(s, Box)
        if a_shape is None:
            a_shape = s.shape
        assert s.shape == a_shape


def _act(env, action):
    assert env.action_space.contains(action)
    obs, reward, terminated, truncated, info = env.step(action)
    _check_shape(env)
    if not terminated["__all__"]:
        assert len(env.vehicles) > 0
    if not (set(obs.keys()) == set(reward.keys()) == set(env.observation_space.spaces.keys())):
        raise ValueError
    assert env.observation_space.contains(obs)
    assert isinstance(reward, dict)
    assert isinstance(info, dict)
    assert isinstance(terminated, dict)
    assert isinstance(truncated, dict)

    return obs, reward, terminated, truncated, info


@pytest.mark.parametrize("num_agents", [1, 3, 5, 8, 12])
def test_ma_racing_env_with_IDM(num_agents):
    env = MultiAgentRacingEnv(dict(
        num_agents=num_agents,
        agent_policy=IDMPolicy,
    ))
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        assert env.observation_space.contains(obs)
        episode_reward_record = defaultdict(float)
        for step in range(3_000):
            act = {k: [1, 1] for k in env.vehicles.keys()}
            o, r, tm, tc, i = _act(env, act)
            # env.render(mode="topdown")
            if step == 0:
                assert not any(tm.values())
                assert not any(tc.values())
            for k, v in r.items():
                episode_reward_record[k] += v
            if tm["__all__"]:
                print("Episode finished at step: ", step)
                print("Episodic return: ", episode_reward_record)
                print(
                    f"Max return {max(episode_reward_record.values())}, Min return {min(episode_reward_record.values())}"
                )

                for k, v in tm.items():
                    assert v
                for k in i.keys():
                    assert i[k][TerminationState.SUCCESS]
                    assert not i[k][TerminationState.MAX_STEP]
                    assert not i[k][TerminationState.CRASH_VEHICLE]
                    assert not i[k][TerminationState.CRASH]
                    assert not i[k][TerminationState.OUT_OF_ROAD]
                assert 1450 < max(episode_reward_record.values()) < 1550
                assert 1400 < min(episode_reward_record.values()) < 1500
                break
    finally:
        env.close()


if __name__ == '__main__':
    test_ma_racing_env_with_IDM(1)
