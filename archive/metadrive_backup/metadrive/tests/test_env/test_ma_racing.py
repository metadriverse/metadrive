from collections import defaultdict

import pytest
from gymnasium.spaces import Box, Dict

from metadrive.constants import TerminationState
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv
from metadrive.policy.idm_policy import IDMPolicy


def _check_spaces_before_reset(env):
    a = set(env.config["agent_configs"].keys())
    b = set(env.observation_space.spaces.keys())
    c = set(env.action_space.spaces.keys())
    assert a == b == c
    _check_space(env)


def _check_spaces_after_reset(env, obs=None):
    a = set(env.config["agent_configs"].keys())
    b = set(env.observation_space.spaces.keys())
    assert a == b
    _check_shape(env)

    if obs:
        assert isinstance(obs, dict)
        assert set(obs.keys()) == a


def _check_shape(env):
    b = set(env.observation_space.spaces.keys())
    c = set(env.action_space.spaces.keys())
    d = set(env.agents.keys())
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
        assert len(env.agents) > 0
    if not (set(obs.keys()) == set(reward.keys()) == set(env.observation_space.spaces.keys())):
        raise ValueError
    assert env.observation_space.contains(obs)
    assert isinstance(reward, dict)
    assert isinstance(info, dict)
    assert isinstance(terminated, dict)
    assert isinstance(truncated, dict)

    return obs, reward, terminated, truncated, info


@pytest.mark.parametrize("num_agents", [1, 2, 3, 5, 8, 12])
def test_ma_racing_env_with_IDM(num_agents):
    config = dict(
        num_agents=num_agents,
        agent_policy=IDMPolicy,
        # use_render=True,
        # prefer_track_agent="agent11",
        debug=True
    )

    if num_agents > 2:
        config["map_config"] = {"exit_length": 60}

    env = MultiAgentRacingEnv(config)
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        assert env.observation_space.contains(obs)
        episode_reward_record = defaultdict(float)
        for step in range(3_000):
            act = {k: [1, 1] for k in env.agents.keys()}
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
                assert 2100 < max(episode_reward_record.values()) < 2300
                assert 2100 < min(episode_reward_record.values()) < 2300
                break
    finally:
        env.close()


@pytest.mark.parametrize("num_agents", [1, 3, 5, 8, 12])
def test_guardrail_collision_detection(num_agents, render=False):
    crash_sidewalk_penalty = 7.7
    config = dict(
        num_agents=num_agents,
        crash_sidewalk_done=True,
        crash_sidewalk_penalty=crash_sidewalk_penalty,
        # agent_policy=IDMPolicy,
        use_render=render,
        # prefer_track_agent="agent11",
        debug=True
    )

    if num_agents > 2:
        config["map_config"] = {"exit_length": 60}

    env = MultiAgentRacingEnv(config)
    crash_side = False
    crash_v = num_agents == 1
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        assert env.observation_space.contains(obs)
        episode_reward_record = defaultdict(float)
        for step in range(3_000):
            act = {k: [0, 1] for k in env.agents.keys()}
            o, r, tm, tc, i = _act(env, act)
            print(i)
            if render:
                env.render(mode="topdown")
            if step == 0:
                assert not any(tm.values())
                assert not any(tc.values())
            for k, v in r.items():
                episode_reward_record[k] += v
            for k in tm.keys():
                if k == "__all__":
                    continue
                if not tm[k]:
                    continue

                if i[k][TerminationState.CRASH_SIDEWALK]:
                    crash_side = True
                if i[k][TerminationState.CRASH_VEHICLE]:
                    crash_v = True
                    if not i[k][TerminationState.CRASH_SIDEWALK]:
                        assert not tm[k], "only crash should not terminate the env!"
                assert int(i[k][TerminationState.CRASH_SIDEWALK] + i[k][TerminationState.CRASH_VEHICLE]) >= 1
                assert not i[k][TerminationState.SUCCESS]
                assert not i[k][TerminationState.MAX_STEP]
                # assert not i[k][TerminationState.CRASH] # crash sidewalk will be counted as crash as well
                assert not i[k][TerminationState.OUT_OF_ROAD]
                # Crash vehicle penalty has higher priority than the crash_sidewalk_penalty
                assert r[k] == -crash_sidewalk_penalty or r[k] == -10
            if tm["__all__"]:
                print("Episode finished at step: ", step)
                print("Episodic return: ", episode_reward_record)
                print(
                    f"Max return {max(episode_reward_record.values())}, Min return {min(episode_reward_record.values())}"
                )
                break
        assert crash_side and crash_v
    finally:
        env.close()


if __name__ == '__main__':
    # test_ma_racing_env_with_IDM(12)
    test_guardrail_collision_detection(4, render=False)
