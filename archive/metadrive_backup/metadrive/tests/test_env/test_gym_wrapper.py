import copy

from typing import Union, Any, Dict

import numpy as np

import gym
import gym.spaces
import gymnasium

from metadrive import MetaDriveEnv
from metadrive.envs.gym_wrapper import createGymWrapper, gymToGymnasium, gymnasiumToGym


def test_conversion_functions():
    gym_space_box = gym.spaces.Box(0, 1, shape=(1, 2, 3), dtype=np.float32)
    gymnasium_space_box = gymnasium.spaces.Box(0, 1, shape=(1, 2, 3), dtype=np.float32)
    assert gymToGymnasium(gym_space_box) == gymnasium_space_box
    assert gymnasiumToGym(gymnasium_space_box) == gym_space_box

    gym_space_discrete = gym.spaces.Discrete(5)
    gymnasium_space_discrete = gymnasium.spaces.Discrete(5)
    assert gymToGymnasium(gym_space_discrete) == gymnasium_space_discrete
    assert gymnasiumToGym(gymnasium_space_discrete) == gym_space_discrete

    gym_space_multi_discrete = gym.spaces.MultiDiscrete([5, 6, 7])
    gymnasium_space_multi_discrete = gymnasium.spaces.MultiDiscrete([5, 6, 7])
    assert gymToGymnasium(gym_space_multi_discrete) == gymnasium_space_multi_discrete
    assert gymnasiumToGym(gymnasium_space_multi_discrete) == gym_space_multi_discrete

    gym_space_tuple = gym.spaces.Tuple([gym_space_box, gym_space_discrete])
    gymnasium_space_tuple = gymnasium.spaces.Tuple([gymnasium_space_box, gymnasium_space_discrete])
    assert gymToGymnasium(gym_space_tuple) == gymnasium_space_tuple
    assert gymnasiumToGym(gymnasium_space_tuple) == gym_space_tuple

    gym_space_dict = gym.spaces.Dict({"a": gym_space_box, "b": gym_space_discrete})
    gymnasium_space_dict = gymnasium.spaces.Dict({"a": gymnasium_space_box, "b": gymnasium_space_discrete})
    assert gymToGymnasium(gym_space_dict) == gymnasium_space_dict
    assert gymnasiumToGym(gymnasium_space_dict) == gym_space_dict


# test that mode is removed from args
class NoModeGymnasiumInnerClass(gymnasium.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.action_space = gymnasium.spaces.Discrete(5)
        self.observation_space = gymnasium.spaces.Box(0, 1, shape=(1, 2, 3), dtype=np.float32)

    def render(self, **kwargs):
        assert 'mode' not in kwargs.keys()


# def test_gym_wrapper_removes_mode():
#     gym_env = createGymWrapper(NoModeGymnasiumInnerClass)(config={})
#     gym_env.render(mode=1)


class ConfigOverrideGymnasiumInnerClass(gymnasium.Env):
    @classmethod
    def default_config(cls) -> dict:
        return {"a": 1, "b": 2}

    def __init__(self, config: dict):
        super().__init__()
        assert self.default_config() == {"a": 1, "b": 2}


def test_config_override():
    # should not raise
    createGymWrapper(ConfigOverrideGymnasiumInnerClass)(config={})

    # if we override the default config it should be visible in the (non-overriden) constructor
    try:
        GymWrapperedClass = createGymWrapper(ConfigOverrideGymnasiumInnerClass)

        class OverridenGymWrapperedClass_1(GymWrapperedClass):
            @classmethod
            def default_config(cls) -> dict:
                return {"a": 3, "b": 4}

        # try to initialize (should raise)
        OverridenGymWrapperedClass_1(config={})
    except AssertionError:
        pass
    else:
        raise AssertionError("OverridenGymWrapperedClass_1 should have raised an AssertionError")

    # if we override the default config it should be visible in the overriden constructor
    GymWrapperedClass = createGymWrapper(ConfigOverrideGymnasiumInnerClass)

    class OverridenGymWrapperedClass_2(GymWrapperedClass):
        @classmethod
        def default_config(cls) -> dict:
            return {"a": 3, "b": 4}

        def __init__(self, config: dict):
            assert self.default_config() == {"a": 3, "b": 4}

    # try to initialize (should not raise)
    OverridenGymWrapperedClass_2(config={})


class SingleAgentEnvironment(gymnasium.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.action_space = gymnasium.spaces.Box(-1, 1, shape=(2, ))
        self.observation_space = gymnasium.spaces.Box(0, 1, shape=(1, 2, 3), dtype=np.float32)

        self.active = True
        self.steps = 0
        self.reward_source = gymnasium.spaces.Box(0, 100, dtype=np.float32)

    def step(self, action: np.ndarray):
        assert self.active
        assert self.action_space.contains(action)

        self.steps += 1

        # truncate if steps above 50
        if self.steps < 50:
            # terminate if action is less than 0
            if action[0] < 0:
                self.active = False
                return self.observation_space.sample(), float(self.reward_source.sample().item()), True, False, {}
            else:
                return self.observation_space.sample(), float(self.reward_source.sample().item()), False, False, {}
        else:
            self.active = False
            return self.observation_space.sample(), float(self.reward_source.sample().item()), False, True, {}

    def reset(self, seed: Union[int, None] = None):
        super().reset(seed=seed)
        self.steps = 0
        self.observation_space.seed(seed)
        self.reward_source.seed(seed)
        self.active = True
        return self.observation_space.sample(), {}


def test_gym_wrapper_single_agent():
    # collect data from gymnasium env
    gymnasium_env = SingleAgentEnvironment(config={})
    gym_env = createGymWrapper(SingleAgentEnvironment)(config={})

    assert gymToGymnasium(gym_env.observation_space) == gymnasium_env.observation_space
    assert gymToGymnasium(gym_env.action_space) == gymnasium_env.action_space

    o_gymnasium, _ = gymnasium_env.reset(seed=0)
    o_gym = gym_env.reset(seed=0)

    assert isinstance(o_gymnasium, np.ndarray)
    assert isinstance(o_gym, np.ndarray)

    assert np.allclose(o_gymnasium, o_gym)

    # test will end due to truncation
    while True:
        o_gymnasium, r_gymnasium, tm_gymnasium, tc_gymnasium, _ = gymnasium_env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        o_gym, r_gym, d_gym, _ = gym_env.step(np.array([0.0, 0.0], dtype=np.float32))

        assert isinstance(o_gymnasium, np.ndarray)
        assert isinstance(o_gym, np.ndarray)

        assert np.allclose(o_gymnasium, o_gym)

        assert isinstance(r_gymnasium, float)
        assert isinstance(r_gym, float)

        assert np.allclose(r_gymnasium, r_gym)

        assert d_gym == tm_gymnasium or tc_gymnasium

        if d_gym:
            break

    # now test termination
    o_gymnasium, _ = gymnasium_env.reset(seed=42)
    o_gym = gym_env.reset(seed=42)

    assert isinstance(o_gymnasium, np.ndarray)
    assert isinstance(o_gym, np.ndarray)

    assert np.allclose(o_gymnasium, o_gym)

    # this should terminate
    o_gymnasium, r_gymnasium, tm_gymnasium, tc_gymnasium, _ = gymnasium_env.step(np.array([-0.5, 0], dtype=np.float32))
    o_gym, r_gym, d_gym, _ = gym_env.step(np.array([-0.5, 0], dtype=np.float32))

    assert isinstance(o_gymnasium, np.ndarray)
    assert isinstance(o_gym, np.ndarray)
    assert np.allclose(o_gymnasium, o_gym)

    assert isinstance(r_gymnasium, float)
    assert isinstance(r_gym, float)
    assert np.allclose(r_gymnasium, r_gym)

    assert not tc_gymnasium
    assert tm_gymnasium

    assert d_gym


class MultiAgentEnvironment(gymnasium.Env):
    def __init__(self, config: dict):
        super().__init__()

        # validate config
        assert "num_agents" in config
        assert isinstance(config["num_agents"], int)
        assert "horizon" in config
        assert isinstance(config["horizon"], int)

        self.num_agents = config["num_agents"]
        self.horizon = config["horizon"]

        self.action_space = gymnasium.spaces.Box(-1, 1, shape=(2, ))
        self.observation_space = gymnasium.spaces.Box(0, 1, shape=(1, 2, 3), dtype=np.float32)

        self.steps = 0
        self.reward_source = gymnasium.spaces.Box(0, 100, dtype=np.float32)

    def _generate_observations(self) -> Dict[str, np.ndarray]:
        return {agent: self.observation_space.sample() for agent in sorted(self.agents)}

    def _generate_rewards(self) -> Dict[str, float]:
        return {agent: float(self.reward_source.sample().item()) for agent in sorted(self.agents)}

    def _validate_actions(self, actions: Dict[str, np.ndarray]) -> bool:
        if set(actions.keys()) != self.agents:
            return False
        return all(self.action_space.contains(action) for action in actions.values())

    def step(self, action: Dict[str, np.ndarray]):
        assert len(self.agents) > 0
        assert self._validate_actions(action)

        obss = self._generate_observations()
        rews = self._generate_rewards()
        # terminate if action is less than 0
        terminateds = {agent: action[agent][0] < 0 for agent in self.agents}
        # truncate if steps above horizon
        truncateds = {agent: self.steps >= self.horizon and not terminateds[agent] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # increment steps
        self.steps += 1
        # remove dead agents
        self.agents = {agent for agent in self.agents if not terminateds[agent] and not truncateds[agent]}

        # add __all__ key to terminateds and truncateds
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        return obss, rews, terminateds, truncateds, infos

    def reset(self, seed: Union[int, None] = None):
        super().reset(seed=seed)
        self.steps = 0
        self.agents = {f"agent_{i}" for i in range(self.num_agents)}
        self.observation_space.seed(seed)
        self.reward_source.seed(seed)
        return self._generate_observations(), {agent: {} for agent in self.agents}


def assert_allclose_dict(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]):
    a_keys = sorted(a.keys())
    b_keys = sorted(b.keys())

    assert a_keys == b_keys

    for key in a_keys:
        assert np.allclose(a[key], b[key])


def test_gym_wrapper_multi_agent():
    # test multi agent metadrive environment
    gymnasium_env = MultiAgentEnvironment(config={"num_agents": 5, "horizon": 50})
    gym_env = createGymWrapper(MultiAgentEnvironment)(config={"num_agents": 5, "horizon": 50})

    assert gymToGymnasium(gym_env.observation_space) == gymnasium_env.observation_space
    assert gymToGymnasium(gym_env.action_space) == gymnasium_env.action_space

    o_gymnasium, _ = gymnasium_env.reset(seed=0)
    o_gym = gym_env.reset(seed=0)

    assert isinstance(o_gymnasium, dict)
    assert isinstance(o_gym, dict)
    assert_allclose_dict(o_gymnasium, o_gym)

    # test will end due to truncation
    while True:
        # action dict: generate [0, 0] for each agent
        a_dict = {agent: np.array([0.0, 0.0], dtype=np.float32) for agent in gymnasium_env.agents}

        o_gymnasium, r_gymnasium, tm_gymnasium, tc_gymnasium, _ = gymnasium_env.step(a_dict)
        o_gym, r_gym, d_gym, _ = gym_env.step(a_dict)

        assert isinstance(o_gymnasium, dict)
        assert isinstance(o_gym, dict)
        assert_allclose_dict(o_gymnasium, o_gym)

        assert isinstance(r_gymnasium, dict)
        assert isinstance(r_gym, dict)
        assert_allclose_dict(r_gymnasium, r_gym)

        # show correct termination behavior:
        assert isinstance(tm_gymnasium, dict)
        assert isinstance(tc_gymnasium, dict)
        agentnames = list(set(tm_gymnasium) | set(tc_gymnasium))

        for agent in agentnames:
            assert d_gym[agent] == tm_gymnasium[agent] or tc_gymnasium[agent]

        if all(d_gym.values()):
            break

    # now test termination
    o_gymnasium, _ = gymnasium_env.reset(seed=42)
    o_gym = gym_env.reset(seed=42)

    assert isinstance(o_gymnasium, dict)
    assert isinstance(o_gym, dict)
    assert_allclose_dict(o_gymnasium, o_gym)

    # test will end due to termination of the last agent
    while True:
        # action dict: generate [0, 0] for each agent except the last one (which is [-0.5, 0], so that we can test termination)
        a_dict = {agent: np.array([0.0, 0.0], dtype=np.float32) for agent in gymnasium_env.agents}
        a_dict[sorted(a_dict.keys())[-1]] = np.array([-0.5, 0], dtype=np.float32)

        o_gymnasium, r_gymnasium, tm_gymnasium, tc_gymnasium, _ = gymnasium_env.step(a_dict)
        o_gym, r_gym, d_gym, _ = gym_env.step(a_dict)

        assert isinstance(o_gymnasium, dict)
        assert isinstance(o_gym, dict)
        assert_allclose_dict(o_gymnasium, o_gym)

        assert isinstance(r_gymnasium, dict)
        assert isinstance(r_gym, dict)
        assert_allclose_dict(r_gymnasium, r_gym)

        # show correct termination behavior:
        assert isinstance(tm_gymnasium, dict)
        assert isinstance(tc_gymnasium, dict)
        agentnames = list(set(tm_gymnasium) | set(tc_gymnasium))

        for agent in agentnames:
            assert d_gym[agent] == tm_gymnasium[agent] or tc_gymnasium[agent]

        if all(d_gym.values()):
            break


if __name__ == '__main__':
    test_conversion_functions()
    test_gym_wrapper_removes_mode()
    test_config_override()
    test_gym_wrapper_single_agent()
    test_gym_wrapper_multi_agent()
