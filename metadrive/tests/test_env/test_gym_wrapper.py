import os

from typing import Union, Any

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


def assert_allclose_dict(a:dict[str, np.ndarray], b:dict[str, np.ndarray]):
    # assert the keys are the same
    ak = sorted(a.keys())
    bk = sorted(a.keys())
    assert ak == bk
    for k in ak:
        assert np.allclose(a[k], b[k])


def test_gym_wrapper_simple():
    # test very simple single agent environment
    class SingleAgentGymnasiumInnerClass(gymnasium.Env):
        def __init__(self, config: dict):
            super().__init__()
            self.action_space = gymnasium.spaces.Discrete(5)
            self.observation_space = gymnasium.spaces.Box(0, 1, shape=(1, 2, 3), dtype=np.float32)

        def step(self, action):
            return self.observation_space.sample(), 0, False, False, {}

        def reset(self, seed: Union[int,None]=None):
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            return self.observation_space.sample(), {}


    gymnasium_env = SingleAgentGymnasiumInnerClass(config={})
    gym_env = createGymWrapper(SingleAgentGymnasiumInnerClass)(config={})
    
    assert gymToGymnasium(gym_env.observation_space) == gymnasium_env.observation_space
    assert gymToGymnasium(gym_env.action_space) == gymnasium_env.action_space

    o_gymnasium, _ = gymnasium_env.reset(seed=0)
    o_gym = gym_env.reset(seed=0)

    assert np.allclose(o_gymnasium, o_gym)

    o_gymnasium, _, _, _, _= gymnasium_env.step([0, 1])
    o_gym, _, _, _ = gym_env.step([0, 1])
    
    assert np.allclose(o_gymnasium, o_gym)

def test_gym_wrapper_single_agent_metadrive():
    # test single agent metadrive environment
    gymnasium_env = MetaDriveEnv(config={"num_scenarios": 1})
    gym_env = createGymWrapper(MetaDriveEnv)(config={"num_scenarios": 1})

    assert gymToGymnasium(gym_env.observation_space) == gymnasium_env.observation_space
    assert gymToGymnasium(gym_env.action_space) == gymnasium_env.action_space

    o_gymnasium, _ = gymnasium_env.reset(seed=0)
    o_gym = gym_env.reset(seed=0)

    assert isinstance(o_gymnasium, np.ndarray)
    assert isinstance(o_gym, np.ndarray)

    assert np.allclose(o_gymnasium, o_gym)

    while True:
        o_gymnasium, r_gymnasium, tm_gymnasium, tc_gymnasium, _ = gymnasium_env.step([0, 1])
        o_gym, r_gym, d_gym, _ = gym_env.step([0, 1])

        assert isinstance(o_gymnasium, np.ndarray)
        assert isinstance(o_gym, np.ndarray)

        assert np.allclose(o_gymnasium, o_gym)

        assert isinstance(r_gymnasium, float)
        assert isinstance(r_gym, float)

        assert np.allclose(r_gymnasium, r_gym)

        assert d_gym == tm_gymnasium or tc_gymnasium

        if d_gym or tm_gymnasium or tc_gymnasium:
            break

def test_gym_wrapper_multi_agent_metadrive():
    # test multi agent metadrive environment
    gymnasium_env = MetaDriveEnv(config={"num_scenarios": 1, "start_seed": 0, "num_agents": 2})
    gym_env = createGymWrapper(MetaDriveEnv)(config={"num_scenarios": 1, "start_seed": 0, "num_agents": 2})

    assert gymToGymnasium(gym_env.observation_space) == gymnasium_env.observation_space
    assert gymToGymnasium(gym_env.action_space) == gymnasium_env.action_space

    o_gymnasium, _ = gymnasium_env.reset(seed=0)
    o_gym = gym_env.reset(seed=0)

    assert isinstance(o_gymnasium, dict)
    assert isinstance(o_gym, dict)
    assert_allclose_dict(o_gymnasium, o_gym)    

    while True:
        o_gymnasium, r_gymnasium, tm_gymnasium, tc_gymnasium, _ = gymnasium_env.step([[0, 1], [0, 1]])
        o_gym, r_gym, d_gym, _ = gym_env.step([[0, 1], [0, 1]])

        assert_allclose_dict(o_gymnasium, o_gym)

        # show correct termination behavior:
        assert isinstance(tm_gymnasium, dict)
        assert isinstance(tc_gymnasium, dict)
        agentnames = list(set(tm_gymnasium) | set(tc_gymnasium))

        alldone = True
        for agent in agentnames:
            assert d_gym[agent] == tm_gymnasium[agent] or tc_gymnasium[agent]
            
            # keep going if even one of these is not done
            if not (tm_gymnasium[agent] or tc_gymnasium[agent]):
                alldone = False

        if all(d_gym.values()) or alldone:
            break


def test_gym_wrapper_removes_mode():
    # test that mode is removed from args
    class NoModeGymnasiumInnerClass(gymnasium.Env):
        def __init__(self, config: dict):
            super().__init__()
            self.action_space = gymnasium.spaces.Discrete(5)
            self.observation_space = gymnasium.spaces.Box(0, 1, shape=(1, 2, 3), dtype=np.float32)

        def render(**kwargs):
            assert 'mode' not in kwargs.keys()

    gym_env = createGymWrapper(NoModeGymnasiumInnerClass)(config={})
    gym_env.render(mode=1)



if __name__ == '__main__':
    test_conversion_functions()
    test_gym_wrapper_simple()

    test_gym_wrapper_removes_mode()

    # end to end test with metadrive
    test_gym_wrapper_single_agent_metadrive()
    test_gym_wrapper_multi_agent_metadrive()