from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.lange_change_policy import LaneChangePolicy


def test_check_discrete_space(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "start_seed": 22,
            "debug": True,
            "use_render": render,
            "decision_repeat": 5,
            "map": "CXO",
            "agent_policy": LaneChangePolicy,
            "discrete_action": True,
            "use_multi_discrete": False,
            "action_check": True,
        }
    )
    assert not env.config["use_multi_discrete"]
    try:
        o, _ = env.reset()
        for s in range(1, 30):
            o, r, tm, tc, info = env.step(env.action_space.sample())
            assert env.action_space.n == env.config["discrete_throttle_dim"] * 3
    finally:
        env.close()


def test_check_multi_discrete_space(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "start_seed": 22,
            "debug": True,
            "use_render": render,
            "decision_repeat": 5,
            "map": "CXO",
            "agent_policy": LaneChangePolicy,
            "discrete_action": True,
            "use_multi_discrete": True,
            "action_check": True,
        }
    )
    assert env.config["use_multi_discrete"]
    try:
        o, _ = env.reset()
        for s in range(1, 30):
            o, r, tm, tc, info = env.step(env.action_space.sample())
            assert env.action_space.nvec[0] == 3 and env.action_space.nvec[1] == env.config["discrete_throttle_dim"]
    finally:
        env.close()


def test_lane_change(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "start_seed": 22,
            "debug": False,
            "use_render": render,
            "decision_repeat": 5,
            "map": "CXO",
            "agent_policy": LaneChangePolicy,
            "discrete_action": True,
            "use_multi_discrete": True,
            "action_check": True,
            # "debug_static_world": True,
            # "debug_physics_world": True,
        }
    )
    try:
        o, _ = env.reset()
        for s in range(1, 60):
            o, r, tm, tc, info = env.step([2, 3])
        assert env.agent.lane.index[-1] == 0
        for s in range(1, 40):
            o, r, tm, tc, info = env.step([0, 3])
        assert env.agent.lane.index[-1] == 2
        for s in range(1, 70):
            o, r, tm, tc, info = env.step([1, 3])
        assert env.agent.lane.index[-1] == 2
    finally:
        env.close()


if __name__ == "__main__":
    test_lane_change(True)
    # test_check_multi_discrete_space()
