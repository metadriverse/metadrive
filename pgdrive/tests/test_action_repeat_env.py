import pytest

from pgdrive import ActionRepeat


def _test_action_repeat(config):
    env = ActionRepeat(config)
    env.reset()
    for i in range(100):
        _, _, d, info = env.step(env.action_space.sample())

        for k in ["simulation_time", "real_return", "action_repeat", "primitive_steps_count", "max_step", "render",
                  "trajectory"]:
            assert k in info

        assert len(info["trajectory"]) == info["action_repeat"]
        for t in info["trajectory"]:
            for k in ["reward", "discounted_reward", "obs", "action", "count"]:
                assert k in t

        if d:
            env.reset()
    env.close()


def test_action_repeat_env():
    _test_action_repeat(dict(max_action_repeat=5))
    _test_action_repeat(dict(max_action_repeat=1))
    _test_action_repeat(dict(fixed_action_repeat=5))
    _test_action_repeat(dict(fixed_action_repeat=1))


if __name__ == '__main__':
    pytest.main(["-sv", "test_action_repeat_env.py"])
