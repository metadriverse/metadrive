import numpy as np
from gym.spaces import Box
from pgdrive.constants import TerminationState
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import Config


class ActionRepeat(PGDriveEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = PGDriveEnv.default_config()

        # Set the internal environment run in 0.02s interval.
        config["decision_repeat"] = 1

        # Speed reward_function is given for current state, so its magnitude need to be reduced
        original_action_repeat = PGDriveEnv.default_config()["decision_repeat"]
        config["speed_reward"] = config["speed_reward"] / original_action_repeat

        # Set the interval from 0.02s to 1s
        config.update({
            "fixed_action_repeat": 0,
            "max_action_repeat": 50,
            "min_action_repeat": 1,
            "horizon": 5000,
        }, )
        # default gamma for ORIGINAL primitive step!
        # Note that we will change this term since ORIGINAL primitive steps is not the internal step!
        # It still contains ORIGINAL_ACTION_STEP internal steps!
        # So we will modify this gamma to make sure it behaves like the one applied to ORIGINAL primitive steps.
        # config.add("gamma", 0.99)
        return config

    def __init__(self, config: dict = None):
        super(ActionRepeat, self).__init__(config)

        if self.config["fixed_action_repeat"] > 0:
            self.fixed_action_repeat = self.config["fixed_action_repeat"]
        else:
            self.fixed_action_repeat = None
        # self.gamma = self.config["gamma"]
        # Modify gamma to make sure it behaves like the one applied to ORIGINAL steps.
        # self.gamma =

        self.low = self.action_space.low[0]
        self.high = self.action_space.high[0]
        self.action_repeat_low = self.config["min_action_repeat"]
        self.action_repeat_high = self.config["max_action_repeat"]

        self.interval = 2e-2  # This is determined by the default config of engine.
        self.primitive_steps_count = 0

        assert self.action_repeat_low > 0

    def step(self, action, render=False, **render_kwargs):
        if self.fixed_action_repeat is None:
            action_repeat = action[-1]
            action_repeat = round(
                (action_repeat - self.low) / (self.high - self.low) *
                (self.action_repeat_high - self.action_repeat_low) + self.action_repeat_low
            )
            assert action_repeat > 0
        else:
            action_repeat = self.fixed_action_repeat

        action_repeat = min(max(self.action_repeat_low, int(action_repeat)), self.action_repeat_high)

        o_list = []
        r_list = []
        d_list = []
        i_list = []
        discounted_r_list = []

        render_list = []
        real_ret = 0.0
        for repeat in range(action_repeat):
            o, r, d, i = super(ActionRepeat, self).step(action[:2])
            if render:
                render_list.append(self.render(**render_kwargs))
            r_list.append(r)
            o_list.append(o)
            d_list.append(d)
            i_list.append(i)
            discounted_r_list.append(0.0)
            real_ret += r
            self.primitive_steps_count += 1
            d = d or self.primitive_steps_count > self.config["horizon"]
            if d:
                break

        discounted = 0.0
        for idx in reversed(range(len(r_list))):
            reward = r_list[idx]
            # discounted = self.config["gamma"] * discounted + reward_function
            discounted = discounted + reward
            discounted_r_list[idx] = discounted

        i["simulation_time"] = (repeat + 1) * self.interval
        i["real_return"] = real_ret
        i["action_repeat"] = action_repeat
        i["primitive_steps_count"] = self.primitive_steps_count
        i[TerminationState.MAX_STEP] = self.primitive_steps_count > self.config["horizon"]
        i["render"] = render_list
        i["trajectory"] = [
            dict(
                reward_function=r_list[idx],
                discounted_reward=discounted_r_list[idx],
                obs=o_list[idx],
                action=action,
                count=self.primitive_steps_count - len(r_list) + idx
            ) for idx in range(len(r_list))
        ]

        if d:
            self.primitive_steps_count = 0

        return o, discounted, d, i

    def _get_reset_return(self):
        o, *_ = self.step(np.array([0.0, 0.0, 0.0]))
        return o

    @property
    def action_space(self):
        super_action_space = super(ActionRepeat, self).action_space
        return Box(
            shape=(super_action_space.shape[0] + 1, ),
            high=super_action_space.high[0],
            low=super_action_space.low[0],
            dtype=super_action_space.dtype
        )


if __name__ == '__main__':
    env = ActionRepeat(dict(fixed_action_repeat=25))
    env.reset()
    for i in range(5000):
        _, _, d, info = env.step([0, 0])
        print("max_step: ", info[TerminationState.MAX_STEP], i, info["primitive_steps_count"])
        if d:
            break
    env.close()
