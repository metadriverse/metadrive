import numpy as np
from gym.spaces import Box
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config import PgConfig


class ActionRepeat(PGDriveEnv):
    @staticmethod
    def default_config() -> PgConfig:
        config = PGDriveEnv.default_config()

        # Set the internal environment run in 0.02s interval.
        config["decision_repeat"] = 1

        # Speed reward is given for current state, so its magnitude need to be reduced
        config["speed_reward"] = config["speed_reward"] / 5

        # Set the interval from 0.02s to 1s
        config.add("fixed_action_repeat", 0)  # 0 stands for using varying action repeat.
        config.add("max_action_repeat", 50)
        config.add("min_action_repeat", 1)
        config.add("gamma", 0.99)  # common config default gamma
        return config

    def __init__(self, config: dict = None):
        super(ActionRepeat, self).__init__(config)

        if self.config["fixed_action_repeat"] > 0:
            self.fixed_action_repeat = self.config["fixed_action_repeat"]
        else:
            self.fixed_action_repeat = None
            self.action_space = Box(
                shape=(self.action_space.shape[0] + 1, ),
                high=self.action_space.high[0],
                low=self.action_space.low[0],
                dtype=self.action_space.dtype
            )

        self.low = self.action_space.low[0]
        self.high = self.action_space.high[0]
        self.action_repeat_low = self.config["min_action_repeat"]
        self.action_repeat_high = self.config["max_action_repeat"]

        self.interval = 2e-2  # This is determined by the default config of pg_world.

        assert self.action_repeat_low > 0

    def step(self, action, render=False, **render_kwargs):
        if self.fixed_action_repeat is None:
            action_repeat = action[-1]
            action_repeat = round(
                (action_repeat - self.low) / (self.high - self.low) *
                (self.action_repeat_high - self.action_repeat_low) + self.action_repeat_low
            )
            # print("[DEBUG] raw action: {}, input action: {}, action repeat: {}".format(
            # action, action[-1], action_repeat))
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
            if d:
                break

        discounted = 0.0
        for idx in reversed(range(len(r_list))):
            reward = r_list[idx]
            discounted = self.config["gamma"] * discounted + reward
            discounted_r_list[idx] = discounted

        i["simulation_time"] = (repeat + 1) * self.interval
        i["real_return"] = real_ret
        i["action_repeat"] = action_repeat
        i["render"] = render_list
        i["trajectory"] = [
            dict(
                reward=r_list[idx],
                discounted_reward=discounted_r_list[idx],
                obs=o_list[idx],
                action=action,
            ) for idx in range(len(r_list))
        ]
        return o, discounted, d, i

    def _get_reset_return(self):
        o, *_ = self.step(np.array([0.0, 0.0, 0.0]))
        return o


if __name__ == '__main__':
    env = ActionRepeat(dict(fixed_action_repeat=5))
    env.reset()
    env.step(env.action_space.sample())
    env.close()
