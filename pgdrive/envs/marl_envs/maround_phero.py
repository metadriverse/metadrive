import time

import numpy as np
from gym.spaces import Box

from pgdrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv as MARound, \
    LidarStateObservationMARound
from pgdrive.envs.marl_envs.pheromone_map import PheromoneMap


class PheroObs(LidarStateObservationMARound):
    @property
    def observation_space(self):
        space = super(PheroObs, self).observation_space
        assert isinstance(space, Box)
        assert len(space.shape) == 1

        # Add extra 9 pheromones information!
        length = space.shape[0] + self.config["num_neighbours"] * self.config["num_channels"]

        space = Box(
            low=np.array([space.low[0]] * length),
            high=np.array([space.high[0]] * length),
            shape=(length, ),
            dtype=space.dtype
        )
        return space


class MARoundPhero(MARound):
    @classmethod
    def default_config(cls):
        config = super(MARoundPhero, cls).default_config()
        config.update(
            dict(
                attenuation_rate=0.99,
                diffusion_rate=0.95,
                num_channels=1,
                num_neighbours=1,  # or 9.
                granularity=0.5
            )
        )
        return config

    def __init__(self, config=None):
        super(MARoundPhero, self).__init__(config)
        assert self.config["num_neighbours"] in [1, 9]
        assert 0 <= self.config["attenuation_rate"] <= 1.0
        assert 0 <= self.config["diffusion_rate"] <= 1.0
        self.phero_map = None

    def _post_process_config(self, config):
        config = super(MARoundPhero, self)._post_process_config(config)
        config["vehicle_config"]["num_neighbours"] = config["num_neighbours"]
        config["vehicle_config"]["num_channels"] = config["num_channels"]
        config["vehicle_config"]["extra_action_dim"] = config["num_channels"]
        return config

    def get_single_observation(self, vehicle_config):
        return PheroObs(vehicle_config)

    def _after_lazy_init(self):
        super(MARoundPhero, self)._after_lazy_init()
        self._update_map()
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        for b in self.current_map.blocks:
            min_x = min(b.bounding_box[0], min_x)
            max_x = max(b.bounding_box[1], max_x)
            min_y = min(b.bounding_box[2], min_y)
            max_y = max(b.bounding_box[3], max_y)
        self.phero_map = PheromoneMap(
            min_x=min_x,
            min_y=min_y,
            total_width=max_x - min_x + 1,
            total_length=max_y - min_y + 1,
            num_channels=self.config["num_channels"],
            diffusion_rate=self.config["diffusion_rate"],
            attenuation_rate=self.config["attenuation_rate"],
            granularity=self.config["granularity"]
        )

    def _get_reset_return(self):
        self.phero_map.clear()
        obses = super(MARoundPhero, self)._get_reset_return()
        ret = {v_id: self._add_phero(v_id, obs) for v_id, obs in obses.items()}
        return ret

    def _step_simulator(self, actions, action_infos):
        ret = super(MARoundPhero, self)._step_simulator(actions, action_infos)
        for v_id, act in actions.items():
            self.phero_map.add(self.vehicles[v_id].position, act[2:])
        self.phero_map.step()
        return ret

    def step(self, actions):
        o, r, d, i = super(MARoundPhero, self).step(actions)
        ret = {v_id: self._add_phero(v_id, obs) for v_id, obs in o.items()}
        return ret, r, d, i

    def _add_phero(self, v_id, o):
        if v_id not in self.vehicles:
            ret = np.zeros((self.config["num_neighbours"] * self.config["num_channels"], ))
        else:
            ret = self.phero_map.get_nearest_pheromone(self.vehicles[v_id].position, self.config["num_neighbours"])
        return np.concatenate([o, ret])

    def _render_topdown(self, *args, **kwargs):
        if self._top_down_renderer is None:
            from pgdrive.obs.top_down_renderer import PheromoneRenderer
            self._top_down_renderer = PheromoneRenderer(self.current_map, *args, **kwargs)
        self._top_down_renderer.render(list(self.vehicles.values()), pheromone_map=self.phero_map)


def _profile():
    env = MARoundPhero({"num_agents": 40})
    obs = env.reset()
    start = time.time()
    for s in range(10000):
        o, r, d, i = env.step(env.action_space.sample())
        if all(d.values()):
            env.reset()
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
                    s + 1,
                    time.time() - start, (s + 1) / (time.time() - start)
                )
            )
    print(f"(PGDriveEnvV2) Total Time Elapse: {time.time() - start}")


def _test():
    env = MARoundPhero({"num_channels": 3})
    o = env.reset()
    assert env.observation_space.contains(o)
    assert all([0 <= oo[-1] <= 1.0 for oo in o.values()])
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(o)
        assert all([0 <= oo[-1] <= 1.0 for oo in o.values()])
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis():
    env = MARoundPhero(
        {
            "num_channels": 1,
            "num_agents": 40,
            "diffusion_rate": 0.95,
            "attenuation_rate": 0.99,
            "granularity": 0.5
        }
    )
    o = env.reset()
    start = time.time()
    for s in range(1, 100000):
        # o, r, d, info = env.step(env.action_space.sample())
        o, r, d, info = env.step(
            {k: [np.random.uniform(-0.01, 0.01), 1, np.random.uniform(0, 1)]
             for k in env.vehicles.keys()}
        )
        # o, r, d, info = env.step({k: [np.random.uniform(-0.01, 0.01), 1, np.random.uyni] for k in env.vehicles.keys()})
        env.render(mode="top_down")
        # env.render(mode="top_down")
        # env.render(mode="top_down", film_size=(1000, 1000))
        if d["__all__"]:
            env.reset()
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
                    s + 1,
                    time.time() - start, (s + 1) / (time.time() - start)
                )
            )
    env.close()


if __name__ == '__main__':
    # _test()
    # _profile()
    _vis()
