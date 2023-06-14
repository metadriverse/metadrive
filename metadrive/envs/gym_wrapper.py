try:
    from typing import Any, Dict
    import gymnasium
    import gym
    import gym.spaces

    class GymEnvWrapper(gym.Env):
        def __init__(self, config: Dict[str, Any]):
            """
            Note that config must contain two items:
            "inner_class": the class of a Metadrive environment (not instantiated)
            "inner_config": The config that will be passed to the Metadrive environment
            """
            inner_class = config["inner_class"]
            inner_config = config["inner_config"]
            assert isinstance(inner_class, type)
            assert isinstance(inner_config, dict)        
            self._inner = inner_class(config=inner_config)

        def step(self, actions):
            o, r, tm, tc, i = self._inner.step(actions)
            if isinstance(tm, dict) and isinstance(tc, dict):
                d = {tm[j] or tc[j] for j in set(list(tm.keys()) + list(tc.keys()))}
            else:
                d = tm or tc
            return o, r, d, i

        def reset(self, *, seed=None, options=None):
            # pass non-none parameters to the reset (which may not support options or seed)
            params = {"seed": seed, "options": options}
            not_none_params = {k:v for k, v in params.items() if v is not None}
            obs, _ = self._inner.reset(**not_none_params)
            return obs

        def render(self, *args, **kwargs):
            # remove mode from kwargs
            kwargs.pop("mode", None)
            return self._inner.render(*args, **kwargs)

        def close(self):
            self._inner.close()

        def seed(self, seed=None):
            """
            We cannot seed a Gymnasium environment while running, so do nothing
            """
            pass

        @property
        def observation_space(self):
            obs_space = self._inner.observation_space
            assert isinstance(obs_space, gymnasium.spaces.Box)
            return gym.spaces.Box(low=obs_space.low, high=obs_space.high, shape=obs_space.shape)

        @property
        def action_space(self):
            action_space = self._inner.action_space
            assert isinstance(action_space, gymnasium.spaces.Box)
            return gym.spaces.Box(low=action_space.low, high=action_space.high, shape=action_space.shape)

        def __getattr__(self, __name: str) -> Any:
            return self._inner[__name]


    if __name__ == '__main__':
        from metadrive.envs.scenario_env import ScenarioEnv

        env = GymEnvWrapper(config={"inner_class": ScenarioEnv, "inner_config":{"manual_control": True}})
        o, i = env.reset()
        assert isinstance(env.observation_space, gymnasium.Space)
        assert isinstance(env.action_space, gymnasium.Space)
        for s in range(600):
            o, r, d, i = env.step([0, -1])
            env.vehicle.set_velocity([0, 0])
            if d:
                assert s == env.config["horizon"] and i["max_step"] and d
                break
except ImportError:
    pass