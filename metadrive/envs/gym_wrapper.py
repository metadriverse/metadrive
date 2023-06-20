try:
    import inspect
    from typing import Any, Dict
    import gymnasium
    import gym
    import gym.spaces

    def gymnasiumToGym(space:gymnasium.spaces.Space) -> gym.spaces.Space:
        if isinstance(space, gymnasium.spaces.Box):
            return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape)
        elif isinstance(space, gymnasium.spaces.Discrete):
            return gym.spaces.Discrete(n=int(space.n), start=int(space.start))
        elif isinstance(space, gymnasium.spaces.Tuple):
            return gym.spaces.Tuple([gymnasiumToGym(subspace) for subspace in space.spaces])
        elif isinstance(space, gymnasium.spaces.Dict):
            return gym.spaces.Dict({key: gymnasiumToGym(subspace) for key, subspace in space.spaces.items()})
        else:
            raise ValueError("unsupported space")

    def defined_in_this_file(a):
        return inspect.getfile(a) == inspect.getfile(createGymWrapper)

    def createGymWrapper(inner_class: type):
        """
        "inner_class": A gymnasium based Metadrive environment class
        """
        class GymEnvWrapper(gym.Env):

            @classmethod
            def default_config(cls):
                """
                This is the default, if you override it, then we will override it within the inner_class to maintain consistency
                """
                return inner_class.default_config()

            def __init__(self, config: Dict[str, Any]):
                # We can only tell if someone has overriden the default config method at init time.

                # we know that the function is overriden if the function was not defined in this file
                # because this is a dynamic class, equality checks will always return false
                # TODO: this is a hack, but i'm not sure how to make it more robust
                if not defined_in_this_file(type(self).default_config):
                    current_wrapper_default_config = type(self).default_config
                    # at this point, if there was an override, we need to provide the overriden method to the inner class.
                    class OverridenDefaultConfigWrapper(inner_class):
                        @classmethod
                        def default_config(cls):
                            print(current_wrapper_default_config)
                            print(GymEnvWrapper.default_config)
                            return current_wrapper_default_config()
                    # init now has access to the new default_config
                    self._inner = OverridenDefaultConfigWrapper(config=config)
                else:
                    # if no override, directly initialize
                    self._inner = inner_class(config=config)

            def step(self, actions):
                o, r, tm, tc, i = self._inner.step(actions)
                if isinstance(tm, dict) and isinstance(tc, dict):
                    d = {j: (j in tm and tm[j]) or (j in tc and tc[j]) for j in set(list(tm.keys()) + list(tc.keys()))}
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
                return gymnasiumToGym(self._inner.observation_space)

            @property
            def action_space(self):
                return gymnasiumToGym(self._inner.action_space)

            def __getattr__(self, __name: str) -> Any:
                return getattr(self._inner, __name)
        return GymEnvWrapper


    if __name__ == '__main__':
        from metadrive.envs.scenario_env import ScenarioEnv

        env = GymEnvWrapper(config={"inner_class": ScenarioEnv, "inner_config": {"manual_control": True}})
        o, i = env.reset()
        assert isinstance(env.observation_space, gymnasium.spaces.Space)
        assert isinstance(env.action_space, gymnasium.spaces.Space)
        for s in range(600):
            o, r, d, i = env.step([0, -1])
            env.vehicle.set_velocity([0, 0])
            if d:
                assert s == env.config["horizon"] and i["max_step"] and d
                break
except ImportError:
    pass
