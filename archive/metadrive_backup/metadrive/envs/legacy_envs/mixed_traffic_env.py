from metadrive.envs.metadrive_env import MetaDriveEnv


class MixedTrafficEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls) -> "Config":
        config = super(MixedTrafficEnv, cls).default_config()
        config["rl_agent_ratio"] = 0.0
        return config

    def setup_engine(self):
        super(MetaDriveEnv, self).setup_engine()
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("q", self.switch_to_third_person_view)
        from metadrive.manager.traffic_manager import MixedPGTrafficManager
        from metadrive.manager.pg_map_manager import PGMapManager
        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("traffic_manager", MixedPGTrafficManager())


if __name__ == '__main__':
    env = MixedTrafficEnv(
        {
            "rl_agent_ratio": 0.5,
            "manual_control": True,
            "use_render": True,
            "disable_model_compression": True,
            # "map": "SS",
            "num_scenarios": 100,
        }
    )
    try:
        obs, _ = env.reset()
        obs_space = env.observation_space
        assert obs_space.contains(obs)
        for _ in range(100000):
            assert env.observation_space.contains(obs)
            o, r, tm, tc, i = env.step(env.action_space.sample())
            assert obs_space.contains(o)
            if tm or tc:
                o, _ = env.reset()
                assert obs_space.contains(o)
    finally:
        env.close()
