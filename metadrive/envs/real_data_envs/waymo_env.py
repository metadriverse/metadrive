from metadrive.envs.scenario_env import ScenarioEnv

WAYMO_ENV_CONFIG = dict(
    # ===== Map Config =====
    waymo_data_directory=None,  # for compatibility
    allow_coordinate_transform=True,  # for compatibility
)


class WaymoEnv(ScenarioEnv):

    @classmethod
    def default_config(cls):
        config = super(WaymoEnv, cls).default_config()
        config.update(WAYMO_ENV_CONFIG)
        return config

    def __init__(self, config=None):
        super(WaymoEnv, self).__init__(config)

    def _merge_extra_config(self, config):
        config = self.default_config().update(config, allow_add_new_key=False)
        if config["waymo_data_directory"] is not None:
            config["data_directory"] = config["waymo_data_directory"]
        return config
