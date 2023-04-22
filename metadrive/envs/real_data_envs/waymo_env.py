from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.engine.asset_loader import AssetLoader
import time

WAYMO_ENV_CONFIG = dict(
    # ===== Map Config =====
    data_directory=AssetLoader.file_path("waymo", return_raw_style=False),
    waymo_data_directory=None,  # for compatibility
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


if __name__ == "__main__":
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy

    env = WaymoEnv(
        {
            "use_render": True,
            # "agent_policy": ReplayEgoCarPolicy,
            "manual_control": True,
            "no_traffic": False,
            # "debug":True,
            # "debug_static_world": True,
            # "no_traffic":True,
            "start_scenario_index": 0,
            "show_coordinates": True,
            # "start_scenario_index": 1000,
            # "show_coordinates": True,
            "num_scenarios": 3,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "show_policy_mark": True,
            "no_static_vehicles": True,
            "reactive_traffic": True,
            "horizon": 1000,
            "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
        }
    )
    success = []
    while True:
        for i in range(3):
            env.reset(force_seed=i)
            while True:
                step_start = time.time()
                o, r, d, info = env.step([0, 0])
                assert env.observation_space.contains(o)
                # c_lane = env.vehicle.lane
                # long, lat, = c_lane.local_coordinates(env.vehicle.position)
                print("Step: {}, Time: {}".format(env.episode_step, time.time() - step_start))
                # if env.config["use_render"]:
                env.render(
                    text={
                        "seed": env.engine.global_seed + env.config["start_scenario_index"],
                    }
                    # mode="topdown"
                )

                if d:
                    if info["arrive_dest"]:
                        print("seed:{}, success".format(env.engine.global_random_seed))
                    break
