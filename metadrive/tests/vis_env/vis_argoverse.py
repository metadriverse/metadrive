from metadrive.component.map.argoverse_map import ArgoverseMap
from metadrive.envs.metadrive_env import MetaDriveEnv


class ArgoverseEnv(MetaDriveEnv):
    def _post_process_config(self, config):
        config = super(ArgoverseEnv, self)._post_process_config(config)
        config["vehicle_config"]["spawn_lane_index"] = ("11713", "4250", 0)
        config["vehicle_config"]["destination"] = "968"
        return config

    def _update_map(self, episode_data: dict = None):
        xcenter, ycenter = 2599.5505965123866, 1200.0214763629717
        if self.current_map is None:
            self.config["map_config"].update(
                {
                    "city": "PIT",
                    # "draw_map_resolution": 1024,
                    "center": ArgoverseMap.metadrive_position([xcenter, ycenter]),
                    "radius": 100
                }
            )
            map = ArgoverseMap(self.config["map_config"])
            self.engine.map_manager.load_map(map)


if __name__ == "__main__":
    env = ArgoverseEnv(
        {
            "traffic_density": 0.,
            # "debug_physics_world": True,
            "pstats": True,
            "global_light": True,
            # "debug_static_world":True,
            "cull_scene": False,
            # "controller":"joystick",
            "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "rgb_clip": True,
            # "debug": False,
            "vehicle_config": {
                "enable_reverse": True,
                "side_detector": dict(num_lasers=2, distance=50),
                "lane_line_detector": dict(num_lasers=2, distance=50),
            }
        }
    )

    o = env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([1.0, 0.])
        info = {}
        info["lane_index"] = env.vehicle.lane_index
        info["heading_diff"] = env.vehicle.heading_diff(env.vehicle.lane)
        # info["left_lane_index"] =
        # info["right_lane_index"]
        env.render(text=info)
    env.close()
