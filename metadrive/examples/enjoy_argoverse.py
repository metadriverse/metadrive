from metadrive.component.map.argoverse_map import ArgoverseMap
from metadrive.envs.metadrive_env import MetaDriveEnv
import pathlib
from metadrive.utils.utils import is_win
import pickle


class ArgoverseEnv(MetaDriveEnv):
    def _post_process_config(self, config):
        config = super(ArgoverseEnv, self)._post_process_config(config)
        config["vehicle_config"]["spawn_lane_index"] = ("11713", "4250", 0)
        config["vehicle_config"]["destination_node"] = "968"

        log_id = "c6911883-1843-3727-8eaa-41dc8cda8993"
        root_path = pathlib.PurePosixPath(__file__).parent.parent if not is_win() else pathlib.Path(__file__).resolve(
        ).parent.parent
        data_path = root_path.joinpath("assets").joinpath("real_data").joinpath("{}.pkl".format(log_id))
        with open(data_path, 'rb') as f:
            locate_info, _ = pickle.load(f)
        config.update({
            "real_data_config": {
                "locate_info": locate_info
            }
        })

        return config

    def setup_engine(self):
        super(ArgoverseEnv, self).setup_engine()
        from metadrive.manager.real_data_manager import RealDataManager
        self.engine.register_manager("real_data_manager", RealDataManager())

    def _update_map(self, episode_data: dict = None):
        xcenter, ycenter = 2599.5505965123866, 1200.0214763629717
        if self.current_map is None:
            self.config["map_config"].update(
                {
                    "city": "PIT",
                    "center": ArgoverseMap.metadrive_position([xcenter, ycenter]),
                    # "draw_map_resolution": 1024,
                    # "center": [xcenter, ycenter],
                    "radius": 100
                }
            )
            map = ArgoverseMap(self.config["map_config"])
            self.engine.map_manager.load_map(map)


if __name__ == "__main__":
    env = ArgoverseEnv(
        {
            "traffic_density": 0.,
            "onscreen_message": True,
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
            "fast": False,
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
        # env.render(text=info)
    env.close()
