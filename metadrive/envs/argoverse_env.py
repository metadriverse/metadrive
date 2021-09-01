import pathlib
import pickle

from metadrive import MetaDriveEnv
from metadrive.component.map.argoverse_map import ArgoverseMap
from metadrive.utils import is_win
from metadrive.manager.map_manager import MapManager


class ArgoverseMapManager(MapManager):
    def before_reset(self):
        # do not unload map
        pass

    def reset(self):
        xcenter, ycenter = 2599.5505965123866, 1200.0214763629717
        if self.current_map is None:
            self.engine.global_config["map_config"].update(
                {
                    "city": "PIT",
                    "center": ArgoverseMap.metadrive_position([xcenter, ycenter]),
                    "radius": 150
                }
            )
            map = ArgoverseMap(self.engine.global_config["map_config"])
            self.engine.map_manager.load_map(map)


class ArgoverseEnv(MetaDriveEnv):
    def _post_process_config(self, config):
        config = super(ArgoverseEnv, self)._post_process_config(config)
        config["vehicle_config"]["spawn_lane_index"] = ('7903', '9713', 0)
        config["vehicle_config"]["destination_node"] = "968"

        log_id = "c6911883-1843-3727-8eaa-41dc8cda8993"
        root_path = pathlib.PurePosixPath(__file__).parent.parent if not is_win() else pathlib.Path(__file__).resolve(
        ).parent.parent
        data_path = root_path.joinpath("assets").joinpath("real_data").joinpath("{}.pkl".format(log_id))
        with open(data_path, 'rb') as f:
            locate_info, _ = pickle.load(f)
        config.update({"real_data_config": {"locate_info": locate_info}})
        config["traffic_density"] = 0.0
        return config

    def setup_engine(self):
        super(ArgoverseEnv, self).setup_engine()
        from metadrive.manager.real_data_manager import RealDataManager
        self.engine.register_manager("real_data_manager", RealDataManager())
        self.engine.update_manager("map_manager", ArgoverseMapManager())
