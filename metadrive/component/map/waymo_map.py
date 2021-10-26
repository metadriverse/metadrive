from metadrive.component.blocks.waymo_block import WaymoBlock
from metadrive.component.map.base_map import BaseMap
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.math_utils import get_boxes_bounding_box
from metadrive.utils.waymo_map_utils import read_waymo_data


class WaymoMap(BaseMap):
    def __init__(self, waymo_data):
        self.map_id = waymo_data["id"]
        self.waymo_data = waymo_data
        super(WaymoMap, self).__init__(dict(id=waymo_data["id"]))

    def _generate(self):
        block = WaymoBlock(0, self.road_network, 0, self.waymo_data["map"])
        block.construct_block(self.engine.worldNP, self.engine.physics_world)
        self.blocks.append(block)

    @staticmethod
    def waymo_position(pos):
        return pos[0], -pos[1]

    @staticmethod
    def metadrive_position(pos):
        return pos[0], -pos[1]

    def get_center_point(self):
        raise DeprecationWarning("This func is broken")
        boxes = []
        for _from, to_dict in self.road_network.graph.items():
            for _to, lanes in to_dict.items():
                for lane in lanes:
                    boxes.append(lane.get_bounding_box())
        x_min, x_max, y_min, y_max = get_boxes_bounding_box(boxes)
        return x_max - x_min, y_max - y_min


if __name__ == "__main__":
    from metadrive.engine.engine_utils import initialize_engine
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.utils.waymo_map_utils import AgentType
    from metadrive.utils.waymo_map_utils import RoadEdgeType
    from metadrive.utils.waymo_map_utils import RoadLineType

    # touch these items so that pickle can work
    _ = AgentType
    _ = RoadLineType
    _ = RoadEdgeType

    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)

    default_config = MetaDriveEnv.default_config()
    default_config["use_render"] = True
    default_config["debug"] = True
    default_config["debug_static_world"] = True
    engine = initialize_engine(default_config)
    map = WaymoMap(data)
    map.attach_to_world()
    engine.enableMouse()

    # argoverse data set is as the same coordinates as panda3d
    pos = WaymoMap.metadrive_position(list(data["map"].values())[0]["polyline"][0])
    engine.main_camera.set_bird_view_pos(pos)
    while True:
        map.engine.step()
