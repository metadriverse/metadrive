import numpy as np
from pgdrive.constants import CamMask
from pgdrive.utils.engine_utils import get_pgdrive_engine
from pgdrive.constants import BodyName
from pgdrive.scene_creator.blocks.block import BlockSocket
from pgdrive.scene_creator.blocks.bottleneck import Block
from pgdrive.scene_creator.blocks.create_block_utils import CreateAdverseRoad, CreateRoadFrom, ExtendStraightLane
from pgdrive.scene_creator.buildings.base_building import BaseBuilding
from pgdrive.scene_creator.lane.abs_lane import LineType, LineColor
from pgdrive.scene_creator.road.road import Road
from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.utils.pg_space import PGSpace, Parameter, BlockParameterSpace

TollGateBuilding = BaseBuilding


class TollGate(Block):
    """
    Toll, like Straight, but has speed limit
    """
    SOCKET_NUM = 1
    PARAMETER_SPACE = PGSpace(BlockParameterSpace.BOTTLENECK_PARAMETER)
    ID = "$"

    SPEED_LIMIT = 3  # m/s ~= 5 miles per hour https://bestpass.com/feed/61-speeding-through-tolls
    BUILDING_LENGTH = 10
    BUILDING_HEIGHT = 5

    def _try_plug_into_previous_block(self) -> bool:
        self.set_part_idx(0)  # only one part in simple block like straight, and curve
        para = self.get_config()
        length = para[Parameter.length]
        self.BUILDING_LENGTH = length
        basic_lane = self.positive_basic_lane
        new_lane = ExtendStraightLane(basic_lane, length, [LineType.CONTINUOUS, LineType.SIDE])
        start = self.pre_block_socket.positive_road.end_node
        end = self.add_road_node()
        socket = Road(start, end)
        _socket = -socket

        # create positive road
        no_cross = CreateRoadFrom(
            new_lane,
            self.positive_lane_num,
            socket,
            self.block_network,
            self._global_network,
            center_line_color=LineColor.YELLOW,
            center_line_type=LineType.CONTINUOUS,
            inner_lane_line_type=LineType.CONTINUOUS,
            side_lane_line_type=LineType.SIDE
        )

        # create negative road
        no_cross = CreateAdverseRoad(
            socket,
            self.block_network,
            self._global_network,
            center_line_color=LineColor.YELLOW,
            center_line_type=LineType.CONTINUOUS,
            inner_lane_line_type=LineType.CONTINUOUS,
            side_lane_line_type=LineType.SIDE
        ) and no_cross

        self.add_sockets(BlockSocket(socket, _socket))
        self._add_building_and_speed_limit(socket)
        self._add_building_and_speed_limit(_socket)
        return no_cross

    def _add_building_and_speed_limit(self, road):
        # add house
        lanes = road.get_lanes(self.block_network)
        for idx, lane in enumerate(lanes):
            lane.set_speed_limit(self.SPEED_LIMIT)
            if idx % 2 == 1:
                # add toll
                position = lane.position(lane.length / 2, 0)
                node_path = self._generate_invisible_static_wall(
                    position,
                    np.rad2deg(lane.heading_at(0)),
                    self.BUILDING_LENGTH,
                    self.lane_width,
                    self.BUILDING_HEIGHT / 2,
                    name=BodyName.TollGate
                )
                if self.render:
                    building_model = self.loader.loadModel(AssetLoader.file_path("models", "tollgate", "booth.gltf"))
                    gate_model = self.loader.loadModel(AssetLoader.file_path("models", "tollgate", "gate.gltf"))
                    building_model.setH(90)
                    building_model.reparentTo(node_path)
                    gate_model.reparentTo(node_path)

                building = TollGateBuilding(
                    lane, (road.start_node, road.end_node, idx), position, lane.heading_at(0), node_path, random_seed=0
                )
                self._block_objects.append(building)

    def construct_block_buildings(self, object_manager):
        engine = get_pgdrive_engine()
        for building in self._block_objects:
            object_manager.add_block_buildings(building, engine.pbr_worldNP)
            # for performance reason
            building.node_path.hide(CamMask.Shadow)
