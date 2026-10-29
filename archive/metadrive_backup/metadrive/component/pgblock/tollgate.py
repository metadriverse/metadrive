from metadrive.component.buildings.tollgate_building import TollGateBuilding
from metadrive.component.pgblock.bottleneck import PGBlock
from metadrive.component.pgblock.create_pg_block_utils import CreateAdverseRoad, CreateRoadFrom, ExtendStraightLane
from metadrive.component.pgblock.pg_block import PGBlockSocket
from metadrive.component.road_network import Road
from metadrive.constants import PGLineType, PGLineColor
from metadrive.engine.engine_utils import get_engine
from metadrive.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace


class TollGate(PGBlock):
    """
    Toll, like Straight, but has speed limit
    """
    SOCKET_NUM = 1
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.BOTTLENECK_PARAMETER)
    ID = "$"

    SPEED_LIMIT = 3  # m/s ~= 5 miles per hour https://bestpass.com/feed/61-speeding-through-tolls

    def _try_plug_into_previous_block(self) -> bool:
        self.set_part_idx(0)  # only one part in simple block like straight, and curve
        para = self.get_config()
        length = para[Parameter.length]
        self.BUILDING_LENGTH = length
        basic_lane = self.positive_basic_lane
        new_lane = ExtendStraightLane(basic_lane, length, [PGLineType.CONTINUOUS, PGLineType.SIDE])
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
            center_line_color=PGLineColor.YELLOW,
            center_line_type=PGLineType.CONTINUOUS,
            inner_lane_line_type=PGLineType.CONTINUOUS,
            side_lane_line_type=PGLineType.SIDE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        # create negative road
        no_cross = CreateAdverseRoad(
            socket,
            self.block_network,
            self._global_network,
            center_line_color=PGLineColor.YELLOW,
            center_line_type=PGLineType.CONTINUOUS,
            inner_lane_line_type=PGLineType.CONTINUOUS,
            side_lane_line_type=PGLineType.SIDE,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross

        self.add_sockets(PGBlockSocket(socket, _socket))
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
                building = get_engine().spawn_object(
                    TollGateBuilding, lane=lane, position=position, heading_theta=lane.heading_theta_at(0)
                )
                self._block_objects.append(building)
