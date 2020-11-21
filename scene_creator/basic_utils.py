import copy
from typing import Tuple, Union, List
import numpy as np
from scene_creator.blocks.block import Block, BlockSocket
from scene_creator.lanes.circular_lane import CircularLane
from scene_creator.lanes.lane import LineType, AbstractLane, LineColor
from scene_creator.lanes.straight_lane import StraightLane
from scene_creator.road.road import Road
from scene_creator.road.road_network import RoadNetwork
from utils.math_utils import get_vertical_vector, check_lane_on_road


class Decoration:
    start = "decoration"
    end = "decoration_"


class Goal:
    """
    Goal at intersection
    The keywors 0, 1, 2 should be reserved, and only be used in roundabout and intersection
    """

    RIGHT = 0
    STRAIGHT = 1
    LEFT = 2
    ADVERSE = 3  # Useless now


def sharpbend(
    previous_lane: StraightLane,
    following_lane_length,
    radius: float,
    angle: float,
    clockwise: bool = True,
    width: float = AbstractLane.DEFAULT_WIDTH,
    line_types: Tuple[LineType, LineType] = None,
    forbidden: bool = False,
    speed_limit: float = 20,
    priority: int = 0
):
    bend_direction = 1 if clockwise else -1
    center = previous_lane.position(previous_lane.length, bend_direction * radius)
    p_lateral = previous_lane.direction_lateral
    x, y = p_lateral
    start_phase = 0
    if y == 0:
        start_phase = 0 if x < 0 else -np.pi
    elif x == 0:
        start_phase = np.pi / 2 if y < 0 else -np.pi / 2
    else:
        base_angel = np.arctan(y / x)
        if x < 0:
            start_phase = base_angel
        elif y < 0:
            start_phase = np.pi + base_angel
        elif y > 0:
            start_phase = -np.pi + base_angel
    end_phase = start_phase + angle
    if not clockwise:
        start_phase = start_phase - np.pi
        end_phase = start_phase - angle
    bend = CircularLane(
        center, radius, start_phase, end_phase, clockwise, width, line_types, forbidden, speed_limit, priority
    )
    length = 2 * radius * angle / 2
    bend_end = bend.position(length, 0)
    next_lane_heading = get_vertical_vector(bend_end - center)
    nxt_dir = next_lane_heading[0] if not clockwise else next_lane_heading[1]
    nxt_dir = np.asarray(nxt_dir)
    following_lane_end = nxt_dir * following_lane_length + bend_end
    following_lane = StraightLane(bend_end, following_lane_end, width, line_types, forbidden, speed_limit, priority)
    return bend, following_lane


def CreateRoadFrom(
    lane: Union[AbstractLane, StraightLane, CircularLane],
    lane_num: int,
    road: Road,
    roadnet_to_add_lanes: RoadNetwork,  # mostly, block network
    roadnet_to_check_cross: RoadNetwork,  # mostly, previous global_network
    toward_smaller_Lane_index: bool = True,
    ignore_start: str = None,
    ignore_end: str = None,
    center_line_type=Block.CENTER_LINE_TYPE,
    detect_one_side=True
) -> bool:
    """
        | | | |
        | | | |
        | | | |
        | | | |
    <-----smaller direction = inside direction
    Usage: give the far left lane, then create lane_num lanes including itself
    :return if the lanes created cross other lanes
    """
    lane_num -= 1  # include lane itself
    origin_lane = lane
    lanes = []
    lane_width = lane.width_at(0)
    for i in range(lane_num, 0, -1):
        side_lane = copy.deepcopy(lane)
        if isinstance(lane, StraightLane):
            width = -lane_width if toward_smaller_Lane_index else lane_width
            start = side_lane.position(0, width)
            end = side_lane.position(side_lane.length, width)
            side_lane.start = start
            side_lane.end = end
        elif isinstance(lane, CircularLane):
            clockwise = True if lane.direction == 1 else False
            radius1 = lane.radius
            if not toward_smaller_Lane_index:
                radius2 = radius1 - lane_width if clockwise else radius1 + lane_width
            else:
                radius2 = radius1 + lane_width if clockwise else radius1 - lane_width
            side_lane.radius = radius2
            side_lane.update_length()
        if i == 1:
            side_lane.line_types = [LineType.CONTINUOUS, LineType.STRIPED]
        else:
            side_lane.line_types = [LineType.STRIPED, LineType.STRIPED]
        lanes.append(side_lane)
        lane = side_lane
    if toward_smaller_Lane_index:
        lanes.reverse()
    lanes.append(origin_lane)

    # check the left lane and right lane
    ignore = (ignore_start, ignore_end)
    from scene_creator.blocks.block import Block
    factor = (Block.SIDE_WALK_WIDTH + Block.SIDE_WALK_LINE_DIST + lane_width / 2.0) * 2.0 / lane_width
    if not detect_one_side:
        # Because of side walk, the width of side walk should be consider
        no_cross = not (
            check_lane_on_road(roadnet_to_check_cross, origin_lane, factor, ignore)
            or check_lane_on_road(roadnet_to_check_cross, lanes[0], -0.95, ignore)
        )
    else:
        no_cross = not check_lane_on_road(roadnet_to_check_cross, origin_lane, factor, ignore)
    for l in lanes:
        roadnet_to_add_lanes.add_lane(road.start_node, road.end_node, l)
    if lane_num == 0:
        lanes[-1].line_types = [center_line_type, LineType.SIDE]
    return no_cross


def ExtendStraightLane(lane: StraightLane, extend_length: float, line_types: (LineType, LineType)) -> StraightLane:
    new_lane = copy.deepcopy(lane)
    start_point = lane.end
    end_point = lane.position(lane.length + extend_length, 0)
    new_lane.start = start_point
    new_lane.end = end_point
    new_lane.line_types = line_types
    new_lane.update_length()
    return new_lane


def get_lanes_on_road(road: Road, roadnet: RoadNetwork) -> List[AbstractLane]:
    return roadnet.graph[road.start_node][road.end_node]


def CreateAdverseRoad(
    positive_road: Road,
    roadnet_to_get_road: RoadNetwork,  # mostly, block network
    roadnet_to_check_cross: RoadNetwork,  # mostly, previous global network
    ignore_start: str = None,
    ignore_end: str = None
) -> (str, str, bool):
    adverse_road = -positive_road
    lanes = get_lanes_on_road(positive_road, roadnet_to_get_road)
    reference_lane = lanes[-1]
    num = len(lanes) * 2
    width = reference_lane.width_at(0)
    if isinstance(reference_lane, StraightLane):
        start_point = reference_lane.position(lanes[-1].length, -(num - 1) * width)
        end_point = reference_lane.position(0, -(num - 1) * width)
        symmetric_lane = StraightLane(
            start_point, end_point, width, lanes[-1].line_types, reference_lane.forbidden, reference_lane.speed_limit,
            reference_lane.priority
        )
    elif isinstance(reference_lane, CircularLane):
        start_phase = reference_lane.end_phase
        end_phase = reference_lane.start_phase
        clockwise = False if reference_lane.direction == 1 else True
        if not clockwise:
            radius = reference_lane.radius + (num - 1) * width
        else:
            radius = reference_lane.radius - (num - 1) * width
        symmetric_lane = CircularLane(
            reference_lane.center, radius, start_phase, end_phase, clockwise, width, reference_lane.line_types,
            reference_lane.forbidden, reference_lane.speed_limit, reference_lane.priority
        )
    else:
        raise ValueError("Creating other lanes is not supported")
    success = CreateRoadFrom(
        symmetric_lane,
        int(num / 2),
        adverse_road,
        roadnet_to_get_road,
        roadnet_to_check_cross,
        ignore_start=ignore_start,
        ignore_end=ignore_end
    )
    inner_lane = roadnet_to_get_road.get_lane((adverse_road.start_node, adverse_road.end_node, 0))
    inner_lane.line_types = [LineType.NONE, LineType.STRIPED
                             ] if int(num / 2) > 1 else [LineType.NONE, LineType.CONTINUOUS]
    positive_road.get_lanes(roadnet_to_get_road)[0].line_color = [LineColor.YELLOW, LineColor.GREY]
    return success


def block_socket_merge(
    socket_1: BlockSocket, socket_2: BlockSocket, global_network: RoadNetwork, positive_merge: False
):
    global_network.graph[socket_1.positive_road.start_node][socket_2.negative_road.start_node] = \
        global_network.graph[socket_1.positive_road.start_node].pop(socket_1.positive_road.end_node)

    global_network.graph[socket_2.positive_road.start_node][socket_1.negative_road.start_node] = \
        global_network.graph[socket_2.positive_road.start_node].pop(socket_2.positive_road.end_node)
