import math
from typing import List, TYPE_CHECKING, Tuple, Union
from panda3d.bullet import BulletBoxShape, BulletGhostNode
from panda3d.core import TransformState
from panda3d.core import Vec3, BitMask32

from pgdrive.constants import CollisionGroup
from pgdrive.utils.coordinates_shift import panda_position, panda_heading
import numpy as np

from pgdrive.constants import Decoration, BodyName
from pgdrive.scene_creator.lane.abs_lane import AbstractLane
from pgdrive.scene_creator.lane.circular_lane import CircularLane
from pgdrive.utils.coordinates_shift import panda_position
from pgdrive.utils.math_utils import get_points_bounding_box
from pgdrive.world.pg_world import PGWorld

if TYPE_CHECKING:
    from pgdrive.scene_creator.blocks.block import BlockSocket
    from pgdrive.scene_creator.road.road import Road
    from pgdrive.scene_creator.road.road_network import RoadNetwork


def get_lanes_on_road(road: "Road", roadnet: "RoadNetwork") -> List["AbstractLane"]:
    return roadnet.graph[road.start_node][road.end_node]


def block_socket_merge(
    socket_1: "BlockSocket", socket_2: "BlockSocket", global_network: "RoadNetwork", positive_merge: False
):
    global_network.graph[socket_1.positive_road.start_node][socket_2.negative_road.start_node] = \
        global_network.graph[socket_1.positive_road.start_node].pop(socket_1.positive_road.end_node)

    global_network.graph[socket_2.positive_road.start_node][socket_1.negative_road.start_node] = \
        global_network.graph[socket_2.positive_road.start_node].pop(socket_2.positive_road.end_node)


def check_lane_on_road(road_network: "RoadNetwork", lane, positive: float = 0, ignored=None) -> bool:
    """
    Calculate if the new lane intersects with other lanes in current road network
    The return Value is True when cross
    Note: the decoration road will be ignored in default
    """
    graph = road_network.graph
    for _from, to_dict in graph.items():
        for _to, lanes in to_dict.items():
            if ignored and (_from, _to) == ignored:
                continue
            if (_from, _to) == (Decoration.start, Decoration.end):
                continue
            if len(lanes) == 0:
                continue
            x_max_1, x_min_1, y_max_1, y_min_1 = get_road_bounding_box(lanes)
            x_max_2, x_min_2, y_max_2, y_min_2 = get_road_bounding_box([lane])
            if x_min_1 > x_max_2 or x_min_2 > x_max_1 or y_min_1 > y_max_2 or y_min_2 > y_max_1:
                continue
            for _id, l in enumerate(lanes):
                for i in range(1, int(lane.length), 1):
                    sample_point = lane.position(i, positive * lane.width_at(i) / 2.0)
                    longitudinal, lateral = l.local_coordinates(sample_point)
                    is_on = math.fabs(lateral) <= l.width_at(longitudinal) / 2.0 and 0 <= longitudinal <= l.length
                    if is_on:
                        return True
    return False


def get_road_bounding_box(lanes, extra_lateral=3) -> Tuple:
    """
    Return (x_max, x_min, y_max, y_min) as bounding box of this road
    :param lanes: Lanes in this road
    :param extra_lateral: extra width in lateral direction, usually sidewalk width
    :return: x_max, x_min, y_max, y_min
    """
    line_points = get_curve_contour(lanes, extra_lateral) if isinstance(lanes[0], CircularLane) \
        else get_straight_contour(lanes, extra_lateral)
    return get_points_bounding_box(line_points)


def get_straight_contour(lanes, extra_lateral) -> List:
    """
    Get several points as bounding box of this road
    :param lanes: lanes contained in road
    :param extra_lateral: extra length in lateral direction, usually sidewalk
    :return: points
    :param lanes:
    :return:
    """
    ret = []
    for lane, dir in [(lanes[0], -1), (lanes[-1], 1)]:
        ret.append(lane.position(0.1, dir * (lane.width / 2.0 + extra_lateral)))
        ret.append(lane.position(lane.length - 0.1, dir * (lane.width / 2.0 + extra_lateral)))
    return ret


def get_curve_contour(lanes, extra_lateral) -> List:
    """
    Get several points as bounding box of this road
    :param lanes: lanes contained in road
    :param extra_lateral: extra length in lateral direction, usually sidewalk
    :return: points
    """
    points = []
    for lane, lateral_dir in [(lanes[0], -1), (lanes[-1], 1)]:
        pi_2 = (np.pi / 2.0)
        points += [
            lane.position(0.1, lateral_dir * (lane.width / 2.0 + extra_lateral)),
            lane.position(lane.length - 0.1, lateral_dir * (lane.width / 2.0 + extra_lateral))
        ]
        start_phase = (lane.start_phase // pi_2) * pi_2
        start_phase += pi_2 if lane.direction == 1 else 0
        for phi_index in range(4):
            phi = start_phase + phi_index * pi_2 * lane.direction
            if lane.direction * phi > lane.direction * lane.end_phase:
                break
            point = lane.center + (lane.radius - lateral_dir * (lane.width / 2.0 + extra_lateral) *
                                   lane.direction) * np.array([math.cos(phi), math.sin(phi)])
            points.append(point)
    return points


def get_all_lanes(roadnet: "RoadNetwork"):
    graph = roadnet.graph
    res = []
    for from_, to_dict in graph.items():
        for _to, lanes in to_dict.items():
            for l in lanes:
                res.append(l)
    return res


def ray_localization(heading: np.ndarray,
                     position: np.ndarray,
                     pg_world: PGWorld,
                     return_all_result=False) -> Union[List[Tuple], Tuple]:
    """
    Get the index of the lane closest to a physx_world position.
    Only used when smoething is on lane ! Otherwise fall back to use get_closest_lane()
    :param heading: heading to help filter lanes
    :param position: a physx_world position [m].
    :param pg_world: PGWorld class
    :param return_all_result: return a list instead of the lane with min L1 distance
    :return: list(closest lane) or closest lane.
    """
    results = pg_world.physics_world.static_world.rayTestAll(
        panda_position(position, 1.0), panda_position(position, -1.0)
    )
    lane_index_dist = []
    if results.hasHits():
        for res in results.getHits():
            if res.getNode().getName() == BodyName.Lane:
                lane = res.getNode().getPythonTag(BodyName.Lane)
                long, _ = lane.info.local_coordinates(position)
                lane_heading = lane.info.heading_at(long)
                dir = np.array([math.cos(lane_heading), math.sin(lane_heading)])
                cosangle = dir.dot(heading) / (np.linalg.norm(dir) * np.linalg.norm(heading))
                if cosangle > 0:
                    lane_index_dist.append((lane.info, lane.index, lane.info.distance(position)))
    if return_all_result:
        ret = []
        if len(lane_index_dist) > 0:
            for lane, index, dist in lane_index_dist:
                ret.append((lane, index, dist))
        sorted(ret, key=lambda k: k[2])
        return ret
    else:
        if len(lane_index_dist) > 0:
            ret_index = np.argmin([d for _, _, d in lane_index_dist])
            lane, index, dist = lane_index_dist[ret_index]
        else:
            lane, index, dist = None, None, None
        return lane, index


def rect_region_detection(
    pg_world: PGWorld,
    position: Tuple,
    heading: float,
    heading_direction_length: float,
    side_direction_width: float,
    detection_group: int,
    height=10
):
    """

     ----------------------------------
     |               *                |  --->>>
     ----------------------------------
     * position
     --->>> heading direction
     ------ longitude length
     | lateral width

     **CAUTION**: position is the middle point of longitude edge

    :param pg_world: PGWorld
    :param position: position in PGDrive
    :param heading: heading in PGDrive [degree]
    :param heading_direction_length: rect length in heading direction
    :param side_direction_width: rect width in side direction
    :param detection_group: which group to detect
    :param height: the detect will be executed from this height to 0
    :return: detection result
    """
    region_detect_start = panda_position(position, z=height)
    region_detect_end = panda_position(position, z=-1)
    tsFrom = TransformState.makePosHpr(region_detect_start, Vec3(panda_heading(heading), 0, 0))
    tsTo = TransformState.makePosHpr(region_detect_end, Vec3(panda_heading(heading), 0, 0))

    shape = BulletBoxShape(Vec3(heading_direction_length / 2, side_direction_width / 2, 1))
    penetration = 0.0

    result = pg_world.physics_world.dynamic_world.sweep_test_closest(
        shape, tsFrom, tsTo, BitMask32.bit(detection_group), penetration
    )
    return result
