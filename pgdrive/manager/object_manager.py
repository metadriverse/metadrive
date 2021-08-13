from typing import Union, Callable, Optional
from pgdrive.component.static_object.traffic_object import TrafficCone, TrafficTriangle

from pgdrive.component.blocks.curve import Curve
from pgdrive.component.blocks.ramp import InRampOnStraight, OutRampOnStraight
from pgdrive.component.blocks.straight import Straight
from pgdrive.component.lane.abs_lane import AbstractLane
from pgdrive.component.road.road import Road
from pgdrive.component.road.road_network import LaneIndex
from pgdrive.component.static_object.traffic_object import TrafficSign
from pgdrive.engine.engine_utils import get_engine
from pgdrive.manager.base_manager import BaseManager


class TrafficSignManager(BaseManager):
    """
    This class is used to manager all static object, such as traffic cones, warning tripod.
    """

    # the distance between break-down vehicle and alert
    ALERT_DIST = 10

    # accident scene setting
    ACCIDENT_AREA_LEN = 10
    ACCIDENT_LANE_MIN_LEN = 50

    # distance between two cones
    CONE_LONGITUDE = 2
    CONE_LATERAL = 1
    PROHIBIT_SCENE_PROB = 0.  # the reset is the probability of break_down_scene

    def __init__(self):
        super(TrafficSignManager, self).__init__()
        self._block_objects = {}
        self.accident_prob = 0.
        self.object = []

    def before_reset(self):
        """
        Clear all objects in th scene
        """
        self.clear_objects()
        self.accident_prob = self.engine.global_config["accident_prob"]
        map = self.engine.map_manager.current_map
        for block in map.blocks:
            block.construct_block_buildings(self)

    def clear_objects(self):
        self.engine.clear_objects(lambda o: isinstance(o, TrafficSign))
        for block_object in self._block_objects.values():
            block_object.origin.detachNode()
        self._block_objects = {}

    def add_block_buildings(self, building, render_node):
        self._block_objects[building.id] = building
        building.origin.reparentTo(render_node)

    def reset(self):
        """
        Generate an accident scene or construction scene on block
        :return: None
        """
        engine = get_engine()
        accident_prob = self.accident_prob
        if abs(accident_prob - 0.0) < 1e-2:
            return
        for block in engine.current_map.blocks:
            if type(block) not in [Straight, Curve, InRampOnStraight, OutRampOnStraight]:
                # blocks with exists do not generate accident scene
                continue
            if self.np_random.rand() > accident_prob:
                # prob filter
                continue

            road_1 = Road(block.pre_block_socket.positive_road.end_node, block.road_node(0, 0))
            road_2 = Road(block.road_node(0, 0), block.road_node(0, 1)) if not isinstance(block, Straight) else None
            accident_road = self.np_random.choice([road_1, road_2]) if not isinstance(block, Curve) else road_2
            accident_road = road_1 if accident_road is None else accident_road
            is_ramp = isinstance(block, InRampOnStraight) or isinstance(block, OutRampOnStraight)
            on_left = True if self.np_random.rand() > 0.5 or (accident_road is road_2 and is_ramp) else False
            accident_lane_idx = 0 if on_left else -1

            _debug = engine.global_config["_debug_crash_object"]
            if _debug:
                on_left = True

            lane = accident_road.get_lanes(engine.current_map.road_network)[accident_lane_idx]
            longitude = lane.length - self.ACCIDENT_AREA_LEN

            if lane.length < self.ACCIDENT_LANE_MIN_LEN:
                continue

            # generate scene
            block.PROHIBIT_TRAFFIC_GENERATION = True

            lateral_len = engine.current_map.config[engine.current_map.LANE_WIDTH]

            lane = engine.current_map.road_network.get_lane(accident_road.lane_index(accident_lane_idx))
            if self.np_random.rand() > self.PROHIBIT_SCENE_PROB or _debug:
                self.prohibit_scene(lane, longitude, lateral_len, on_left)
            else:
                self.break_down_scene(lane, longitude)

    def break_down_scene(self, lane: AbstractLane, longitude: float):
        engine = get_engine()
        breakdown_vehicle = engine.spawn_object(engine.traffic_manager.random_vehicle_type(), lane, longitude, False)
        breakdown_vehicle.set_break_down()
        self.engine.spawn_object(TrafficTriangle, lane=lane, longitude=longitude - self.ALERT_DIST, lateral=0)

    def prohibit_scene(self, lane: AbstractLane, longitude_position: float, lateral_len: float, on_left=False):
        """
        Generate an accident scene on the most left or most right lane
        :param lane lane object
        :param longitude_position: longitude position of the accident on the lane
        :param lateral_len: the distance that traffic cones extend on lateral direction
        :param on_left: on left or right side
        :return: None
        """
        engine = get_engine()
        lat_num = int(lateral_len / self.CONE_LATERAL)
        longitude_num = int(self.ACCIDENT_AREA_LEN / self.CONE_LONGITUDE)
        lat_1 = [lat * self.CONE_LATERAL for lat in range(lat_num)]
        lat_2 = [lat_num * self.CONE_LATERAL] * (longitude_num + 1)
        lat_3 = [(lat_num - lat - 1) * self.CONE_LATERAL for lat in range(int(lat_num))]

        total_long_num = lat_num * 2 + longitude_num + 1
        pos = [
            (long * self.CONE_LONGITUDE, lat - lane.width / 2)
            for long, lat in zip(range(-int(total_long_num / 2), int(total_long_num / 2)), lat_1 + lat_2 + lat_3)
        ]
        left = 1 if on_left else -1
        for p in pos:
            p_ = (p[0] + longitude_position, left * p[1])
            cone = self.engine.spawn_object(TrafficCone, lane=lane, longitude=p_[0], lateral=p_[1])

    def destroy(self):
        self._block_objects = {}
        super(TrafficSignManager, self).destroy()

    @property
    def objects(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, TrafficSign)).values())
