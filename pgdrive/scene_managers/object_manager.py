from typing import Union, Callable, Optional

from pgdrive.scene_creator.blocks.curve import Curve
from pgdrive.scene_creator.blocks.ramp import InRampOnStraight, OutRampOnStraight
from pgdrive.scene_creator.blocks.straight import Straight
from pgdrive.scene_creator.lane.abs_lane import AbstractLane
from pgdrive.scene_creator.map.map import Map
from pgdrive.scene_creator.object.static_object import StaticObject
from pgdrive.scene_creator.object.traffic_object import TrafficSign
from pgdrive.scene_creator.road.road import Road
from pgdrive.scene_creator.road.road_network import LaneIndex
from pgdrive.scene_managers.base_manager import BaseManager
from pgdrive.utils.engine_utils import get_pgdrive_engine


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
    PROHIBIT_SCENE_PROB = 0.5  # the reset is the probability of break_down_scene

    def __init__(self, config=None):
        super(TrafficSignManager, self).__init__()
        self._block_objects = {}
        self.accident_prob = 0.

        # init random engine
        super(TrafficSignManager, self).__init__()

    def before_reset(self):
        """
        Clear all objects in th scene
        """
        self.clear_objects()
        self.accident_prob = self.pgdrive_engine.global_config["accident_prob"]
        map = self.pgdrive_engine.map_manager.current_map
        for block in map.blocks:
            block.construct_block_buildings(self)

    def clear_objects(self, filter_func: Optional[Callable] = None):
        super(TrafficSignManager, self).clear_objects()
        for block_object in self._block_objects.values():
            block_object.node_path.detachNode()
        self._block_objects = {}

    def add_block_buildings(self, building: StaticObject, render_node):
        self._block_objects[building.id] = building
        building.node_path.reparentTo(render_node)

    def spawn_object(
        self,
        object_class: Union[TrafficSign, str],
        *args,
        **kwargs,
    ):
        """
        Spawn an object by assigning its type and position on the lane
        :param object_class: object name or the class name of the object
        """
        cls = None
        for t in TrafficSign.type():
            if t.__name__ == object_class or t.NAME == object_class:
                cls = t
        if cls is None:
            raise ValueError("No object named {}, so it can not be spawned".format(object_class))
        return super(TrafficSignManager, self).spawn_object(cls, *args, **kwargs)

    def reset(self):
        """
        Generate an accident scene or construction scene on block
        :return: None
        """
        engine = get_pgdrive_engine()
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

            _debug = engine.world_config["_debug_crash_object"]
            if _debug:
                on_left = True

            lane = accident_road.get_lanes(engine.current_map.road_network)[accident_lane_idx]
            longitude = lane.length - self.ACCIDENT_AREA_LEN

            if lane.length < self.ACCIDENT_LANE_MIN_LEN:
                continue

            # generate scene
            block.PROHIBIT_TRAFFIC_GENERATION = True

            # TODO(pzh) This might not worked in MARL envs when the route is also has changeable lanes.
            lateral_len = engine.current_map.config[engine.current_map.LANE_WIDTH]

            lane = engine.current_map.road_network.get_lane(accident_road.lane_index(accident_lane_idx))
            if self.np_random.rand() > 0.5 or _debug:
                self.prohibit_scene(lane, accident_road.lane_index(accident_lane_idx), longitude, lateral_len, on_left)
            else:
                accident_lane_idx = self.np_random.randint(engine.current_map.config[engine.current_map.LANE_NUM])
                self.break_down_scene(lane, accident_road.lane_index(accident_lane_idx), longitude)

    def break_down_scene(self, lane: AbstractLane, lane_index: LaneIndex, longitude: float):
        engine = get_pgdrive_engine()
        breakdown_vehicle = engine.traffic_manager.spawn_object(
            engine.traffic_manager.random_vehicle_type(), lane, longitude, False
        )
        breakdown_vehicle.attach_to_world(engine.pbr_worldNP, engine.physics_world)
        breakdown_vehicle.set_break_down()

        alert = self.spawn_object("Traffic Triangle", lane, lane_index, longitude - self.ALERT_DIST, 0)
        alert.attach_to_world(engine.pbr_worldNP, engine.physics_world)

    def prohibit_scene(
        self, lane: AbstractLane, lane_index: LaneIndex, longitude_position: float, lateral_len: float, on_left=False
    ):
        """
        Generate an accident scene on the most left or most right lane
        :param lane lane object
        :param lane_index: lane index used to find the lane in map
        :param longitude_position: longitude position of the accident on the lane
        :param lateral_len: the distance that traffic cones extend on lateral direction
        :param on_left: on left or right side
        :return: None
        """
        engine = get_pgdrive_engine()
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
            cone = self.spawn_object("Traffic Cone", lane, lane_index, *p_)
            cone.attach_to_world(engine.pbr_worldNP, engine.physics_world)
            # TODO refactor traffic and traffic system to make it compatible

    def destroy(self):
        self._block_objects = {}
        super(TrafficSignManager, self).destroy()

    @property
    def objects(self):
        return list(self._spawned_objects.values()) + list(self._block_objects.values())
