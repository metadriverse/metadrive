from pgdrive.scene_creator.object.traffic_object import Object
from pgdrive.scene_creator.road.road_network import LaneIndex
from pgdrive.scene_creator.lane.abs_lane import AbstractLane
from pgdrive.world.pg_world import PGWorld


class ObjectsManager:
    """
    This class is used to manager all static object, such as traffic cones, warning tripod.
    """
    def __init__(self):
        self.spawned_objects = []

    def clear_objects(self, pg_world: PGWorld):
        """
        Clear all objects in th scene
        """
        for obj in self.spawned_objects:
            obj.destroy(pg_world=pg_world)
        self.spawned_objects = []

    def spawn_one_object(
        self, object_type: str, lane: AbstractLane, lane_index: LaneIndex, longitude: float, lateral: float
    ) -> None:
        """
        Spawn an object by assigning its type and position on the lane
        :param object_type: object name or the class name of the object
        :param lane: object will be spawned on this lane
        :param lane_index:the lane index of the spawn point
        :param longitude: longitude position on this lane
        :param lateral: lateral position on  this lane
        :return: None
        """
        for t in Object.type():
            if t.__name__ == object_type or t.NAME == object_type:
                obj = t.make_on_lane(lane, lane_index, longitude, lateral)
                self.spawned_objects.append(obj)
                return obj
        raise ValueError("No object named {}, so it can not be spawned".format(object_type))
