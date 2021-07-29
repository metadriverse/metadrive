"""
Physics Node is the subclass of BulletNode (BulletRigidBBodyNode/BulletGhostNode and so on)
Since callback method in BulletPhysicsEngine returns PhysicsNode class and sometimes we need to do some custom
calculation and tell Object about these results, inheriting from these BulletNode class will help communicate between
Physics Callbacks and Object class
"""

from panda3d.bullet import BulletRigidBodyNode
from pgdrive.constants import BodyName


class TrafficSignNode(BulletRigidBodyNode):
    """
    Collision Properties should place here, info here can used for collision callback
    """
    COST_ONCE = True  # cost will give at the first time

    def __init__(self, object_body_name: str):
        BulletRigidBodyNode.__init__(self, object_body_name)
        BulletRigidBodyNode.setPythonTag(self, object_body_name, self)
        self.crashed = False


class LaneNode(BulletRigidBodyNode):
    """
    It is the body of land in panda3d, which can help quickly find current lane of vehicles
    """
    def __init__(self, node_name, lane, lane_index=(str, str, int)):
        """
        Using ray cast to query the lane information
        :param node_name: node_name
        :param lane: CircularLane or StraightLane
        :param lane_index: Lane index
        """
        BulletRigidBodyNode.__init__(self, node_name)
        BulletRigidBodyNode.setPythonTag(self, BodyName.Lane, self)
        from pgdrive.scene_creator.lane.abs_lane import AbstractLane
        assert isinstance(lane, AbstractLane)
        self.info = lane
        self.index = lane_index


class BaseVehicleNode(BulletRigidBodyNode):
    """
    Collision Properties should place here, info here can used for collision callback
    """
    def __init__(self, body_name: str, base_vehicle):
        BulletRigidBodyNode.__init__(self, body_name)
        BulletRigidBodyNode.setPythonTag(self, body_name, self)
        # mutual reference here
        self._base_vehicle = base_vehicle

        self.crash_vehicle = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False

        # lane line detection
        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False

    def init_collision_info(self):
        self.crash_vehicle = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False
        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False

    @property
    def position(self):
        return self._base_vehicle.position

    @property
    def velocity(self):
        return self._base_vehicle.velocity

    def destroy(self):
        # release pointer
        self._base_vehicle = None

    def get_vehicle(self):
        return self._base_vehicle


class TrafficVehicleNode(BulletRigidBodyNode):

    # for lidar detection and other purposes
    def __init__(self, node_name, kinematics_model):
        BulletRigidBodyNode.__init__(self, node_name)
        TrafficVehicleNode.setPythonTag(self, BodyName.Traffic_vehicle, self)
        self.kinematic_model = kinematics_model

    def reset(self, kinematics_model):
        from pgdrive.scene_creator.highway_vehicle.behavior import IDMVehicle
        self.kinematic_model = IDMVehicle.create_from(kinematics_model)
