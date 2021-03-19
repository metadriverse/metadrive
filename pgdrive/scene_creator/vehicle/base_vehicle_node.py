from panda3d.bullet import BulletRigidBodyNode
from pgdrive.constants import BodyName


class BaseVehilceNode(BulletRigidBodyNode):
    """
    Collision Properties should place here, info here can used for collision callback
    """
    def __init__(self, body_name: str):
        BulletRigidBodyNode.__init__(self, body_name)
        BulletRigidBodyNode.setPythonTag(self, body_name, self)
        self.crash_vehicle = False
        self.crash_object = False
        self.crash_sidewalk = False

    def init_collision_info(self):
        self.crash_vehicle = False
        self.crash_object = False
        self.crash_sidewalk = False
