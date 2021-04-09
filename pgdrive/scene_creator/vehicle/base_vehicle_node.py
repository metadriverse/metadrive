from panda3d.bullet import BulletRigidBodyNode


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

        # lane line detection
        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False

    def init_collision_info(self):
        self.crash_vehicle = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False
