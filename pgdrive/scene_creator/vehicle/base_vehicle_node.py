from panda3d.bullet import BulletRigidBodyNode


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