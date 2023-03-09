from typing import Tuple, Sequence

from panda3d.core import LVector3, NodePath

from metadrive.base_class.base_object import BaseObject

LaneIndex = Tuple[str, str, int]


class BaseTrafficParticipant(BaseObject):
    NAME = None
    COLLISION_GROUP = None
    HEIGHT = None

    def __init__(self, position: Sequence[float], heading_theta: float = 0., random_seed=None):
        super(BaseTrafficParticipant, self).__init__(random_seed=random_seed)
        self.set_position(position, self.HEIGHT / 2 if hasattr(self, "HEIGHT") else 0)
        self.set_heading_theta(heading_theta)
        assert self.MASS is not None, "No mass for {}".format(self.class_name)
        assert self.NAME is not None, "No name for {}".format(self.class_name)
        assert self.COLLISION_GROUP is not None, "No collision group for {}".format(self.class_name)

    def top_down_color(self):
        raise NotImplementedError(
            "Implement this func for rendering class {} in top down renderer".format(self.class_name)
        )

    def set_roll(self, roll):
        self.origin.setP(roll)

    def set_pitch(self, pitch):
        self.origin.setR(pitch)

    @property
    def roll(self):
        return self.origin.getP()

    @property
    def pitch(self):
        return self.origin.getR()

    def add_body(self, physics_body):
        super(BaseTrafficParticipant, self).add_body(physics_body)
        self._body.set_friction(0.)
        self._body.set_anisotropic_friction(LVector3(0., 0., 0.))

    def show_coordinates(self):
        if not self.need_show_coordinates:
            return
        if self.coordinates_debug_np is not None:
            self.coordinates_debug_np.reparentTo(self.origin)
        height = self.HEIGHT
        self.coordinates_debug_np = NodePath("debug coordinate")
        x = self.engine.add_line([0, 0, height], [1, 0, height], [1, 0, 0, 1], 1)
        y = self.engine.add_line([0, 0, height], [0, 0.5, height], [1, 0, 0, 1], 1)
        z = self.engine.add_line([0, 0, height], [0, 0, height + 0.25], [0, 0, 1, 1], 2)
        x.reparentTo(self.coordinates_debug_np)
        y.reparentTo(self.coordinates_debug_np)
        z.reparentTo(self.coordinates_debug_np)
        self.coordinates_debug_np.reparentTo(self.origin)
