from typing import Tuple, Sequence
from metadrive.constants import CamMask

from panda3d.core import LVector3, NodePath

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CollisionGroup

LaneIndex = Tuple[str, str, int]


class BaseTrafficParticipant(BaseObject):
    TYPE_NAME = None
    COLLISION_MASK = CollisionGroup.TrafficParticipants
    HEIGHT = None

    def __init__(self, position: Sequence[float], heading_theta: float = 0., random_seed=None, name=None, config=None):
        super(BaseTrafficParticipant, self).__init__(random_seed=random_seed, name=name, config=config)
        self.set_position(position, self.HEIGHT / 2 if hasattr(self, "HEIGHT") else 0)
        self.set_heading_theta(heading_theta)
        assert self.MASS is not None, "No mass for {}".format(self.class_name)
        assert self.TYPE_NAME is not None, "No name for {}".format(self.class_name)
        assert self.COLLISION_MASK is not None, "No collision group for {}".format(self.class_name)

    def reset(self, position: Sequence[float], heading_theta: float = 0., random_seed=None, name=None, *args, **kwargs):
        self.seed(random_seed)
        self.rename(name)
        self.set_position(position, self.HEIGHT / 2 if hasattr(self, "HEIGHT") else 0)
        self.set_heading_theta(heading_theta)
        assert self.MASS is not None, "No mass for {}".format(self.class_name)
        assert self.TYPE_NAME is not None, "No name for {}".format(self.class_name)
        assert self.COLLISION_MASK is not None, "No collision group for {}".format(self.class_name)

    @property
    def top_down_width(self):
        return self.WIDTH

    @property
    def top_down_length(self):
        return self.LENGTH

    # def set_roll(self, roll):
    #     self.origin.setP(roll)
    #
    # def set_pitch(self, pitch):
    #     self.origin.setR(pitch)
    #
    # @property
    # def roll(self):
    #     return np.deg2rad(self.origin.getP())
    #
    # @property
    # def pitch(self):
    #     return np.deg2rad(self.origin.getR())

    def add_body(self, physics_body):
        super(BaseTrafficParticipant, self).add_body(physics_body)
        self._body.set_friction(0.)
        self._body.set_anisotropic_friction(LVector3(0., 0., 0.))

    def standup(self):
        self.set_pitch(0)
        self.set_roll(0)

    def set_position(self, position, height=None):
        super(BaseTrafficParticipant, self).set_position(position, height)
        self.standup()

    def show_coordinates(self):
        if not self.need_show_coordinates:
            return
        if self.coordinates_debug_np is not None:
            self.coordinates_debug_np.reparentTo(self.origin)
        height = self.HEIGHT
        self.coordinates_debug_np = NodePath("debug coordinate")
        self.coordinates_debug_np.hide(CamMask.AllOn)
        self.coordinates_debug_np.show(CamMask.MainCam)
        x = self.engine._draw_line_3d([0, 0, height], [1, 0, height], [1, 0, 0, 1], 3)
        y = self.engine._draw_line_3d([0, 0, height], [0, 0.5, height], [0, 1, 0, 1], 3)
        z = self.engine._draw_line_3d([0, 0, height], [0, 0, height + 0.5], [0, 0, 1, 1], 3)
        x.reparentTo(self.coordinates_debug_np)
        y.reparentTo(self.coordinates_debug_np)
        z.reparentTo(self.coordinates_debug_np)
        self.coordinates_debug_np.reparentTo(self.origin)
