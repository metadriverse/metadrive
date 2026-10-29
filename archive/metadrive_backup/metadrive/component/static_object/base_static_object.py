from typing import Sequence, Tuple
from metadrive.constants import CamMask
from metadrive.type import MetaDriveType

from panda3d.core import NodePath

from metadrive.base_class.base_object import BaseObject

LaneIndex = Tuple[str, str, int]


class BaseStaticObject(BaseObject):
    MASS = 1
    HEIGHT = None

    def __init__(self, position: Sequence[float], heading_theta: float = 0., lane=None, random_seed=None, name=None):
        """
        :param lane: the lane to spawn object
        :param position: cartesian position of object in the surface
        :param heading_theta: the angle from positive direction of horizontal axis
        """
        super(BaseStaticObject, self).__init__(random_seed=random_seed, name=name)
        self.set_metadrive_type(MetaDriveType.TRAFFIC_OBJECT)
        self.set_position(position, self.HEIGHT / 2 if hasattr(self, "HEIGHT") else 0)
        self.set_heading_theta(heading_theta)
        self.lane = lane
        self.lane_index = lane.index if lane is not None else None

    def show_coordinates(self):
        if self.coordinates_debug_np is not None:
            return
        height = self.HEIGHT
        self.coordinates_debug_np = NodePath("debug coordinate")
        self.coordinates_debug_np.hide(CamMask.AllOn)
        self.coordinates_debug_np.show(CamMask.MainCam)
        x = self.engine._draw_line_3d([0, 0, height], [2, 0, height], [1, 0, 0, 1], 3)
        y = self.engine._draw_line_3d([0, 0, height], [0, 1, height], [0, 1, 0, 1], 3)
        z = self.engine._draw_line_3d([0, 0, height], [0, 0, height + 0.5], [0, 0, 1, 1], 3)
        x.reparentTo(self.coordinates_debug_np)
        y.reparentTo(self.coordinates_debug_np)
        z.reparentTo(self.coordinates_debug_np)
        self.coordinates_debug_np.reparentTo(self.origin)

    def reset(self, position, heading_theta, lane=None, random_seed=None, name=None, *args, **kwargs):
        self.seed(random_seed)
        self.rename(name)
        self.set_pitch(0)
        self.set_roll(0)
        self.set_position(position, self.HEIGHT / 2 if hasattr(self, "HEIGHT") else 0)
        self.set_heading_theta(heading_theta)
        self.lane = lane
        self.lane_index = lane.index if lane is not None else None
