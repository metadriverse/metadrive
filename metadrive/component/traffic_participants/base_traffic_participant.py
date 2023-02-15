from typing import Tuple, Sequence

from metadrive.base_class.base_object import BaseObject

LaneIndex = Tuple[str, str, int]


class BaseTrafficParticipant(BaseObject):
    NAME = None
    COLLISION_GROUP = None

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
