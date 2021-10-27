import logging
from typing import List, Tuple, Union

from metadrive.component.lane.abs_lane import AbstractLane

logger = logging.getLogger(__name__)

LaneIndex = Union[str, Tuple[str, str, int]]


class BaseRoadNetwork:
    def __init__(self, debug=False):
        self.graph = None

    def clear(self):
        self.graph.clear()

    def get_bounding_box(self):
        raise NotImplementedError

    def add_lane(self, *args, **kwargs) -> None:
        """
        Add one lane to the roadnetwork for querying
        """
        raise NotImplementedError

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane corresponding to a given index in the road network.
        """
        raise NotImplementedError

    def get_closest_lane_index(self, position, return_all=False):
        raise NotImplementedError

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        Breadth-first search of shortest checkpoints from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest checkpoints from start to goal.
        """
        raise NotImplementedError
