import re
from typing import List, Tuple

from metadrive.constants import Decoration

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class Road:
    """
    Road is a bunch of lanes connecting two nodes, one start and the other end
    """
    NEGATIVE_DIR = "-"

    def __init__(self, start_node: str, end_node: str):
        self.start_node = start_node
        self.end_node = end_node

    def get_lanes(self, road_network):
        return road_network.graph[self.start_node][self.end_node]

    def __neg__(self):
        sub_index = self.end_node.find(Road.NEGATIVE_DIR)
        if sub_index == -1:
            return Road(Road.NEGATIVE_DIR + self.end_node, Road.NEGATIVE_DIR + self.start_node)
        else:
            return Road(self.end_node[sub_index + 1:], self.start_node[sub_index + 1:])

    def is_negative_road(self):
        return False if self.end_node.find(Road.NEGATIVE_DIR) == -1 else True

    def is_valid_road(self):
        return False if self.start_node == Decoration.start and self.end_node == Decoration.end else True

    def lane_index(self, index: int) -> LaneIndex:
        return self.start_node, self.end_node, index

    def lane_num(self, road_network):
        return len(self.get_lanes(road_network))

    def block_ID(self):
        search_node = self.end_node if not self.is_negative_road() else self.start_node
        if re.search(">", search_node) is not None:
            return ">"
        block_id = re.search("[a-zA-Z$]", search_node).group(0)
        return block_id

    def __eq__(self, other):
        if isinstance(other, Road):
            return True if self.start_node == other.start_node and self.end_node == other.end_node else False
        else:
            return super(Road, self).__eq__(other)

    def __repr__(self):
        return "Road from {} to {}".format(self.start_node, self.end_node)

    def __hash__(self):
        return hash((self.start_node, self.end_node))

    def to_json(self):
        return (self.start_node, self.end_node)
