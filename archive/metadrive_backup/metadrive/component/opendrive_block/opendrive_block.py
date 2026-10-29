from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.opendrive_lane import OpenDriveLane
from metadrive.component.road_network.edge_road_network import OpenDriveRoadNetwork
from metadrive.utils.opendrive.map_load import get_lane_width


class OpenDriveBlock(BaseBlock):
    """
    The OpenDriveBlock instance will wrap a section in a Road, which serves as a basic element for building a map
    """
    def __init__(self, block_index: int, global_network, random_seed, section_data):
        self.section_data = section_data
        super(OpenDriveBlock, self).__init__(block_index, global_network, random_seed)

    def _sample_topology(self) -> bool:
        for lane in self.section_data.allLanes:
            # if lane.type == "driving":
            width = get_lane_width(lane)
            opendrive_lane = OpenDriveLane(width, lane)
            self.block_network.add_lane(opendrive_lane)
        return True

    def create_in_world(self):
        """
        The lane line should be created separately
        """
        graph = self.block_network.graph
        for id, lane_info in graph.items():
            lane = lane_info.lane
            self._construct_lane(lane, lane_index=id)

    @property
    def block_network_type(self):
        return OpenDriveRoadNetwork

    def destroy(self):
        self.section_data = None
        super(OpenDriveBlock, self).destroy()
