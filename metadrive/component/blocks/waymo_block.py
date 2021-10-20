from metadrive.component.blocks.base_block import BaseBlock
from metadrive.component.lane.waymo_lane import WaymoLane
from metadrive.constants import WaymoLaneProperty
from metadrive.component.road.road import Road

class WaymoBlock(BaseBlock):

    def __init__(self, block_index: int, global_network, random_seed, waymo_map_data:dict):
        self.waymo_map_data = waymo_map_data
        super(WaymoBlock, self).__init__(block_index, global_network, random_seed)

    def _sample_topology(self) -> bool:
        waymo_lanes = []
        for lane_id, data in self.waymo_map_data.items():
            if data.get("type", False)==WaymoLaneProperty.TYPE:
                if len(data[WaymoLaneProperty.CENTER_POINTS])<=1:
                    continue
                waymo_lane = WaymoLane(lane_id, self.waymo_map_data)
                waymo_lanes.append(waymo_lane)
        self.block_network.add_road(Road("test","test"), waymo_lanes)
        return True

