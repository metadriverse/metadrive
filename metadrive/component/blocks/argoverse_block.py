import copy
from typing import Dict

from metadrive.component.blocks.base_block import BaseBlock
from metadrive.component.lane.argoverse_lane import ArgoverseLane
from metadrive.component.road.road import Road
from metadrive.component.road.road_network import RoadNetwork


class ArgoverseBlock(BaseBlock):
    def __init__(self, block_index: int, global_network: RoadNetwork, argoverse_lanes: Dict[int, ArgoverseLane]):
        """
        No randomization when create argoverse block, Split Argoverse Map to several blocks to boost efficiency
        """
        super(ArgoverseBlock, self).__init__(block_index, global_network, 0)
        self.origin_block_network = None
        self.argo_lanes = argoverse_lanes

    def _sample_topology(self) -> bool:
        self.add_lanes()
        return True

    def _create_in_world(self):
        # redundant_road_to_remove = []
        # self.origin_block_network = copy.copy(self.block_network)
        # for _from, dest in self.block_network.graph.items():
        #     for _to, lanes in dest.items():
        #         for lane in lanes:
        #             if lane.start_node != _from and lane.end_node != _to:
        #                 redundant_road_to_remove.append(Road(lane.start_node, lane.end_node))
        # for road in redundant_road_to_remove:
        #     self.block_network.remove_road(road)
        return super(ArgoverseBlock, self)._create_in_world()

    def add_lanes(self):
        for lane in self.argo_lanes.values():
            ret = [lane.id]
            # find left neighbour
            left_id = lane.l_neighbor_id if lane.l_neighbor_id in self.argo_lanes else None
            in_same_dir = True if left_id is not None and lane.is_in_same_direction(
                self.argo_lanes[left_id]) else False
            while left_id is not None and in_same_dir:
                ret.append(left_id)
                left_id = self.argo_lanes[left_id].l_neighbor_id
                in_same_dir = True if left_id is not None and lane.is_in_same_direction(
                    self.argo_lanes[left_id]) else False

            # find right neighbour
            right_id = lane.r_neighbor_id if lane.r_neighbor_id in self.argo_lanes else None
            in_same_dir = True if right_id is not None and lane.is_in_same_direction(
                self.argo_lanes[right_id]) else False
            while right_id is not None and in_same_dir:
                ret.append(right_id)
                right_id = self.argo_lanes[right_id].r_neighbor_id
                in_same_dir = True if right_id is not None and lane.is_in_same_direction(
                    self.argo_lanes[right_id]) else False
            self.block_network.add_road(Road(lane.start_node, lane.end_node), [self.argo_lanes[id] for id in ret])
