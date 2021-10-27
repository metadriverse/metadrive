from metadrive.component.vehicle_navigation_module.base_navigation import BaseNavigation
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork


class EdgeNetworkNavigation(BaseNavigation):
    """
   This class define a helper for localizing vehicles and retrieving navigation information.
   It now only support EdgeRoadNetwork
   """
    def __init__(
        self,
        engine,
        show_navi_mark: bool = False,
        random_navi_mark_color=False,
        show_dest_mark=False,
        show_line_to_dest=False
    ):
        super(EdgeNetworkNavigation,
              self).__init__(engine, show_navi_mark, random_navi_mark_color, show_dest_mark, show_line_to_dest)

    def reset(self, map, current_lane, destination=None, random_seed=None):
        super(EdgeNetworkNavigation, self).reset(map, current_lane)
        assert self.map.road_network_type == EdgeRoadNetwork, "This Navigation module only support EdgeRoadNetwork type"

    def set_route(self, current_lane_index: str, destination: str):
        pass

    def update_localization(self, ego_vehicle):
        pass

    def _update_target_checkpoints(self, ego_lane_index, ego_lane_longitude):
        pass

    def _get_info_for_checkpoint(self, lanes_id, lanes, ego_vehicle):
        pass

    def get_current_lateral_range(self, current_position, engine) -> float:
        pass
