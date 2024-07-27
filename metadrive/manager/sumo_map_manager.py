from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.sumo.map_utils import extract_map_features, RoadLaneJunctionGraph


class SumoMapManager(BaseManager):
    """
    It currently only support load one map into the simulation.
    """
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self, sumo_map_path):
        """
        Init the map manager. It can be extended to manage more maps
        """
        super(SumoMapManager, self).__init__()
        self.current_map = None
        self.graph = RoadLaneJunctionGraph(sumo_map_path)
        self.map_feature = extract_map_features(self.graph)

    def destroy(self):
        """
        Delete the map manager
        """
        self.current_map.destroy()
        super(SumoMapManager, self).destroy()
        self.current_map = None

    def before_reset(self):
        """
        Detach existing maps
        """
        if self.current_map:
            self.current_map.detach_from_world()

    def reset(self):
        """
        Rebuild the map and load it into the scene
        """
        if not self.current_map:
            self.current_map = ScenarioMap(map_index=0, map_data=self.map_feature, need_lane_localization=True)
        self.current_map.attach_to_world()
