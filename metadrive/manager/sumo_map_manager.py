from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.sumo.map_utils import StreetMap, extract_map_features


class SumoMapManager(BaseManager):
    """
    It currently only support load one map into the simulation.
    """
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self, sumo_map_path):
        super(SumoMapManager, self).__init__()
        self.current_map = None
        street_map = StreetMap()
        street_map.reset(sumo_map_path)
        self.map_feature = extract_map_features(street_map)

    def destroy(self):
        self.current_map.destroy()
        super(SumoMapManager, self).destroy()
        self.current_map = None

    def before_reset(self):
        if self.current_map:
            self.current_map.detach_from_world()

    def reset(self):
        if not self.current_map:
            self.current_map = ScenarioMap(map_index=0, map_data=self.map_feature)
        self.current_map.attach_to_world()
