import queue

from metadrive.manager.base_manager import BaseManager
from metadrive.manager.scenario_data_manager import ScenarioDataManager


class ScenarioCurriculumManager(BaseManager):
    PRIORITY = ScenarioDataManager.PRIORITY - 1

    def __init__(self):
        super().__init__()
        self.recent_route_completion = queue.Queue(self.engine.global_config["episodes_to_evaluate_curriculum"])
        self.recent_success = queue.Queue(self.engine.global_config["episodes_to_evaluate_curriculum"])
        self.target_success_rate = self.engine.global_config["target_success_rate"]

    def log_episode(self, success, route_completion):
        self.recent_route_completion.put(route_completion)
        self.recent_success.put(success)

    def before_reset(self):
        """
        It should be called before reseting all managers
        """
        if sum(self.recent_success) / self.recent_success.maxsize > self.target_success_rate:
            self.engine.level_up()
            self.recent_route_completion = queue.Queue(self.engine.global_config["episodes_to_evaluate_curriculum"])
            self.recent_success = queue.Queue(self.engine.global_config["episodes_to_evaluate_curriculum"])
            self.engine.map_manager.clear_stored_maps()
            self.engine.map_manager.clear_stored_scenarios()
