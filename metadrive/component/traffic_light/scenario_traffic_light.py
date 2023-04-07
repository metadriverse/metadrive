from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.type import MetaDriveType


class ScenarioTrafficLight(BaseTrafficLight):
    def set_status(self, status, data_source="metadrive"):
        status = MetaDriveType.parse_light_status(status, simplifying=True, data_source=data_source)
        if status == MetaDriveType.LIGHT_GREEN:
            self.set_green()
        elif status == MetaDriveType.LIGHT_RED:
            self.set_red()
        elif status == MetaDriveType.LIGHT_YELLOW:
            self.set_yellow()
        elif status == MetaDriveType.LIGHT_UNKNOWN:
            self.set_unknown()
