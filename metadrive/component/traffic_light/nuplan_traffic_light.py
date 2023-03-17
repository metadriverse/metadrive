from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType


class NuplanTrafficLight(BaseTrafficLight):
    def set_status(self, status: TrafficLightStatusType):
        if status == TrafficLightStatusType.GREEN:
            self.set_green()
        elif status == TrafficLightStatusType.RED:
            self.set_red()
        elif status == TrafficLightStatusType.YELLOW:
            self.set_yellow()
        elif status == TrafficLightStatusType.UNKNOWN:
            self.set_unknown()
