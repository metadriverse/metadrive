from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight


class ScenarioTrafficLight(BaseTrafficLight):
    def set_status(self, status):
        self.set_green()
        # if status == TrafficLightStatusType.GREEN:
        #     self.set_green()
        # elif status == TrafficLightStatusType.RED:
        #     self.set_red()
        # elif status == TrafficLightStatusType.YELLOW:
        #     self.set_yellow()
        # elif status == TrafficLightStatusType.UNKNOWN:
        #     self.set_unknown()
