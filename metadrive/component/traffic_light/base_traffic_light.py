from metadrive.base_class.base_object import BaseObject
from metadrive.constants import TrafficLightStatus, BodyName
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.scene_utils import generate_static_box_physics_body


class BaseTrafficLight(BaseObject):
    """
    Traffic light should be associated with a lane before using. It is basically an unseen wall object on the route, so
    actors have to react to it.
    """
    AIR_WALL_LENGTH = 0.25
    AIR_WALL_HEIGHT = 1.5
    TRAFFIC_LIGHT_HEIGHT = 3
    TRAFFIC_LIGHT_MODEL = {}
    LIGHT_VIS_HEIGHT = 0.8
    LIGHT_VIS_WIDTH = 0.8

    def __init__(self, lane, name=None, random_seed=None, config=None, escape_random_seed_assertion=False):
        super(BaseTrafficLight, self).__init__(name, random_seed, config, escape_random_seed_assertion)
        self.lane = lane
        self.status = TrafficLightStatus.UNKNOWN

        width = lane.width_at(0)
        air_wall = generate_static_box_physics_body(
            self.AIR_WALL_LENGTH,
            width,
            self.AIR_WALL_HEIGHT,
            object_id=self.id,
            type_name=BodyName.TrafficLight,
            ghost_node=True,
        )
        self.add_body(air_wall, add_to_static_world=True)

        self.set_position(lane.position(0, 0), self.AIR_WALL_HEIGHT / 2)
        self.set_heading_theta(lane.heading_theta_at(0))

        if self.render:
            if len(BaseTrafficLight.TRAFFIC_LIGHT_MODEL) == 0:
                for color in ["green", "red", "yellow", "unknown"]:
                    model = self.loader.loadModel(
                        AssetLoader.file_path("models", "traffic_light", "{}.gltf".format(color))
                    )
                    model.setPos(0, 0, self.TRAFFIC_LIGHT_HEIGHT)
                    model.setH(-90)
                    BaseTrafficLight.TRAFFIC_LIGHT_MODEL[color] = model
            self.origin.setScale(0.5, 1.2, 1.2)

    def set_green(self):
        if self.render:
            BaseTrafficLight.TRAFFIC_LIGHT_MODEL["green"].instanceTo(self.origin)
        self.status = TrafficLightStatus.GREEN

    def set_red(self):
        if self.render:
            BaseTrafficLight.TRAFFIC_LIGHT_MODEL["red"].instanceTo(self.origin)
        self.status = TrafficLightStatus.RED

    def set_yellow(self):
        if self.render:
            BaseTrafficLight.TRAFFIC_LIGHT_MODEL["yellow"].instanceTo(self.origin)
        self.status = TrafficLightStatus.YELLOW

    def set_unknown(self):
        if self.render:
            BaseTrafficLight.TRAFFIC_LIGHT_MODEL["known"].instanceTo(self.origin)
        self.status = TrafficLightStatus.UNKNOWN

    def destroy(self):
        super(BaseTrafficLight, self).destroy()
        self.lane = None
