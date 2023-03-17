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
    TRAFFIC_LIGHT_HEIGHT = 3.5
    TRAFFIC_LIGHT_MODEL = None
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
            if BaseTrafficLight.TRAFFIC_LIGHT_MODEL is None:
                BaseTrafficLight.TRAFFIC_LIGHT_MODEL = self.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                BaseTrafficLight.TRAFFIC_LIGHT_MODEL.setPos(0, 0, self.TRAFFIC_LIGHT_HEIGHT)
            BaseTrafficLight.TRAFFIC_LIGHT_MODEL.instanceTo(self.origin)
            self.origin.setScale(self.AIR_WALL_LENGTH / 2, self.LIGHT_VIS_WIDTH, self.LIGHT_VIS_HEIGHT)

    def set_green(self):
        self.origin.setColor(0 / 255, 255 / 255, 0, 1)
        self.status = TrafficLightStatus.GREEN

    def set_red(self):
        self.origin.setColor(255 / 255, 0, 0)
        self.status = TrafficLightStatus.RED

    def set_yellow(self):
        self.origin.setColor(252 / 255, 244 / 255, 3 / 255)
        self.status = TrafficLightStatus.YELLOW

    def destroy(self):
        super(BaseTrafficLight, self).destroy()
        self.lane = None
