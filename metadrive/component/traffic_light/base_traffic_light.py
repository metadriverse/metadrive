from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CamMask
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.constants import MetaDriveType
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.pg.utils import generate_static_box_physics_body


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
    PLACE_LONGITUDE = 5

    def __init__(
        self, lane, position=None, name=None, random_seed=None, config=None, escape_random_seed_assertion=False
    ):
        super(BaseTrafficLight, self).__init__(name, random_seed, config, escape_random_seed_assertion)
        self.lane = lane
        self.status = MetaDriveType.LIGHT_UNKNOWN

        self.lane_width = lane.width_at(0) if lane else 4
        air_wall = generate_static_box_physics_body(
            self.AIR_WALL_LENGTH,
            self.lane_width,
            self.AIR_WALL_HEIGHT,
            object_id=self.id,
            type_name=MetaDriveType.TRAFFIC_LIGHT,
            ghost_node=True,
        )
        self.add_body(air_wall, add_to_static_world=True)

        if position is None:
            # auto determining
            position = lane.position(self.PLACE_LONGITUDE, 0)

        self.set_position(position, self.AIR_WALL_HEIGHT / 2)
        self.set_heading_theta(lane.heading_theta_at(self.PLACE_LONGITUDE) if lane else 0)
        self.current_light = None

        if self.render:
            if len(BaseTrafficLight.TRAFFIC_LIGHT_MODEL) == 0:
                for color in ["green", "red", "yellow", "unknown"]:
                    model = self.loader.loadModel(
                        AssetLoader.file_path("models", "traffic_light", "{}.gltf".format(color))
                    )
                    model.setPos(0, 0, self.TRAFFIC_LIGHT_HEIGHT)
                    model.setH(-90)
                    model.hide(CamMask.Shadow)
                    BaseTrafficLight.TRAFFIC_LIGHT_MODEL[color] = model
            self.origin.setScale(0.5, 1.2, 1.2)

    def set_status(self, status):
        """
        People should overwrite this method to parse traffic light status and to determine which traffic light to set
        """
        pass

    def set_green(self):
        if self.render:
            if self.current_light is not None:
                self.current_light.detachNode()
            self.current_light = BaseTrafficLight.TRAFFIC_LIGHT_MODEL["green"].instanceTo(self.origin)
        self.status = MetaDriveType.LIGHT_GREEN

    def set_red(self):
        if self.render:
            if self.current_light is not None:
                self.current_light.detachNode()
            self.current_light = BaseTrafficLight.TRAFFIC_LIGHT_MODEL["red"].instanceTo(self.origin)
        self.status = MetaDriveType.LIGHT_RED

    def set_yellow(self):
        if self.render:
            if self.current_light is not None:
                self.current_light.detachNode()
            self.current_light = BaseTrafficLight.TRAFFIC_LIGHT_MODEL["yellow"].instanceTo(self.origin)
        self.status = MetaDriveType.LIGHT_YELLOW

    def set_unknown(self):
        if self.render:
            if self.current_light is not None:
                self.current_light.detachNode()
            self.current_light = BaseTrafficLight.TRAFFIC_LIGHT_MODEL["unknown"].instanceTo(self.origin)
        self.status = MetaDriveType.LIGHT_UNKNOWN

    def destroy(self):
        super(BaseTrafficLight, self).destroy()
        self.lane = None

    @property
    def top_down_color(self):
        status = self.status
        if status == MetaDriveType.LIGHT_GREEN:
            return [0, 255, 0]
        if status == MetaDriveType.LIGHT_RED:
            return [1, 255, 0]
        if status == MetaDriveType.LIGHT_YELLOW:
            return [255, 255, 0]
        if status == MetaDriveType.LIGHT_UNKNOWN:
            return [180, 180, 180]

    @property
    def top_down_width(self):
        return 1.5

    @property
    def top_down_length(self):
        return 1.5

    def set_action(self, *args, **kwargs):
        return self.set_status(*args, **kwargs)

    def get_state(self):
        pos = self.position
        state = {
            ScenarioDescription.TRAFFIC_LIGHT_POSITION: pos,
            ScenarioDescription.TRAFFIC_LIGHT_STATUS: self.status,
            ScenarioDescription.TYPE: type(self)
        }
        return state

    @property
    def LENGTH(self):
        return self.AIR_WALL_LENGTH

    @property
    def WIDTH(self):
        return self.lane_width
