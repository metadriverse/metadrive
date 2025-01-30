import numpy as np

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CamMask, CollisionGroup
from metadrive.constants import MetaDriveType, Semantics
from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.pg.utils import generate_static_box_physics_body


class BaseTrafficLight(BaseObject):
    """
    Traffic light should be associated with a lane before using. It is basically an unseen wall object on the route, so
    actors have to react to it.
    """
    SEMANTIC_LABEL = Semantics.TRAFFIC_LIGHT.label
    AIR_WALL_LENGTH = 0.25
    AIR_WALL_HEIGHT = 1.5
    TRAFFIC_LIGHT_HEIGHT = 3
    TRAFFIC_LIGHT_MODEL = {}
    LIGHT_VIS_HEIGHT = 0.8
    LIGHT_VIS_WIDTH = 0.8
    PLACE_LONGITUDE = 5

    def __init__(
        self,
        lane,
        position=None,
        name=None,
        random_seed=None,
        config=None,
        escape_random_seed_assertion=False,
        draw_line=False,
        show_model=True,
    ):
        super(BaseTrafficLight, self).__init__(name, random_seed, config, escape_random_seed_assertion)
        self.set_metadrive_type(MetaDriveType.TRAFFIC_LIGHT)
        self.lane = lane
        self.status = MetaDriveType.LIGHT_UNKNOWN
        self._draw_line = draw_line
        self._show_model = show_model
        self._lane_center_line = None

        self.lane_width = lane.width_at(0) if lane else 4
        air_wall = generate_static_box_physics_body(
            self.AIR_WALL_LENGTH,
            self.lane_width,
            self.AIR_WALL_HEIGHT,
            object_id=self.id,
            type_name=MetaDriveType.TRAFFIC_LIGHT,
            ghost_node=True,
        )
        self.add_body(air_wall, add_to_static_world=False)  # add to dynamic world so the lidar can detect it

        if position is None:
            # auto determining
            position = lane.position(self.PLACE_LONGITUDE, 0)

        self.set_position(position, self.AIR_WALL_HEIGHT / 2)
        self.set_heading_theta(lane.heading_theta_at(self.PLACE_LONGITUDE) if lane else 0)
        self.current_light = None

        if self.render:
            if len(BaseTrafficLight.TRAFFIC_LIGHT_MODEL) == 0 and self._show_model:
                for color in ["green", "red", "yellow", "unknown"]:
                    model = self.loader.loadModel(
                        AssetLoader.file_path("models", "traffic_light", "{}.gltf".format(color))
                    )
                    model.setPos(0, 0, self.TRAFFIC_LIGHT_HEIGHT)
                    model.setH(-90)
                    model.hide(CamMask.Shadow)
                    BaseTrafficLight.TRAFFIC_LIGHT_MODEL[color] = model
            self.origin.setScale(0.5, 1.2, 1.2)
            if self._draw_line:
                self._line_drawer = self.engine.make_line_drawer(thickness=2)
                self._lane_center_line = np.array([[p[0], p[1], 0.4] for p in self.lane.get_polyline()])

    def before_step(self, *args, **kwargs):
        self.set_status(*args, **kwargs)

    def set_status(self, status):
        """
        People should overwrite this method to parse traffic light status and to determine which traffic light to set
        """
        pass

    def _try_draw_line(self, color):
        if self._draw_line:
            self._line_drawer.reset()
            self._line_drawer.draw_lines([self._lane_center_line], [[color for _ in self._lane_center_line]])

    def set_green(self):
        if self.render:
            if self.current_light is not None:
                self.current_light.detachNode()
            if self._show_model:
                self.current_light = BaseTrafficLight.TRAFFIC_LIGHT_MODEL["green"].instanceTo(self.origin)
            self._try_draw_line([3 / 255, 255 / 255, 3 / 255])
        self.status = MetaDriveType.LIGHT_GREEN
        self._body.setIntoCollideMask(CollisionGroup.AllOff)  # can not be detected by anything

    def set_red(self):
        if self.render:
            if self.current_light is not None:
                self.current_light.detachNode()
            if self._show_model:
                self.current_light = BaseTrafficLight.TRAFFIC_LIGHT_MODEL["red"].instanceTo(self.origin)
            self._try_draw_line([252 / 255, 0 / 255, 0 / 255])
        self.status = MetaDriveType.LIGHT_RED
        self._body.setIntoCollideMask(CollisionGroup.InvisibleWall)  # will be detected by lidar and object detector

    def set_yellow(self):
        if self.render:
            if self.current_light is not None:
                self.current_light.detachNode()
            if self._show_model:
                self.current_light = BaseTrafficLight.TRAFFIC_LIGHT_MODEL["yellow"].instanceTo(self.origin)
            self._try_draw_line([252 / 255, 227 / 255, 3 / 255])
        self.status = MetaDriveType.LIGHT_YELLOW
        self._body.setIntoCollideMask(CollisionGroup.InvisibleWall)  # will be detected by lidar and object detector

    def set_unknown(self):
        if self.render:
            if self.current_light is not None:
                self.current_light.detachNode()
            if self._show_model:
                self.current_light = BaseTrafficLight.TRAFFIC_LIGHT_MODEL["unknown"].instanceTo(self.origin)
        self.status = MetaDriveType.LIGHT_UNKNOWN
        self._body.setIntoCollideMask(CollisionGroup.AllOff)  # can not be detected by anything

    def destroy(self):
        super(BaseTrafficLight, self).destroy()
        self.lane = None
        if self._draw_line:
            self._line_drawer.reset()
            self._line_drawer.removeNode()

    @property
    def top_down_color(self):
        status = self.status
        if status == MetaDriveType.LIGHT_GREEN:
            return [0, 255, 0]
        if status == MetaDriveType.LIGHT_RED:
            return [255, 0, 0]
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
