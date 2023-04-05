"""
Similar to top_down_obs_multi_channel, this file defines a new BEV observation that resembles NuScene dataset.

Credit: Alex Swerdlow
"""
import gym
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import Decoration, DEFAULT_AGENT
from metadrive.obs.top_down_obs import TopDownObservation
from metadrive.obs.top_down_obs_impl import WorldSurface, COLOR_BLACK, VehicleGraphics, LaneGraphics, \
    ObservationWindowMultiChannel, ObservationWindow
from metadrive.utils import import_pygame

pygame = import_pygame()
COLOR_WHITE = pygame.Color("white")
CHANNEL_NAMES = ["driveable_area", "lane_lines", "actors"]


class ObservationWindowNuScenes(ObservationWindowMultiChannel):
    CHANNEL_NAMES = CHANNEL_NAMES

    def __init__(self, names, max_range, resolution):
        """Overwrite the parent class."""
        assert isinstance(names, list)
        assert set(self.CHANNEL_NAMES)
        self.sub_observations = {
            k: ObservationWindow(max_range=max_range, resolution=resolution)
            for k in ["driveable_area", "lane_lines", "actors"]
        }
        self.resolution = (resolution[0] * 2, resolution[1] * 2)
        self.canvas_display = None

    def get_screen_window(self):
        canvas = self.get_canvas_display()
        ret = self.get_observation_window()

        # for k in ret.keys():
        #     if k == "road_network":
        #         continue
        #     ret[k] = pygame.transform.scale2x(ret[k])

        def _draw(canvas, key, color):
            mask = pygame.mask.from_threshold(ret[key], (0, 0, 0, 0), (10, 10, 10, 255))
            mask.to_surface(canvas, setcolor=None, unsetcolor=color)

        # if "navigation" in ret:
        #     _draw(canvas, "navigation", pygame.Color("Blue"))
        _draw(canvas, "driveable_area", pygame.Color("White"))
        _draw(canvas, "lane_lines", pygame.Color("Red"))
        _draw(canvas, "actors", pygame.Color("Green"))
        return canvas


class TopDownNuScenes(TopDownObservation):
    RESOLUTION = (100, 100)  # pix x pix
    MAP_RESOLUTION = (2000, 2000)  # pix x pix

    CHANNEL_NAMES = CHANNEL_NAMES

    def __init__(
            self,
            vehicle_config,
            onscreen,
            clip_rgb,
            resolution,
            max_distance
    ):
        super(TopDownNuScenes, self).__init__(
            vehicle_config, clip_rgb, onscreen=onscreen, resolution=resolution, max_distance=max_distance
        )
        self.max_distance = max_distance
        self.scaling = self.resolution[0] / max_distance
        assert self.scaling == self.resolution[1] / self.max_distance

    def init_obs_window(self):
        names = self.CHANNEL_NAMES.copy()
        self.obs_window = ObservationWindowNuScenes(names, (self.max_distance, self.max_distance), self.resolution)

    def init_canvas(self):
        self.canvas_actors = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_driveable_area = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_lane_lines = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))

    def reset(self, env, vehicle=None):
        self.road_network = env.current_map.road_network
        self.target_vehicle = vehicle
        self._should_draw_map = True
        self._should_fill_stack = True

    def draw_map(self) -> pygame.Surface:
        """
        :return: a big map surface, clip  and rotate to use a piece of it
        """
        # Setup the maximize size of the canvas
        # scaling and center can be easily found by bounding box
        b_box = self.road_network.get_bounding_box()
        self.canvas_driveable_area.fill(COLOR_BLACK)
        self.canvas_lane_lines.fill(COLOR_BLACK)
        self.canvas_actors.fill(COLOR_BLACK)
        x_len = b_box[1] - b_box[0]
        y_len = b_box[3] - b_box[2]
        max_len = max(x_len, y_len) + 20  # Add more 20 meters
        scaling = self.MAP_RESOLUTION[1] / max_len - 0.1
        assert scaling > 0

        # real-world distance * scaling = pixel in canvas
        self.canvas_driveable_area.scaling = scaling
        self.canvas_lane_lines.scaling = scaling
        self.canvas_actors.scaling = scaling

        centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
        self.canvas_driveable_area.move_display_window_to(centering_pos)
        self.canvas_lane_lines.move_display_window_to(centering_pos)
        self.canvas_actors.move_display_window_to(centering_pos)

        for _from in self.road_network.graph.keys():
            decoration = True if _from == Decoration.start else False
            for _to in self.road_network.graph[_from].keys():
                for l in self.road_network.graph[_from][_to]:
                    # two_side = True if l is self.road_network.graph[_from][_to][-1] or decoration else False
                    l.line_types = list(map(lambda x: 'continuous' if x == 'broken' else x, l.line_types))
                    LaneGraphics.LANE_LINE_WIDTH = 0.5
                    LaneGraphics.display(l, self.canvas_lane_lines, False, color=(255, 255, 255))
                    LaneGraphics.draw_drivable_area(l, self.canvas_driveable_area, color=(255, 255, 255))

        self.obs_window.reset(self.canvas_actors)
        self._should_draw_map = False

    def _refresh(self, canvas, pos, clip_size):
        canvas.set_clip((pos[0] - clip_size[0] / 2, pos[1] - clip_size[1] / 2, clip_size[0], clip_size[1]))
        canvas.fill(COLOR_BLACK)

    def draw_scene(self):
        # Set the active area that can be modify to accelerate
        assert len(self.engine.agents) == 1, "Don't support multi-agent top-down observation yet!"
        vehicle = self.engine.agents[DEFAULT_AGENT]
        pos = self.canvas_actors.pos2pix(*vehicle.position)

        clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))

        self._refresh(self.canvas_actors, pos, clip_size)

        # Draw vehicles
        # TODO PZH: I hate computing these in pygame-related code!!!
        ego_heading = vehicle.heading_theta
        ego_heading = ego_heading if abs(ego_heading) > 2 * np.pi / 180 else 0

        for v in self.engine.traffic_manager.vehicles:
            # if v is vehicle:
            #     continue
            h = v.heading_theta
            h = h if abs(h) > 2 * np.pi / 180 else 0
            VehicleGraphics.display(vehicle=v, surface=self.canvas_actors, heading=h, color=(255, 255, 255))

        self.obs_window.render(
            canvas_dict=dict(
                driveable_area=self.canvas_driveable_area,
                lane_lines=self.canvas_lane_lines,
                actors=self.canvas_actors
            ),
            position=pos,
            heading=vehicle.heading_theta
        )

    def _transform(self, img):
        # img = np.mean(img, axis=-1)
        # Use Atari-like processing

        # img = img[..., 0]
        # img = np.dot(img[..., :], [0.299, 0.587, 0.114])

        assert (img == img[..., 0, np.newaxis]).all()
        img = img[..., 0]

        if self.rgb_clip:
            img = img.astype(np.float32) / 255
        else:
            img = img.astype(np.uint8)
        return img

    def observe(self, vehicle: BaseVehicle):
        self.render()
        surface_dict = self.get_observation_window()
        surface_dict["driveable_area"] = pygame.transform.smoothscale(surface_dict["driveable_area"], self.resolution)
        surface_dict["lane_lines"] = pygame.transform.smoothscale(surface_dict["lane_lines"], self.resolution)
        surface_dict["actors"] = pygame.transform.smoothscale(surface_dict["actors"], self.resolution)
        img_dict = {k: pygame.surfarray.array3d(surface) for k, surface in surface_dict.items()}

        # Gray scale
        img_dict = {k: self._transform(img) for k, img in img_dict.items()}

        img = [
            img_dict["driveable_area"],
            img_dict["lane_lines"],
            img_dict["actors"]
        ]

        # Stack
        img = np.stack(img, axis=2)
        if self.rgb_clip:
            img = np.clip(img, 0, 1.0)
        else:
            img = np.clip(img, 0, 255)
        return np.transpose(img, (1, 0, 2))

    @property
    def observation_space(self):
        shape = self.obs_shape + (self.num_stacks,)
        if self.rgb_clip:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
