import math
from collections import deque

import cv2
import gym
import numpy as np
from pgdrive.constants import Decoration, DEFAULT_AGENT
from pgdrive.obs.top_down_obs_impl import WorldSurface, COLOR_BLACK, VehicleGraphics, LaneGraphics, \
    ObservationWindowMultiChannel
from pgdrive.obs.top_down_obs import TopDownObservation
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.utils import import_pygame

pygame = import_pygame()
COLOR_WHITE = pygame.Color("white")


class TopDownMultiChannel(TopDownObservation):
    """
    Most of the source code is from Highway-Env, we only optimize and integrate it in PGDrive
    See more information on its Github page: https://github.com/eleurent/highway-env
    """
    RESOLUTION = (100, 100)  # pix x pix
    MAP_RESOLUTION = (2000, 2000)  # pix x pix
    MAX_RANGE = (50, 50)  # maximum detection distance = 50 M

    CHANNEL_NAMES = ["road_network", "traffic_flow", "target_vehicle", "navigation", "past_pos"]

    def __init__(
        self,
        vehicle_config,
        env,
        clip_rgb: bool,
        frame_stack: int = 5,
        post_stack: int = 5,
        frame_skip: int = 5,
        resolution=None
    ):
        super(TopDownMultiChannel, self).__init__(vehicle_config, env, clip_rgb, resolution=resolution)
        self.num_stacks = 4 + frame_stack
        self.stack_traffic_flow = deque([], maxlen=(frame_stack - 1) * frame_skip + 1)
        self.stack_past_pos = deque(
            [], maxlen=(post_stack - 1) * frame_skip + 1
        )  # In the coordination of target vehicle
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self._should_fill_stack = True

    def init_obs_window(self):
        names = self.CHANNEL_NAMES.copy()
        names.remove("past_pos")
        self.obs_window = ObservationWindowMultiChannel(names, self.MAX_RANGE, self.resolution)

    def init_canvas(self):
        self.canvas_background = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_navigation = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_road_network = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_runtime = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_ego = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_past_pos = pygame.Surface(self.resolution)  # A local view

    def reset(self, env, vehicle=None):
        self.scene_manager = env.scene_manager
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
        self.canvas_navigation.fill(COLOR_BLACK)
        self.canvas_ego.fill(COLOR_BLACK)
        self.canvas_road_network.fill(COLOR_BLACK)
        self.canvas_runtime.fill(COLOR_BLACK)
        self.canvas_background.fill(COLOR_BLACK)
        self.canvas_background.set_colorkey(self.canvas_background.BLACK)
        x_len = b_box[1] - b_box[0]
        y_len = b_box[3] - b_box[2]
        max_len = max(x_len, y_len) + 20  # Add more 20 meters
        scaling = self.MAP_RESOLUTION[1] / max_len - 0.1
        assert scaling > 0

        # real-world distance * scaling = pixel in canvas
        self.canvas_background.scaling = scaling
        self.canvas_runtime.scaling = scaling
        self.canvas_navigation.scaling = scaling
        self.canvas_ego.scaling = scaling
        self.canvas_road_network.scaling = scaling

        centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
        self.canvas_runtime.move_display_window_to(centering_pos)
        self.canvas_navigation.move_display_window_to(centering_pos)
        self.canvas_ego.move_display_window_to(centering_pos)
        self.canvas_background.move_display_window_to(centering_pos)
        self.canvas_road_network.move_display_window_to(centering_pos)

        for _from in self.road_network.graph.keys():
            decoration = True if _from == Decoration.start else False
            for _to in self.road_network.graph[_from].keys():
                for l in self.road_network.graph[_from][_to]:
                    two_side = True if l is self.road_network.graph[_from][_to][-1] or decoration else False
                    LaneGraphics.LANE_LINE_WIDTH = 0.5
                    LaneGraphics.display(l, self.canvas_background, two_side)
        self.canvas_road_network.blit(self.canvas_background, (0, 0))
        self.obs_window.reset(self.canvas_runtime)

        self._should_draw_map = False

        self.draw_navigation()

    def _refresh(self, canvas, pos, clip_size):
        canvas.set_clip((pos[0] - clip_size[0] / 2, pos[1] - clip_size[1] / 2, clip_size[0], clip_size[1]))
        canvas.fill(COLOR_BLACK)

    def draw_scene(self):
        # Set the active area that can be modify to accelerate
        assert len(self.scene_manager.target_vehicles) == 1, "Don't support multi-agent top-down observation yet!"
        vehicle = self.scene_manager.target_vehicles[DEFAULT_AGENT]
        pos = self.canvas_runtime.pos2pix(*vehicle.position)

        clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))

        self._refresh(self.canvas_ego, pos, clip_size)
        self._refresh(self.canvas_runtime, pos, clip_size)
        self.canvas_past_pos.fill(COLOR_BLACK)

        # Draw vehicles
        # TODO PZH: I hate computing these in pygame-related code!!!
        ego_heading = vehicle.heading_theta
        ego_heading = ego_heading if abs(ego_heading) > 2 * np.pi / 180 else 0

        VehicleGraphics.display(
            vehicle=vehicle,
            surface=self.canvas_ego,  # Draw target vehicle in this canvas!
            heading=ego_heading,
            color=VehicleGraphics.GREEN
        )
        for v in self.scene_manager.traffic_mgr.vehicles:
            if v is vehicle:
                continue
            h = v.heading
            h = h if abs(h) > 2 * np.pi / 180 else 0
            VehicleGraphics.display(vehicle=v, surface=self.canvas_runtime, heading=h, color=VehicleGraphics.BLUE)

        pos = self.canvas_runtime.pos2pix(*vehicle.position)
        self.stack_past_pos.append(pos)
        for p in self._get_stack_indices(len(self.stack_past_pos)):
            p = self.stack_past_pos[p]
            # TODO PZH: Could you help me check this part? I just engineering out the equation. Not sure if correct!@LQY
            p = (p[0] - pos[0], p[1] - pos[1])
            p = (p[1], p[0])
            p = pygame.math.Vector2(p)
            p = p.rotate(np.rad2deg(ego_heading) + 90)
            p = (p[1], p[0])
            p = (p[0] + self.resolution[0] / 2, p[1] + self.resolution[1] / 2)
            pygame.draw.circle(self.canvas_past_pos, color=COLOR_WHITE, radius=1, center=p)

        ret = self.obs_window.render(
            canvas_dict=dict(
                road_network=self.canvas_road_network,  # TODO
                traffic_flow=self.canvas_runtime,
                target_vehicle=self.canvas_ego,  # TODO
                navigation=self.canvas_navigation,
            ),
            position=pos,
            heading=vehicle.heading_theta
        )
        ret["past_pos"] = self.canvas_past_pos
        return ret

    def get_observation_window(self):
        ret = self.obs_window.get_observation_window()
        ret["past_pos"] = self.canvas_past_pos
        return ret

    def _transform(self, img):
        # img = np.mean(img, axis=-1)
        # Use Atari-like processing

        # img = img[..., 0]
        # img = np.dot(img[..., :], [0.299, 0.587, 0.114])
        img = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114

        if self.rgb_clip:
            img = img.astype(np.float32) / 255
        else:
            img = img.astype(np.uint8)
        return img

    def observe(self, vehicle: BaseVehicle):
        self.render()
        surface_dict = self.get_observation_window()
        img_dict = {k: pygame.surfarray.array3d(surface) for k, surface in surface_dict.items()}

        # Gray scale
        img_dict = {k: self._transform(img) for k, img in img_dict.items()}

        if self._should_fill_stack:
            self.stack_past_pos.clear()
            self.stack_traffic_flow.clear()
            for _ in range(self.stack_traffic_flow.maxlen):
                self.stack_traffic_flow.append(img_dict["traffic_flow"])
            self._should_fill_stack = False
        self.stack_traffic_flow.append(img_dict["traffic_flow"])

        # Reorder
        img_road_network = img_dict["road_network"]
        img_road_network = cv2.resize(img_road_network, self.resolution, interpolation=cv2.INTER_LINEAR)
        img = [
            img_road_network,
            img_dict["navigation"],
            img_dict["target_vehicle"],
            img_dict["past_pos"],
        ]  # + list(self.stack_traffic_flow)

        for i in self._get_stack_indices(len(self.stack_traffic_flow)):
            img.append(self.stack_traffic_flow[i])

        # Stack
        img = np.stack(img, axis=2)
        return np.transpose(img, (1, 0, 2))

    def draw_navigation(self):
        checkpoints = self.target_vehicle.routing_localization.checkpoints
        for i, c in enumerate(checkpoints[:-1]):
            lanes = self.road_network.graph[c][checkpoints[i + 1]]
            for lane in lanes:
                LaneGraphics.simple_draw(lane, self.canvas_navigation, color=(255, 255, 255))

    def _get_stack_indices(self, length):
        num = int(math.ceil(length / self.frame_skip))
        indices = []
        for i in range(num):
            indices.append(length - 1 - i * self.frame_skip)
        return indices

    @property
    def observation_space(self):
        shape = self.obs_shape + (self.num_stacks, )
        if self.rgb_clip:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
