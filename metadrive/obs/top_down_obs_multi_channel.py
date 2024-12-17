from collections import deque

import gymnasium as gym
import math
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.component.lane.point_lane import PointLane
from metadrive.constants import Decoration, DEFAULT_AGENT
from metadrive.obs.top_down_obs import TopDownObservation
from metadrive.obs.top_down_obs_impl import WorldSurface, COLOR_BLACK, ObjectGraphics, LaneGraphics, \
    ObservationWindowMultiChannel
from metadrive.utils import import_pygame, clip

from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation

pygame = import_pygame()
COLOR_WHITE = pygame.Color("white")
DEFAULT_TRAJECTORY_LANE_WIDTH = 3


class TopDownMultiChannel(TopDownObservation):
    """
    Most of the source code is from Highway-Env, we only optimize and integrate it in MetaDrive
    See more information on its Github page: https://github.com/eleurent/highway-env
    """
    RESOLUTION = (100, 100)  # pix x pix
    MAP_RESOLUTION = (2000, 2000)  # pix x pix
    # MAX_RANGE = (50, 50)  # maximum detection distance = 50 M

    # CHANNEL_NAMES = ["road_network", "traffic_flow", "target_vehicle", "navigation", "past_pos"]
    CHANNEL_NAMES = ["road_network", "traffic_flow", "navigation", "past_pos"]

    def __init__(
        self,
        vehicle_config,
        onscreen,
        clip_rgb: bool,
        frame_stack: int = 5,
        post_stack: int = 5,
        frame_skip: int = 5,
        resolution=None,
        max_distance=50
    ):
        super(TopDownMultiChannel, self).__init__(
            vehicle_config, clip_rgb, onscreen=onscreen, resolution=resolution, max_distance=max_distance
        )
        self.num_stacks = 2 + frame_stack
        self.stack_traffic_flow = deque([], maxlen=(frame_stack - 1) * frame_skip + 1)
        self.stack_past_pos = deque(
            [], maxlen=(post_stack - 1) * frame_skip + 1
        )  # In the coordination of target vehicle
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self._should_fill_stack = True
        self.max_distance = max_distance
        self.scaling = self.resolution[0] / max_distance
        assert self.scaling == self.resolution[1] / self.max_distance

    def init_obs_window(self):
        names = self.CHANNEL_NAMES.copy()
        names.remove("past_pos")
        self.obs_window = ObservationWindowMultiChannel(names, (self.max_distance, self.max_distance), self.resolution)

    def init_canvas(self):
        self.canvas_background = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_navigation = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_road_network = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_runtime = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_ego = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_past_pos = pygame.Surface(self.resolution)  # A local view

    def reset(self, env, vehicle=None):
        # self.engine = env.engine
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

        if isinstance(self.target_vehicle.navigation, NodeNetworkNavigation):
            self.draw_navigation_node(self.canvas_background, (64, 64, 64))
        elif isinstance(self.target_vehicle.navigation, EdgeNetworkNavigation):
            # TODO: draw edge network navigation
            pass
        elif isinstance(self.target_vehicle.navigation, TrajectoryNavigation):
            self.draw_navigation_trajectory(self.canvas_background, (64, 64, 64))

        if isinstance(self.road_network, NodeRoadNetwork):
            for _from in self.road_network.graph.keys():
                decoration = True if _from == Decoration.start else False
                for _to in self.road_network.graph[_from].keys():
                    for l in self.road_network.graph[_from][_to]:
                        two_side = True if l is self.road_network.graph[_from][_to][-1] or decoration else False
                        LaneGraphics.LANE_LINE_WIDTH = 0.5
                        LaneGraphics.display(l, self.canvas_background, two_side)
        elif hasattr(self.engine, "map_manager"):
            for data in self.engine.map_manager.current_map.blocks[-1].map_data.values():
                if ScenarioDescription.POLYLINE in data:
                    LaneGraphics.display_scenario_line(
                        data[ScenarioDescription.POLYLINE], data[ScenarioDescription.TYPE], self.canvas_background
                    )

        self.canvas_road_network.blit(self.canvas_background, (0, 0))
        self.obs_window.reset(self.canvas_runtime)
        self._should_draw_map = False

    def _refresh(self, canvas, pos, clip_size):
        canvas.set_clip((pos[0] - clip_size[0] / 2, pos[1] - clip_size[1] / 2, clip_size[0], clip_size[1]))
        canvas.fill(COLOR_BLACK)

    def draw_scene(self):
        # Set the active area that can be modify to accelerate
        assert len(self.engine.agents) == 1, "Don't support multi-agent top-down observation yet!"
        vehicle = self.engine.agents[DEFAULT_AGENT]
        pos = self.canvas_runtime.pos2pix(*vehicle.position)

        clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))

        # self._refresh(self.canvas_ego, pos, clip_size)
        self._refresh(self.canvas_runtime, pos, clip_size)
        self.canvas_past_pos.fill(COLOR_BLACK)
        # self._draw_ego_vehicle()

        # Draw vehicles
        # TODO PZH: I hate computing these in pygame-related code!!!
        ego_heading = vehicle.heading_theta
        ego_heading = ego_heading if abs(ego_heading) > 2 * np.pi / 180 else 0

        for v in self.engine.get_objects(lambda o: isinstance(o, BaseVehicle) or isinstance(o, BaseTrafficParticipant)
                                         ).values():
            if v is vehicle:
                continue
            h = v.heading_theta
            h = h if abs(h) > 2 * np.pi / 180 else 0
            ObjectGraphics.display(object=v, surface=self.canvas_runtime, heading=h, color=ObjectGraphics.BLUE)

        raw_pos = vehicle.position
        self.stack_past_pos.append(raw_pos)
        for p_index in self._get_stack_indices(len(self.stack_past_pos)):
            p_old = self.stack_past_pos[p_index]
            diff = p_old - raw_pos
            diff = (diff[0] * self.scaling, diff[1] * self.scaling)
            # p = (p_old[0] - pos[0], p_old[1] - pos[1])
            diff = (diff[1], diff[0])
            p = pygame.math.Vector2(tuple(diff))
            # p = pygame.math.Vector2(p)
            p = p.rotate(np.rad2deg(ego_heading) + 90)
            p = (p[1], p[0])
            p = (
                clip(p[0] + self.resolution[0] / 2, -self.resolution[0],
                     self.resolution[0]), clip(p[1] + self.resolution[1] / 2, -self.resolution[1], self.resolution[1])
            )
            # p = self.canvas_background.pos2pix(p[0], p[1])
            self.canvas_past_pos.fill((255, 255, 255), (p, (1, 1)))
            # pygame.draw.circle(self.canvas_past_pos, (255, 255, 255), p, radius=1)

        ret = self.obs_window.render(
            canvas_dict=dict(
                road_network=self.canvas_road_network,
                traffic_flow=self.canvas_runtime,
                target_vehicle=self.canvas_ego,
                # navigation=self.canvas_navigation,
            ),
            position=pos,
            heading=vehicle.heading_theta
        )
        ret["past_pos"] = self.canvas_past_pos
        return ret

    def _draw_ego_vehicle(self):
        vehicle = self.engine.agents[DEFAULT_AGENT]
        w = vehicle.top_down_width * self.scaling
        h = vehicle.top_down_length * self.scaling
        position = (self.resolution[0] / 2, self.resolution[1] / 2)
        angle = 90
        box = [pygame.math.Vector2(p) for p in [(-h / 2, -w / 2), (-h / 2, w / 2), (h / 2, w / 2), (h / 2, -w / 2)]]
        box_rotate = [p.rotate(angle) + position for p in box]
        pygame.draw.polygon(self.canvas_past_pos, color=(128, 128, 128), points=box_rotate)

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

        if self.norm_pixel:
            img = img.astype(np.float32) / 255
        else:
            img = img.astype(np.uint8)
        return img

    def observe(self, vehicle: BaseVehicle):
        self.render()
        surface_dict = self.get_observation_window()
        surface_dict["road_network"] = pygame.transform.smoothscale(surface_dict["road_network"], self.resolution)
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

        img = [
            img_dict["road_network"] * 2,
            # img_navigation,
            # img_dict["navigation"],
            # img_dict["target_vehicle"],
            img_dict["past_pos"],
        ]  # + list(self.stack_traffic_flow)

        # Stacked traffic flow
        # stacked = np.zeros_like(img_navigation)
        indices = self._get_stack_indices(len(self.stack_traffic_flow))
        # for i in reversed(indices):
        #     stacked = self.stack_traffic_flow[i] + stacked / 2
        # if self.norm_pixel:
        #     stacked = np.clip(stacked, 0.0, 1.0)
        # else:
        #     stacked = np.clip(stacked, 0, 255)
        for i in indices:
            img.append(self.stack_traffic_flow[i])

        # Stack
        img = np.stack(img, axis=2)
        if self.norm_pixel:
            img = np.clip(img, 0, 1.0)
        else:
            img = np.clip(img, 0, 255)
        return np.transpose(img, (1, 0, 2))

    def draw_navigation_node(self, canvas, color=(128, 128, 128)):
        checkpoints = self.target_vehicle.navigation.checkpoints
        for i, c in enumerate(checkpoints[:-1]):
            lanes = self.road_network.graph[c][checkpoints[i + 1]]
            for lane in lanes:
                LaneGraphics.draw_drivable_area(lane, canvas, color=color)

    def draw_navigation_trajectory(self, canvas, color=(128, 128, 128)):
        lane = PointLane(self.target_vehicle.navigation.checkpoints, DEFAULT_TRAJECTORY_LANE_WIDTH)
        LaneGraphics.draw_drivable_area(lane, canvas, color=color)

    def _get_stack_indices(self, length, frame_skip=None):
        frame_skip = frame_skip or self.frame_skip
        num = int(math.ceil(length / frame_skip))
        indices = []
        for i in range(num):
            indices.append(length - 1 - i * frame_skip)
        return indices

    @property
    def observation_space(self):
        shape = self.obs_shape + (self.num_stacks, )
        if self.norm_pixel:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
