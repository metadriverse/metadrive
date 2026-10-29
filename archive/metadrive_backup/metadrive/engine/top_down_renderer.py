import copy
from metadrive.engine.logger import get_logger

from metadrive.utils.doc_utils import generate_gif
import math
from collections import deque
from typing import Optional, Union, Iterable

import numpy as np

from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.constants import Decoration, TARGET_VEHICLES
from metadrive.constants import TopDownSemanticColor, MetaDriveType, PGDrivableAreaProperty
from metadrive.obs.top_down_obs_impl import WorldSurface, ObjectGraphics, LaneGraphics, history_object
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.utils import import_pygame
from metadrive.utils.utils import is_map_related_instance

pygame = import_pygame()

color_white = (255, 255, 255)


def draw_top_down_map_native(
    map,
    semantic_map=True,
    draw_center_line=False,
    return_surface=False,
    film_size=(2000, 2000),
    scaling=None,
    semantic_broken_line=True
) -> Optional[Union[np.ndarray, pygame.Surface]]:
    """
    Draw the top_down map on a pygame surface
    Args:
        map: MetaDrive.BaseMap instance
        semantic_map: return semantic map
        draw_center_line: Draw the center line of the lane
        return_surface: Return the pygame.Surface in fime_size instead of cv2.image
        film_size: The size of the film to draw the map
        scaling: the scaling factor, how many pixels per meter
        semantic_broken_line: Draw broken line on semantic map

    Returns: cv2.image or pygame.Surface

    """
    surface = WorldSurface(film_size, 0, pygame.Surface(film_size))
    if map is None:
        surface.move_display_window_to([0, 0])
        surface.fill([230, 230, 230])
        return surface if return_surface else WorldSurface.to_cv2_image(surface)

    b_box = map.road_network.get_bounding_box()
    x_len = b_box[1] - b_box[0]
    y_len = b_box[3] - b_box[2]
    max_len = max(x_len, y_len)
    # scaling and center can be easily found by bounding box
    if scaling is None:
        scaling = (film_size[1] / max_len - 0.1)
    else:
        scaling = min(scaling, (film_size[1] / max_len - 0.1))
    surface.scaling = scaling
    centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
    surface.move_display_window_to(centering_pos)
    line_sample_interval = 2

    if semantic_map:
        all_lanes = map.get_map_features(line_sample_interval)

        for obj in all_lanes.values():
            if MetaDriveType.is_lane(obj["type"]) and not draw_center_line:
                pygame.draw.polygon(
                    surface, TopDownSemanticColor.get_color(obj["type"]),
                    [surface.pos2pix(p[0], p[1]) for p in obj["polygon"]]
                )

            elif (MetaDriveType.is_road_line(obj["type"]) or MetaDriveType.is_road_boundary_line(obj["type"])
                  or (MetaDriveType.is_lane(obj["type"]) and draw_center_line)):
                if semantic_broken_line and MetaDriveType.is_broken_line(obj["type"]):
                    points_to_skip = math.floor(PGDrivableAreaProperty.STRIPE_LENGTH * 2 / line_sample_interval) * 2
                else:
                    points_to_skip = 1
                for index in range(0, len(obj["polyline"]) - 1, points_to_skip):
                    color = [255, 0, 0] if MetaDriveType.is_lane(obj["type"]) and index==0\
                        else TopDownSemanticColor.get_color(obj["type"])
                    if index + 1 < len(obj["polyline"]):
                        s_p = obj["polyline"][index]
                        e_p = obj["polyline"][index + 1]
                        pygame.draw.line(
                            surface,
                            color,
                            surface.vec2pix([s_p[0], s_p[1]]),
                            surface.vec2pix([e_p[0], e_p[1]]),
                            # max(surface.pix(LaneGraphics.STRIPE_WIDTH),
                            surface.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH) * 2
                        )
    else:
        if isinstance(map, ScenarioMap):
            line_sample_interval = 2
            all_lanes = map.get_map_features(line_sample_interval)
            for id, data in all_lanes.items():
                if ScenarioDescription.POLYLINE not in data:
                    continue
                LaneGraphics.display_scenario_line(
                    data["polyline"], data["type"], surface, line_sample_interval=line_sample_interval
                )
        else:
            for _from in map.road_network.graph.keys():
                decoration = True if _from == Decoration.start else False
                for _to in map.road_network.graph[_from].keys():
                    for l in map.road_network.graph[_from][_to]:
                        two_side = True if l is map.road_network.graph[_from][_to][-1] or decoration else False
                        LaneGraphics.display(l, surface, two_side, use_line_color=True)

    return surface if return_surface else WorldSurface.to_cv2_image(surface)


def draw_top_down_trajectory(
    surface: WorldSurface, episode_data: dict, entry_differ_color=False, exit_differ_color=False, color_list=None
):
    if entry_differ_color or exit_differ_color:
        assert color_list is not None
    color_map = {}
    if not exit_differ_color and not entry_differ_color:
        color_type = 0
    elif exit_differ_color ^ entry_differ_color:
        color_type = 1
    else:
        color_type = 2

    if entry_differ_color:
        # init only once
        if "spawn_roads" in episode_data:
            spawn_roads = episode_data["spawn_roads"]
        else:
            spawn_roads = set()
            first_frame = episode_data["frame"][0]
            for state in first_frame[TARGET_VEHICLES].values():
                spawn_roads.add((state["spawn_road"][0], state["spawn_road"][1]))
        keys = [road[0] for road in list(spawn_roads)]
        keys.sort()
        color_map = {key: color for key, color in zip(keys, color_list)}

    for frame in episode_data["frame"]:
        for k, state, in frame[TARGET_VEHICLES].items():
            if color_type == 0:
                color = color_white
            elif color_type == 1:
                if exit_differ_color:
                    key = state["destination"][1]
                    if key not in color_map:
                        color_map[key] = color_list.pop()
                    color = color_map[key]
                else:
                    color = color_map[state["spawn_road"][0]]
            else:
                key_1 = state["spawn_road"][0]
                key_2 = state["destination"][1]
                if key_1 not in color_map:
                    color_map[key_1] = dict()
                if key_2 not in color_map[key_1]:
                    color_map[key_1][key_2] = color_list.pop()
                color = color_map[key_1][key_2]
            start = state["position"]
            pygame.draw.circle(surface, color, surface.pos2pix(start[0], start[1]), 1)
    for step, frame in enumerate(episode_data["frame"]):
        for k, state in frame[TARGET_VEHICLES].items():
            if not state["done"]:
                continue
            start = state["position"]
            if state["done"]:
                pygame.draw.circle(surface, (0, 0, 0), surface.pos2pix(start[0], start[1]), 5)
    return surface


class TopDownRenderer:
    def __init__(
        self,
        film_size=(2000, 2000),  # draw map in size = film_size/scaling. By default, it is set to 400m
        scaling=5,  # None for auto-scale
        screen_size=(800, 800),
        num_stack=15,
        history_smooth=0,
        show_agent_name=False,
        camera_position=None,
        target_agent_heading_up=False,
        target_vehicle_heading_up=None,
        draw_target_vehicle_trajectory=False,
        semantic_map=False,
        draw_center_line=False,
        semantic_broken_line=True,
        draw_contour=True,
        window=True,
        screen_record=False,
        center_on_map=False,
    ):
        """
        Launch a top-down renderer for current episode. Usually, it is launched by env.render(mode="topdown") and will
        be closed when next env.reset() is called or next episode starts.
        Args:
            film_size: The size of the film used to draw the map. The unit is pixel. It should cover the whole map to
            ensure it is complete in the rendered result. It works with the argument scaling to select the region
            to draw. For example, (2000, 2000) film size with scaling=5 can draw any maps whose width and height
            less than 2000/5 = 400 meters.

            scaling: The scaling determines how many pixels are used to draw one meter.

            screen_size: The size of the window popped up. It shows a region with width and length = screen_size/scaling

            num_stack: How many history steps to keep. History trajectory will show in faded color. It should be > 1

            history_smooth: Smoothing the trajectory by drawing less history positions. This value determines the sample
            rate. By default, this value is 0, meaning positions in previous num_stack steps will be shown.

            show_agent_name: Draw the name of the agent.

            camera_position: Set the (x,y) position of the top_down camera. If it is not specified, the camera will move
            with the ego car.

            target_agent_heading_up: Whether to rotate the camera according to the ego agent's heading. When enabled,
            The ego car always faces upwards.

            target_vehicle_heading_up: Deprecated, use target_agent_heading_up instead!

            draw_target_vehicle_trajectory: Whether to draw the ego car's whole trajectory without faded color

            semantic_map: Whether to draw semantic color for each object. The color scheme is in TopDownSemanticColor.

            draw_center_line: Whether to draw center line for each lane, this can be used to debug the lane connectivity

            semantic_broken_line: Whether to draw broken line for semantic map

            draw_contour: Whether to draw a counter for objects

            window: Whether to pop up the window. Setting it to 'False' enables off-screen rendering

            screen_record: Whether to record the episode. The recorded result can be accessed by
            env.top_down_renderer.screen_frames or env.top_down_renderer.generate_gif(file_name, fps)

            center_on_map: Whether to center the camera on the map. If set to True, the camera will not move with the
            ego car, and the camera position will be fixed at the center of the map.
        """
        # doc-end
        # LQY: do not delete the above line !!!!!

        # Setup some useful flags
        self.logger = get_logger()
        if num_stack < 1:
            self.logger.warning("num_stack should be greater than 0. Current value: {}. Set to 1".format(num_stack))
            num_stack = 1

        if target_vehicle_heading_up is not None:
            self.logger.warning("target_vehicle_heading_up is deprecated! Use target_agent_heading_up instead!")
            assert target_agent_heading_up is False
            target_agent_heading_up = target_vehicle_heading_up

        self.position = camera_position
        self.center_on_map = center_on_map
        self.target_agent_heading_up = target_agent_heading_up
        self.show_agent_name = show_agent_name
        self.draw_target_vehicle_trajectory = draw_target_vehicle_trajectory
        self.contour = draw_contour
        self.semantic_broken_line = semantic_broken_line
        self.no_window = not window

        if self.show_agent_name:
            pygame.init()

        self.screen_record = screen_record
        self._screen_frames = []
        self.pygame_font = None
        self.map = self.engine.current_map
        self.stack_frames = deque(maxlen=num_stack)
        self.history_objects = deque(maxlen=num_stack)
        self.history_target_vehicle = []
        self.history_smooth = history_smooth
        # self.current_track_agent = current_track_agent
        if self.target_agent_heading_up:
            assert self.current_track_agent is not None, "Specify which vehicle to track"
        self._text_render_pos = [50, 50]
        self._font_size = 25
        self._text_render_interval = 20
        self.semantic_map = semantic_map
        self.scaling = scaling
        self.film_size = film_size
        self._screen_size = screen_size

        # Setup the canvas
        # (1) background is the underlying layer that draws map.
        # It is fixed and will never change unless the map changes.
        self._background_canvas = draw_top_down_map_native(
            self.map,
            draw_center_line=draw_center_line,
            scaling=self.scaling,
            semantic_map=self.semantic_map,
            return_surface=True,
            film_size=self.film_size,
            semantic_broken_line=self.semantic_broken_line
        )
        self.scaling = self._background_canvas.scaling

        # (2) frame is a copy of the background so you can draw movable things on it.
        # It is super large as the background.
        self._frame_canvas = self._background_canvas.copy()

        # (3) canvas_rotate is only used when target_vehicle_heading_up=True and is use to center the tracked agent.
        if self.target_agent_heading_up:
            max_screen_size = max(self._screen_size[0], self._screen_size[1])
            self.canvas_rotate = pygame.Surface((max_screen_size * 2, max_screen_size * 2))

        # (4) screen_canvas is a regional surface where only part of the super large background will draw.
        # This will be used to as the final image shown in the screen & saved.
        self._screen_canvas = pygame.Surface(self._screen_size
                                             ) if self.no_window else pygame.display.set_mode(self._screen_size)
        self._screen_canvas.set_alpha(None)
        self._screen_canvas.fill(color_white)

        # Draw
        self.blit()

        # key accept
        self.need_reset = False

    @property
    def screen_canvas(self):
        return self._screen_canvas

    def refresh(self):
        self._frame_canvas.blit(self._background_canvas, (0, 0))
        self.screen_canvas.fill(color_white)

    def render(self, text, to_image=True, *args, **kwargs):
        if "semantic_map" in kwargs:
            self.semantic_map = kwargs["semantic_map"]

        self.need_reset = False
        if not self.no_window:
            key_press = pygame.key.get_pressed()
            if key_press[pygame.K_r]:
                self.need_reset = True

        # Record current target vehicle
        objects = self.engine.get_objects(lambda obj: not is_map_related_instance(obj))
        this_frame_objects = self._append_frame_objects(objects)
        self.history_objects.append(this_frame_objects)

        if self.draw_target_vehicle_trajectory:
            self.history_target_vehicle.append(
                history_object(
                    type=MetaDriveType.VEHICLE,
                    name=self.current_track_agent.name,
                    heading_theta=self.current_track_agent.heading_theta,
                    WIDTH=self.current_track_agent.top_down_width,
                    LENGTH=self.current_track_agent.top_down_length,
                    position=self.current_track_agent.position,
                    color=self.current_track_agent.top_down_color,
                    done=False
                )
            )

        self._handle_event()
        self.refresh()
        self._draw(*args, **kwargs)
        self._add_text(text)
        self.blit()
        ret = self.screen_canvas
        if not self.no_window:
            ret = ret.convert(24)
        ret = WorldSurface.to_cv2_image(ret) if to_image else ret
        if self.screen_record:
            self._screen_frames.append(ret)
        return ret

    def generate_gif(self, gif_name="demo.gif", duration=30):
        return generate_gif(self._screen_frames, gif_name, is_pygame_surface=False, duration=duration)

    def _add_text(self, text: dict):
        if not text:
            return
        if not pygame.get_init():
            pygame.init()
        font2 = pygame.font.SysFont('didot.ttc', 25)
        # pygame do not support multiline text render
        count = 0
        for key, value in text.items():
            line = str(key) + ":" + str(value)
            img2 = font2.render(line, True, (0, 0, 0))
            this_line_pos = copy.copy(self._text_render_pos)
            this_line_pos[-1] += count * self._text_render_interval
            self._screen_canvas.blit(img2, this_line_pos)
            count += 1

    def blit(self):
        if not self.no_window:
            pygame.display.update()

    def close(self):
        self.clear()
        pygame.quit()

    def clear(self):
        # # Reset the super large background
        self._background_canvas = None

        # Reset several useful variables.
        self._frame_canvas = None
        self.canvas_rotate = None

        self.history_objects.clear()
        self.stack_frames.clear()
        self.history_target_vehicle.clear()
        self.screen_frames.clear()

    @property
    def current_track_agent(self):
        return self.engine.current_track_agent

    @staticmethod
    def _append_frame_objects(objects):
        """
        Extract information for drawing objects
        Args:
            objects: list of BaseObject

        Returns: list of history_object

        """
        frame_objects = []
        for name, obj in objects.items():
            frame_objects.append(
                history_object(
                    name=name,
                    type=obj.metadrive_type if hasattr(obj, "metadrive_type") else MetaDriveType.OTHER,
                    heading_theta=obj.heading_theta,
                    WIDTH=obj.top_down_width,
                    LENGTH=obj.top_down_length,
                    position=obj.position,
                    color=obj.top_down_color,
                    done=False
                )
            )
        return frame_objects

    def _draw(self, *args, **kwargs):
        """
        This is the core function to process the
        """
        if len(self.history_objects) == 0:
            return

        for i, objects in enumerate(self.history_objects):
            if i == len(self.history_objects) - 1:
                continue
            i = len(self.history_objects) - i
            if self.history_smooth != 0 and (i % self.history_smooth != 0):
                continue
            for v in objects:
                c = v.color
                h = v.heading_theta
                h = h if abs(h) > 2 * np.pi / 180 else 0
                x = abs(int(i))
                alpha_f = x / len(self.history_objects)
                if self.semantic_map:
                    c = TopDownSemanticColor.get_color(v.type) * (1 - alpha_f) + alpha_f * 255
                else:
                    c = (c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2]))
                ObjectGraphics.display(object=v, surface=self._frame_canvas, heading=h, color=c, draw_contour=False)

        # Draw the whole trajectory of ego vehicle with no gradient colors:
        if self.draw_target_vehicle_trajectory:
            for i, v in enumerate(self.history_target_vehicle):
                i = len(self.history_target_vehicle) - i
                if self.history_smooth != 0 and (i % self.history_smooth != 0):
                    continue
                c = v.color
                h = v.heading_theta
                h = h if abs(h) > 2 * np.pi / 180 else 0
                x = abs(int(i))
                alpha_f = min(x / len(self.history_target_vehicle), 0.5)
                # alpha_f = 0
                ObjectGraphics.display(
                    object=v,
                    surface=self._frame_canvas,
                    heading=h,
                    color=(c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2])),
                    draw_contour=False
                )

        # Draw current vehicle with black contour
        # Use this line if you wish to draw "future" trajectory.
        # i is the index of vehicle that we will render a black box for it.
        # i = int(len(self.history_vehicles) / 2)
        i = -1
        for v in self.history_objects[i]:
            h = v.heading_theta
            c = v.color
            h = h if abs(h) > 2 * np.pi / 180 else 0
            alpha_f = 0
            if self.semantic_map:
                c = TopDownSemanticColor.get_color(v.type) * (1 - alpha_f) + alpha_f * 255
            else:
                c = (c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2]))
            ObjectGraphics.display(
                object=v, surface=self._frame_canvas, heading=h, color=c, draw_contour=self.contour, contour_width=2
            )

        if not hasattr(self, "_deads"):
            self._deads = []

        for v in self._deads:
            pygame.draw.circle(
                surface=self._frame_canvas,
                color=(255, 0, 0),
                center=self._frame_canvas.pos2pix(v.position[0], v.position[1]),
                radius=5
            )

        for v in self.history_objects[i]:
            if v.done:
                pygame.draw.circle(
                    surface=self._frame_canvas,
                    color=(255, 0, 0),
                    center=self._frame_canvas.pos2pix(v.position[0], v.position[1]),
                    radius=5
                )
                self._deads.append(v)

        v = self.current_track_agent
        canvas = self._frame_canvas
        field = self._screen_canvas.get_size()
        if not self.target_agent_heading_up:
            if self.position is not None or v is not None:
                if self.center_on_map:
                    frame_canvas_size = self._frame_canvas.get_size()
                    position = (frame_canvas_size[0] / 2, frame_canvas_size[1] / 2)
                else:
                    cam_pos = (self.position or v.position)
                    position = self._frame_canvas.pos2pix(*cam_pos)
            else:
                position = (field[0] / 2, field[1] / 2)
            off = (position[0] - field[0] / 2, position[1] - field[1] / 2)
            self.screen_canvas.blit(source=canvas, dest=(0, 0), area=(off[0], off[1], field[0], field[1]))
        else:
            position = self._frame_canvas.pos2pix(*v.position)
            area = (
                position[0] - self.canvas_rotate.get_size()[0] / 2, position[1] - self.canvas_rotate.get_size()[1] / 2,
                self.canvas_rotate.get_size()[0], self.canvas_rotate.get_size()[1]
            )
            self.canvas_rotate.fill(color_white)
            self.canvas_rotate.blit(source=canvas, dest=(0, 0), area=area)

            rotation = -np.rad2deg(v.heading_theta) + 90
            new_canvas = pygame.transform.rotozoom(self.canvas_rotate, rotation, 1)

            size = self._screen_canvas.get_size()
            self._screen_canvas.blit(
                new_canvas,
                (0, 0),
                (
                    new_canvas.get_size()[0] / 2 - size[0] / 2,  # Left
                    new_canvas.get_size()[1] / 2 - size[1] / 2,  # Top
                    size[0],  # Width
                    size[1]  # Height
                )
            )

        if self.show_agent_name:
            raise ValueError("This function is broken")
            # FIXME check this later
            if self.pygame_font is None:
                self.pygame_font = pygame.font.SysFont("Arial.ttf", 30)
            agents = [agent.name for agent in list(self.engine.agents.values())]
            for v in self.history_objects[i]:
                if v.name in agents:
                    position = self._frame_canvas.pos2pix(*v.position)
                    new_position = (position[0] - off[0], position[1] - off[1])
                    img = self.pygame_font.render(
                        text=self.engine.object_to_agent(v.name),
                        antialias=True,
                        color=(0, 0, 0, 128),
                    )
                    # img.set_alpha(None)
                    self.screen_canvas.blit(
                        source=img,
                        dest=(new_position[0] - img.get_width() / 2, new_position[1] - img.get_height() / 2),
                        # special_flags=pygame.BLEND_RGBA_MULT
                    )

    def _handle_event(self) -> None:
        """
        Handle pygame events for moving and zooming in the displayed area.
        """
        if self.no_window:
            return
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    import sys
                    sys.exit()

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    @property
    def screen_frames(self):
        return copy.deepcopy(self._screen_frames)

    def get_map(self):
        """
        Convert the map pygame surface to array

        Returns: map in array

        """
        return pygame.surfarray.array3d(self._background_canvas)
