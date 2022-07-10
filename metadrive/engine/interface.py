import math
import time

import numpy as np
from panda3d.core import NodePath, TextNode, PGTop, CardMaker, Vec3, LQuaternionf

from metadrive.constants import RENDER_MODE_ONSCREEN, COLLISION_INFO_COLOR, COLOR, BodyName, CamMask
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer


class Interface:
    """
    Visualization interface, state banner and vehicle panel
    """
    ARROW_COLOR = COLLISION_INFO_COLOR["green"][1]

    def __init__(self, base_engine):
        self.engine = base_engine
        self.vehicle_panel = None
        self.right_panel = None
        self.mid_panel = None
        self.left_panel = None
        self.contact_result_render = None
        self.arrow = None
        self._left_arrow = None
        self._right_arrow = None
        self._contact_banners = {}  # to save time/memory
        self.current_banner = None
        self.need_interface = self.engine.mode == RENDER_MODE_ONSCREEN and not self.engine.global_config[
            "debug_physics_world"]
        self.need_interface = self.need_interface and base_engine.global_config["show_interface"]
        self.init_interface()
        self._is_showing_arrow = True  # store the state of navigation mark

    def after_step(self):
        if self.engine.current_track_vehicle is not None and self.need_interface:
            track_v = self.engine.current_track_vehicle
            self.vehicle_panel.update_vehicle_state(track_v)
            self._render_contact_result(track_v.contact_results)
            if hasattr(track_v, "navigation"):
                self._update_navi_arrow(track_v.navigation.navi_arrow_dir)

    def init_interface(self):
        from metadrive.component.vehicle_module.mini_map import MiniMap
        from metadrive.component.vehicle_module.rgb_camera import RGBCamera
        from metadrive.component.vehicle_module.depth_camera import DepthCamera
        if self.need_interface:
            info_np = NodePath("Collision info nodepath")
            info_np.reparentTo(self.engine.aspect2d)
            self.contact_result_render = info_np
            self.vehicle_panel = VehiclePanel(self.engine)
            self.right_panel = self.vehicle_panel
            self.mid_panel = RGBCamera(
            ) if self.engine.global_config["vehicle_config"]["image_source"] != "depth_camera" else DepthCamera()
            self.left_panel = MiniMap()
            self.arrow = self.engine.aspect2d.attachNewNode("arrow")
            navi_arrow_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "navi_arrow.gltf"))
            navi_arrow_model.setScale(0.1, 0.12, 0.2)
            navi_arrow_model.setPos(2, 1.15, -0.221)
            self._left_arrow = self.arrow.attachNewNode("left arrow")
            self._left_arrow.setP(180)
            self._right_arrow = self.arrow.attachNewNode("right arrow")
            self._left_arrow.setColor(self.ARROW_COLOR)
            self._right_arrow.setColor(self.ARROW_COLOR)
            if self.engine.global_config["show_interface_navi_mark"]:
                navi_arrow_model.instanceTo(self._left_arrow)
                navi_arrow_model.instanceTo(self._right_arrow)
            self.arrow.setPos(0, 0, 0.08)
            self.arrow.hide(CamMask.AllOn)
            self.arrow.show(CamMask.MainCam)
            self.arrow.setQuat(LQuaternionf(math.cos(-math.pi / 4), 0, 0, math.sin(-math.pi / 4)))
            # the transparency attribute of gltf model is invalid on windows
            # self.arrow.setTransparency(TransparencyAttrib.M_alpha)

    def stop_track(self):
        if self.need_interface:
            self.vehicle_panel.remove_display_region()
            self.vehicle_panel.buffer.set_active(False)
            self.contact_result_render.detachNode()
            self.mid_panel.remove_display_region()
            self.left_panel.remove_display_region()
            self.arrow.detachNode()

    def track(self, vehicle):
        if self.need_interface:
            self.vehicle_panel.buffer.set_active(True)
            self.arrow.reparentTo(self.engine.aspect2d)
            self.contact_result_render.reparentTo(self.engine.aspect2d)
            self.vehicle_panel.add_display_region(self.vehicle_panel.display_region_size)
            for p in [self.left_panel, self.mid_panel]:
                p.track(vehicle)
                p.add_display_region(p.display_region_size)

    def _render_banner(self, text, color=COLLISION_INFO_COLOR["green"][1]):
        """
        Render the banner in the left bottom corner.
        """
        if self.contact_result_render is None:
            return
        if self.current_banner is not None:
            self.current_banner.detachNode()
        if text in self._contact_banners:
            self._contact_banners[text].reparentTo(self.contact_result_render)
            self.current_banner = self._contact_banners[text]
        else:
            new_banner = NodePath(TextNode("collision_info:{}".format(text)))
            self._contact_banners[text] = new_banner
            text_node = new_banner.node()
            text_node.setCardColor(color)
            text_node.setText(text)
            text_node.setCardActual(-5 * self.engine.w_scale, 5.1 * self.engine.w_scale, -0.3, 1)
            text_node.setCardDecal(True)
            text_node.setTextColor(1, 1, 1, 1)
            text_node.setAlign(TextNode.A_center)
            new_banner.setScale(0.05)
            new_banner.setPos(-0.75 * self.engine.w_scale, 0, -0.8 * self.engine.h_scale)
            new_banner.reparentTo(self.contact_result_render)
            self.current_banner = new_banner

    def _render_contact_result(self, contacts):
        contacts = sorted(list(contacts), key=lambda c: COLLISION_INFO_COLOR[COLOR[c]][0])
        text = contacts[0] if len(contacts) != 0 else None
        if text is None:
            text = "Normal" if time.time() - self.engine._episode_start_time > 10 else "Press H to see help message"
            self._render_banner(text, COLLISION_INFO_COLOR["green"][1])
        else:
            if text == BodyName.Vehicle:
                text = BodyName.Vehicle
            self._render_banner(text, COLLISION_INFO_COLOR[COLOR[text]][1])

    def destroy(self):
        if self.need_interface:
            self.stop_track()
            self.vehicle_panel.destroy()
            self.contact_result_render.removeNode()
            self.arrow.removeNode()
            self.contact_result_render = None
            self._contact_banners = None
            self.current_banner = None
            self.mid_panel.destroy()
            self.left_panel.destroy()

    def _update_navi_arrow(self, lanes_heading):
        lane_0_heading = lanes_heading[0]
        lane_1_heading = lanes_heading[1]
        if abs(lane_0_heading - lane_1_heading) < 0.01:
            if self._is_showing_arrow:
                self._left_arrow.detachNode()
                self._right_arrow.detachNode()
                self._is_showing_arrow = False
        else:
            dir_0 = np.array([math.cos(lane_0_heading), math.sin(lane_0_heading), 0])
            dir_1 = np.array([math.cos(lane_1_heading), math.sin(lane_1_heading), 0])
            cross_product = np.cross(dir_1, dir_0)
            left = False if cross_product[-1] < 0 else True
            if not self._is_showing_arrow:
                self._is_showing_arrow = True
            if left:
                if not self._left_arrow.hasParent():
                    self._left_arrow.reparentTo(self.arrow)
                if self._right_arrow.hasParent():
                    self._right_arrow.detachNode()
            else:
                if not self._right_arrow.hasParent():
                    self._right_arrow.reparentTo(self.arrow)
                if self._left_arrow.hasParent():
                    self._left_arrow.detachNode()


class VehiclePanel(ImageBuffer):
    PARA_VIS_LENGTH = 12
    PARA_VIS_HEIGHT = 1
    MAX_SPEED = 120
    BUFFER_W = 2
    BUFFER_H = 1
    CAM_MASK = CamMask.PARA_VIS
    GAP = 4.1
    TASK_NAME = "update panel"
    display_region_size = [2 / 3, 1, ImageBuffer.display_bottom, ImageBuffer.display_top]

    def __init__(self, engine: EngineCore):
        if engine.win is None:
            return
        self.aspect2d_np = NodePath(PGTop("aspect2d"))
        self.aspect2d_np.show(self.CAM_MASK)
        self.para_vis_np = {}
        # make_buffer_func, make_camera_func = engine.win.makeTextureBuffer, engine.makeCamera

        # don't delete the space in word, it is used to set a proper position
        for i, np_name in enumerate(["Steering", " Throttle", "     Brake", "    Speed"]):
            text = TextNode(np_name)
            text.setText(np_name)
            text.setSlant(0.1)
            textNodePath = self.aspect2d_np.attachNewNode(text)
            textNodePath.setScale(0.052)
            text.setFrameColor(0, 0, 0, 1)
            text.setTextColor(0, 0, 0, 1)
            text.setFrameAsMargin(-self.GAP, self.PARA_VIS_LENGTH, 0, 0)
            text.setAlign(TextNode.ARight)
            textNodePath.setPos(-1.125111, 0, 0.9 - i * 0.08)
            if i != 0:
                cm = CardMaker(np_name)
                cm.setFrame(0, self.PARA_VIS_LENGTH - 0.21, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2)
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                card.setPos(0.21, 0, 0.22)
                self.para_vis_np[np_name.lstrip()] = card
            else:
                # left
                name = "Left"
                cm = CardMaker(name)
                cm.setFrame(
                    0, (self.PARA_VIS_LENGTH - 0.4) / 2, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2
                )
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                card.setPos(0.2 + self.PARA_VIS_LENGTH / 2, 0, 0.22)
                self.para_vis_np[name] = card
                # right
                name = "Right"
                cm = CardMaker(np_name)
                cm.setFrame(
                    -(self.PARA_VIS_LENGTH - 0.1) / 2, 0, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2
                )
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                card.setPos(0.2 + self.PARA_VIS_LENGTH / 2, 0, 0.22)
                self.para_vis_np[name] = card
        super(VehiclePanel, self).__init__(
            self.BUFFER_W,
            self.BUFFER_H,
            Vec3(-0.9, -1.01, 0.78),
            self.BKG_COLOR,
            parent_node=self.aspect2d_np,
            # engine=engine
        )
        self.add_display_region(self.display_region_size)

    def update_vehicle_state(self, vehicle):
        steering, throttle_brake, speed = vehicle.steering, vehicle.throttle_brake, vehicle.speed
        if throttle_brake < 0:
            self.para_vis_np["Throttle"].setScale(0, 1, 1)
            self.para_vis_np["Brake"].setScale(-throttle_brake, 1, 1)
        elif throttle_brake > 0:
            self.para_vis_np["Throttle"].setScale(throttle_brake, 1, 1)
            self.para_vis_np["Brake"].setScale(0, 1, 1)
        else:
            self.para_vis_np["Throttle"].setScale(0, 1, 1)
            self.para_vis_np["Brake"].setScale(0, 1, 1)

        steering_value = abs(steering)
        if steering < 0:
            self.para_vis_np["Left"].setScale(steering_value, 1, 1)
            self.para_vis_np["Right"].setScale(0, 1, 1)
        elif steering > 0:
            self.para_vis_np["Right"].setScale(steering_value, 1, 1)
            self.para_vis_np["Left"].setScale(0, 1, 1)
        else:
            self.para_vis_np["Right"].setScale(0, 1, 1)
            self.para_vis_np["Left"].setScale(0, 1, 1)
        speed_value = speed / self.MAX_SPEED
        self.para_vis_np["Speed"].setScale(speed_value, 1, 1)

    def remove_display_region(self):
        super(VehiclePanel, self).remove_display_region()

    def add_display_region(self, display_region):
        super(VehiclePanel, self).add_display_region(display_region)
        self.origin.reparentTo(self.aspect2d_np)

    def destroy(self):
        super(VehiclePanel, self).destroy()
        for para in self.para_vis_np.values():
            para.removeNode()
        self.aspect2d_np.removeNode()
