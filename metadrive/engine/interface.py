import logging
import math
import time

import numpy as np
from panda3d.core import NodePath, TextNode, LQuaternionf

from metadrive.constants import COLLISION_INFO_COLOR, COLOR, MetaDriveType, \
    CamMask, RENDER_MODE_NONE
from metadrive.engine.asset_loader import AssetLoader


class DisplayRegionPosition:
    left = [0., 1 / 3, 0.8, 1.0]
    mid = [1 / 3, 2 / 3, 0.8, 1.0]
    right = [2 / 3, 1, 0.8, 1.0]


class Interface:
    """
    Visualization interface, state banner and vehicle panel
    """
    ARROW_COLOR = COLLISION_INFO_COLOR["green"][1]

    def __init__(self, base_engine):
        self._node_path_list = []
        # self.engine = base_engine
        self.dashboard = None
        self.right_panel = None
        self.mid_panel = None
        self.left_panel = None
        self.contact_result_render = None
        self.arrow = None
        self._left_arrow = None
        self._right_arrow = None
        self._contact_banners = {}  # to save time/memory
        self.current_banner = None
        self.need_interface = base_engine.mode != RENDER_MODE_NONE and not base_engine.global_config[
            "debug_physics_world"] and base_engine.global_config["show_interface"]
        if base_engine.mode == RENDER_MODE_NONE:
            assert self.need_interface is False, \
                "We should not using interface with extra cameras when in offscreen mode!"
        self._init_interface()
        self._is_showing_arrow = True  # store the state of navigation mark

    def after_step(self):
        if self.engine.current_track_agent is not None and self.need_interface and self.engine.mode != RENDER_MODE_NONE:
            track_v = self.engine.current_track_agent
            if self.dashboard is not None:
                self.dashboard.update_vehicle_state(track_v)
            self._render_contact_result(track_v.contact_results)
            if hasattr(track_v, "navigation") and track_v.navigation is not None:
                self._update_navi_arrow(track_v.navigation.navi_arrow_dir)

    def _init_interface(self):
        if self.need_interface:
            info_np = NodePath("Collision info nodepath")
            info_np.reparentTo(self.engine.aspect2d)

            self._node_path_list.append(info_np)

            self.contact_result_render = info_np
            for idx, panel_name, in enumerate(reversed(self.engine.global_config["interface_panel"])):
                if idx == 0:
                    self.right_panel = self.engine.get_sensor(panel_name)
                elif idx == 1:
                    self.mid_panel = self.engine.get_sensor(panel_name)
                elif idx == 2:
                    self.left_panel = self.engine.get_sensor(panel_name)
                else:
                    raise ValueError("Can not add > 3 panels!")
                if panel_name == "dashboard":
                    self.dashboard = self.engine.get_sensor(panel_name)

            self.arrow = self.engine.aspect2d.attachNewNode("arrow")
            self._node_path_list.append(self.arrow)

            navi_arrow_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "navi_arrow.gltf"))
            navi_arrow_model.setScale(0.1, 0.12, 0.2)
            navi_arrow_model.setPos(2, 1.15, -0.221)
            self._left_arrow = self.arrow.attachNewNode("left arrow")
            self._node_path_list.append(self._left_arrow)

            self._left_arrow.setP(180)
            self._right_arrow = self.arrow.attachNewNode("right arrow")
            self._node_path_list.append(self._right_arrow)

            self._left_arrow.setColor(self.ARROW_COLOR)
            self._right_arrow.setColor(self.ARROW_COLOR)
            if self.engine.global_config["show_interface_navi_mark"]:
                navi_arrow_model.instanceTo(self._left_arrow)
                navi_arrow_model.instanceTo(self._right_arrow)
            self._left_arrow.detachNode()
            self._right_arrow.detachNode()
            self.arrow.setPos(0, 0, 0.08)
            self.arrow.hide(CamMask.AllOn)
            self.arrow.show(CamMask.MainCam)
            self.arrow.setQuat(LQuaternionf(math.cos(-math.pi / 4), 0, 0, math.sin(-math.pi / 4)))
            # the transparency attribute of gltf model is invalid on windows
            # self.arrow.setTransparency(TransparencyAttrib.M_alpha)

    def undisplay(self):
        """
        Remove the panels and the badge
        """
        if self.need_interface:
            if self.right_panel:
                self.right_panel.remove_display_region()
            if self.mid_panel:
                self.mid_panel.remove_display_region()
            if self.left_panel:
                self.left_panel.remove_display_region()
            self.contact_result_render.detachNode()
            self.arrow.detachNode()

    def display(self):
        """
        Add the panels and the badge
        """
        if self.need_interface:
            if self.right_panel:
                self.right_panel.add_display_region(DisplayRegionPosition.right)
            if self.mid_panel:
                self.mid_panel.add_display_region(DisplayRegionPosition.mid)
            if self.left_panel:
                self.left_panel.add_display_region(DisplayRegionPosition.left)

            self.arrow.reparentTo(self.engine.aspect2d)
            self.contact_result_render.reparentTo(self.engine.aspect2d)

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

            self._node_path_list.append(new_banner)

            self._contact_banners[text] = new_banner
            text_node = new_banner.node()
            text_node.setCardColor(color)
            text_node.setText(text)
            text_node.setTextScale(0.96)
            text_node.setCardActual(-9, 9.1, -0.26, 1)
            text_node.setCardDecal(True)
            text_node.setTextColor(1, 1, 1, 1)
            text_node.setAlign(TextNode.A_center)
            new_banner.setScale(0.05 * 3 / 4 * self.engine.w_scale)
            new_banner.setPos(-0.662 * self.engine.w_scale, 0, -0.987 * self.engine.h_scale)
            # new_banner.setPos(-0.75 * self.engine.w_scale, 0, -0.8 * self.engine.h_scale)
            new_banner.reparentTo(self.contact_result_render)
            self.current_banner = new_banner

    def _render_contact_result(self, contacts):
        contacts = sorted(list(contacts), key=lambda c: COLLISION_INFO_COLOR[COLOR[c]][0])
        text = contacts[0] if len(contacts) != 0 else MetaDriveType.UNSET
        color = COLLISION_INFO_COLOR[COLOR[text]][1]
        if time.time() - self.engine._episode_start_time < 10:
            text = "Press H to see help message"
        self._render_banner(text, color)

    def destroy(self):

        for np in self._node_path_list:
            np.detachNode()
            np.removeNode()

        if self.need_interface:
            self.undisplay()
            self.contact_result_render.removeNode()
            self.arrow.removeNode()
            self.contact_result_render = None
            self._contact_banners = None
            self.current_banner = None
            if self.right_panel:
                self.right_panel.destroy()
            if self.mid_panel:
                self.mid_panel.destroy()
            if self.left_panel:
                self.left_panel.destroy()

    def _update_navi_arrow(self, lanes_heading):
        if not self.engine.global_config["vehicle_config"]["show_navigation_arrow"]:
            return
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
            left = True if cross_product[-1] < 0 else False
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

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()
