import logging
import sys
import time
from typing import Optional, Union, Tuple

import gltf
from direct.gui.OnscreenImage import OnscreenImage
from direct.showbase import ShowBase
from panda3d.bullet import BulletDebugNode
from panda3d.core import AntialiasAttrib, loadPrcFileData, LineSegs, PythonCallbackObject, Vec3, NodePath

from metadrive.constants import RENDER_MODE_OFFSCREEN, RENDER_MODE_NONE, RENDER_MODE_ONSCREEN, EDITION, CamMask, \
    BKG_COLOR
from metadrive.engine.asset_loader import initialize_asset_loader, close_asset_loader, randomize_cover, get_logo_file
from metadrive.engine.core.collision_callback import collision_callback
from metadrive.engine.core.force_fps import ForceFPS
from metadrive.engine.core.light import Light
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.engine.core.sky_box import SkyBox
from metadrive.engine.core.terrain import Terrain
from metadrive.utils.utils import is_mac, setup_logger


def _suppress_warning():
    loadPrcFileData("", "notify-level-glgsg fatal")
    loadPrcFileData("", "notify-level-pgraph fatal")
    loadPrcFileData("", "notify-level-pnmimage fatal")
    loadPrcFileData("", "notify-level-thread fatal")
    loadPrcFileData("", "notify-level-bullet fatal")


def _free_warning():
    loadPrcFileData("", "notify-level-glgsg debug")
    # loadPrcFileData("", "notify-level-pgraph debug")  # press 4 to use toggle analyze to do this
    loadPrcFileData("", "notify-level-pnmimage debug")
    loadPrcFileData("", "notify-level-thread debug")


def attach_cover_image(window_width, window_height):
    cover_file_path = randomize_cover()
    image = OnscreenImage(image=cover_file_path, pos=(0, 0, 0), scale=(1, 1, 1))
    if window_width > window_height:
        scale = window_width / window_height
    else:
        scale = window_height / window_width
    image.set_scale((scale, 1, scale))
    image.setTransparency(True)
    return image


def attach_logo(engine):
    cover_file_path = get_logo_file()
    image = OnscreenImage(image=cover_file_path)
    scale = 0.075
    image.set_scale((scale * 3, 1, scale))
    image.set_pos((0.8325 * engine.w_scale, 0, -0.94 * engine.h_scale))
    image.set_antialias(AntialiasAttrib.MMultisample)
    image.setTransparency(True)
    return image


class EngineCore(ShowBase.ShowBase):
    DEBUG = False
    loadPrcFileData("", "window-title {}".format(EDITION))
    loadPrcFileData("", "framebuffer-multisample 1")
    loadPrcFileData("", "multisamples 8")
    loadPrcFileData("", 'bullet-filter-algorithm groups-mask')
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "model-cache-compressed-textures 1")

    # loadPrcFileData("", "transform-cache 0")
    # loadPrcFileData("", "state-cache 0")
    loadPrcFileData("", "garbage-collect-states 0")

    # loadPrcFileData("", " framebuffer-srgb truein")
    # loadPrcFileData("", "geom-cache-size 50000")

    # v-sync, it seems useless
    # loadPrcFileData("", "sync-video 1")

    # for debug use
    # loadPrcFileData("", "gl-version 3 2")

    def __init__(self, global_config):
        self.global_config = global_config
        if self.global_config["pstats"]:
            # pstats debug provided by panda3d
            loadPrcFileData("", "want-pstats 1")

        loadPrcFileData("", "win-size {} {}".format(*self.global_config["window_size"]))

        # Setup onscreen render
        if self.global_config["use_render"]:
            self.mode = RENDER_MODE_ONSCREEN
            # Warning it may cause memory leak, Pand3d Official has fixed this in their master branch.
            # You can enable it if your panda version is latest.
            loadPrcFileData("", "threading-model Cull/Draw")  # multi-thread render, accelerate simulation when evaluate
        else:
            if self.global_config["offscreen_render"]:
                self.mode = RENDER_MODE_OFFSCREEN
                loadPrcFileData("", "threading-model Cull/Draw")
            else:
                self.mode = RENDER_MODE_NONE

        if is_mac() and (self.mode == RENDER_MODE_OFFSCREEN):  # Mac don't support offscreen rendering
            self.mode = RENDER_MODE_ONSCREEN

        # Setup some debug options
        if self.global_config["headless_machine_render"]:
            # headless machine support
            loadPrcFileData("", "load-display  pandagles2")
        if self.global_config["debug"]:
            # debug setting
            EngineCore.DEBUG = True
            _free_warning()
            setup_logger(debug=True)
            self.accept('1', self.toggleDebug)
            self.accept('2', self.toggleWireframe)
            self.accept('3', self.toggleTexture)
            self.accept('4', self.toggleAnalyze)
        else:
            # only report fatal error when debug is False
            _suppress_warning()
            # a special debug mode
            if self.global_config["debug_physics_world"]:
                self.accept('1', self.toggleDebug)
                self.accept('4', self.toggleAnalyze)

        if self.global_config["disable_model_compression"]:
            pass
        else:
            loadPrcFileData("", "compressed-textures 1")  # Default to compress

        super(EngineCore, self).__init__(windowType=self.mode)

        # main_window_position = (0, 0)
        if self.mode == RENDER_MODE_ONSCREEN:
            h = self.pipe.getDisplayHeight()
            w = self.pipe.getDisplayWidth()
            if self.global_config["window_size"][0] > 0.9 * w or self.global_config["window_size"][1] > 0.9 * h:
                old_scale = self.global_config["window_size"][0] / self.global_config["window_size"][1]
                new_w = int(min(0.9 * w, 0.9 * h * old_scale))
                new_h = int(min(0.9 * h, 0.9 * w / old_scale))
                self.global_config["window_size"] = tuple([new_w, new_h])
                from panda3d.core import WindowProperties
                props = WindowProperties()
                props.setSize(self.global_config["window_size"][0], self.global_config["window_size"][1])
                self.win.requestProperties(props)
                logging.warning(
                    "Since your screen is too small ({}, {}), we resize the window to {}.".format(
                        w, h, self.global_config["window_size"]
                    )
                )
            # main_window_position = (
            #     (w - self.global_config["window_size"][0]) / 2, (h - self.global_config["window_size"][1]) / 2
            # )

        # screen scale factor
        self.w_scale = max(self.global_config["window_size"][0] / self.global_config["window_size"][1], 1)
        self.h_scale = max(self.global_config["window_size"][1] / self.global_config["window_size"][0], 1)

        if self.mode == RENDER_MODE_ONSCREEN:
            self.disableMouse()

        if not self.global_config["debug_physics_world"] and (self.mode in [RENDER_MODE_ONSCREEN, RENDER_MODE_OFFSCREEN
                                                                            ]):
            initialize_asset_loader(self)
            gltf.patch_loader(self.loader)

            # Display logo
            if self.mode == RENDER_MODE_ONSCREEN and (not self.global_config["debug"]):
                if self.global_config["show_logo"]:
                    self._window_logo = attach_logo(self)
                self._loading_logo = attach_cover_image(
                    window_width=self.get_size()[0], window_height=self.get_size()[1]
                )
                for i in range(5):
                    self.graphicsEngine.renderFrame()
                self.taskMgr.add(self.remove_logo, "remove _loading_logo in first frame")

        self.closed = False

        # add element to render and pbr render, if is exists all the time.
        # these element will not be removed when clear_world() is called
        self.pbr_render = self.render.attachNewNode("pbrNP")

        # attach node to this root whose children nodes will be clear after calling clear_world()
        self.worldNP = self.render.attachNewNode("world_np")

        # same as worldNP, but this node is only used for render gltf model with pbr material
        self.pbr_worldNP = self.pbr_render.attachNewNode("pbrNP")
        self.debug_node = None

        # some render attribute
        self.pbrpipe = None
        self.world_light = None

        # physics world
        self.physics_world = PhysicsWorld(self.global_config["debug_static_world"])

        # collision callback
        self.physics_world.dynamic_world.setContactAddedCallback(PythonCallbackObject(collision_callback))

        # for real time simulation
        self.force_fps = ForceFPS(self)

        # init terrain
        self.terrain = Terrain(self.global_config["show_terrain"])
        self.terrain.attach_to_world(self.render, self.physics_world)

        # init other world elements
        if self.mode != RENDER_MODE_NONE:

            from metadrive.engine.core.our_pbr import OurPipeline
            self.pbrpipe = OurPipeline(
                render_node=None,
                window=None,
                camera_node=None,
                msaa_samples=4,
                max_lights=8,
                use_normal_maps=False,
                use_emission_maps=True,
                exposure=1.0,
                enable_shadows=False,
                enable_fog=False,
                use_occlusion_maps=False
            )
            self.pbrpipe.render_node = self.pbr_render
            self.pbrpipe.render_node.set_antialias(AntialiasAttrib.M_auto)
            self.pbrpipe._recompile_pbr()
            self.pbrpipe.manager.cleanup()

            # set main cam
            self.cam.node().setCameraMask(CamMask.MainCam)
            self.cam.node().getDisplayRegion(0).setClearColorActive(True)
            self.cam.node().getDisplayRegion(0).setClearColor(BKG_COLOR)
            lens = self.cam.node().getLens()
            lens.setFov(80)

            self.sky_box = SkyBox(not self.global_config["show_skybox"])
            self.sky_box.attach_to_world(self.render, self.physics_world)

            self.world_light = Light(self.global_config)
            self.world_light.attach_to_world(self.render, self.physics_world)
            self.render.setLight(self.world_light.direction_np)
            self.render.setLight(self.world_light.ambient_np)

            self.render.setShaderAuto()
            self.render.setAntialias(AntialiasAttrib.MAuto)

            # ui and render property
            if self.global_config["show_fps"]:
                self.setFrameRateMeter(True)

            # onscreen message
            self.on_screen_message = ScreenMessage(debug=self.DEBUG) if self.mode == RENDER_MODE_ONSCREEN else None
            self._show_help_message = False
            self._episode_start_time = time.time()

            self.accept("h", self.toggle_help_message)
            self.accept("f", self.force_fps.toggle)
            self.accept("escape", sys.exit)
        else:
            self.on_screen_message = None

        # task manager
        self.taskMgr.remove('audioLoop')

    def render_frame(self, text: Optional[Union[dict, str]] = None):
        """
        The real rendering is conducted by the igLoop task maintained by panda3d.
        Frame will be drawn and refresh, when taskMgr.step() is called.
        This function is only used to pass the message that needed to be printed in the screen to underlying renderer.
        :param text: A dict containing key and values or a string.
        :return: None
        """

        if self.on_screen_message is not None:
            self.on_screen_message.render(text)
        if self.mode == RENDER_MODE_ONSCREEN:
            self.sky_box.step()

    def step_physics_world(self):
        dt = self.global_config["physics_world_step_size"]
        self.physics_world.dynamic_world.doPhysics(dt, 1, dt)

    def _debug_mode(self):
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(True)
        debugNP = self.render.attachNewNode(debugNode)
        self.physics_world.dynamic_world.setDebugNode(debugNP.node())
        self.debug_node = debugNP

    def toggleAnalyze(self):
        self.worldNP.analyze()
        print(self.physics_world.report_bodies())
        # self.worldNP.ls()

    def toggleDebug(self):
        if self.debug_node is None:
            self._debug_mode()
        if self.debug_node.isHidden():
            self.debug_node.show()
        else:
            self.debug_node.hide()

    def report_body_nums(self, task):
        logging.debug(self.physics_world.report_bodies())
        return task.done

    def close_world(self):
        self.taskMgr.stop()
        # It will report a warning said AsynTaskChain is created when taskMgr.destroy() is called but a new showbase is
        # created.
        logging.debug(
            "Before del taskMgr: task_chain_num={}, all_tasks={}".format(
                self.taskMgr.mgr.getNumTaskChains(), self.taskMgr.getAllTasks()
            )
        )
        self.taskMgr.destroy()
        logging.debug(
            "After del taskMgr: task_chain_num={}, all_tasks={}".format(
                self.taskMgr.mgr.getNumTaskChains(), self.taskMgr.getAllTasks()
            )
        )
        self.physics_world.dynamic_world.clearContactAddedCallback()
        self.physics_world.destroy()
        self.destroy()
        close_asset_loader()

        import sys
        if sys.version_info >= (3, 0):
            import builtins
        else:
            import __builtin__ as builtins
        if hasattr(builtins, 'base'):
            del builtins.base

    def clear_world(self):
        self.worldNP.removeNode()
        self.pbr_worldNP.removeNode()

    def toggle_help_message(self):
        if self.on_screen_message:
            self.on_screen_message.toggle_help_message()

    def draw_line(self, start_p, end_p, color, thickness: float):
        """
        Draw line use LineSegs coordinates system. Since a resolution problem is solved, the point on screen should be
        described by [horizontal ratio, vertical ratio], each of them are ranged in [-1, 1]
        :param start_p: 2d vec
        :param end_p: 2d vec
        :param color: 4d vec, line color
        :param thickness: line thickness
        """
        line_seg = LineSegs("interface")
        line_seg.setColor(*color)
        line_seg.moveTo(start_p[0] * self.w_scale, 0, start_p[1] * self.h_scale)
        line_seg.drawTo(end_p[0] * self.w_scale, 0, end_p[1] * self.h_scale)
        line_seg.setThickness(thickness)
        line_np = self.aspect2d.attachNewNode(line_seg.create(False))
        return line_np

    def remove_logo(self, task):
        alpha = self._loading_logo.getColor()[-1]
        if alpha < 0.1:
            self._loading_logo.destroy()
            return task.done
        else:
            new_alpha = alpha - 0.08
            self._loading_logo.setColor((1, 1, 1, new_alpha))
            return task.cont

    def add_line(self, start_p: Union[Vec3, Tuple], end_p: Union[Vec3, Tuple], color, thickness: float):
        line_seg = LineSegs("interface")
        line_seg.setColor(*color)
        line_seg.moveTo(start_p)
        line_seg.drawTo(end_p)
        line_seg.setThickness(thickness)
        NodePath(line_seg.create(False)).reparentTo(self.render)


if __name__ == "__main__":
    world = EngineCore({"debug": True})
    world.run()
