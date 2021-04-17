import logging
import time
from typing import Optional, Union

import gltf
from direct.gui.OnscreenImage import OnscreenImage
from direct.showbase import ShowBase
from panda3d.bullet import BulletDebugNode
from panda3d.core import AntialiasAttrib, loadPrcFileData, LineSegs, PythonCallbackObject
from pgdrive.constants import RENDER_MODE_OFFSCREEN, RENDER_MODE_NONE, RENDER_MODE_ONSCREEN, PG_EDITION, CamMask
from pgdrive.utils import is_mac, setup_logger
from pgdrive.utils.asset_loader import AssetLoader, initialize_asset_loader
from pgdrive.world.collision_callback import pg_collision_callback
from pgdrive.world.force_fps import ForceFPS
from pgdrive.world.image_buffer import ImageBuffer
from pgdrive.world.light import Light
from pgdrive.world.onscreen_message import PGOnScreenMessage
from pgdrive.world.pg_physics_world import PGPhysicsWorld
from pgdrive.world.sky_box import SkyBox
from pgdrive.world.terrain import Terrain


def _suppress_warning():
    loadPrcFileData("", "notify-level-glgsg fatal")
    loadPrcFileData("", "notify-level-pgraph fatal")
    loadPrcFileData("", "notify-level-pnmimage fatal")
    loadPrcFileData("", "notify-level-thread fatal")


def _free_warning():
    loadPrcFileData("", "notify-level-glgsg debug")
    # loadPrcFileData("", "notify-level-pgraph debug")  # press 4 to use toggle analyze to do this
    loadPrcFileData("", "notify-level-pnmimage debug")
    loadPrcFileData("", "notify-level-thread debug")


class PGWorld(ShowBase.ShowBase):
    DEBUG = False
    loadPrcFileData("", "window-title {}".format(PG_EDITION))
    loadPrcFileData("", "framebuffer-multisample 1")
    loadPrcFileData("", "multisamples 8")
    loadPrcFileData("", 'bullet-filter-algorithm groups-mask')
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "model-cache-compressed-textures 1")

    # loadPrcFileData("", " framebuffer-srgb truein")
    # loadPrcFileData("", "geom-cache-size 50000")

    # v-sync, it seems useless
    # loadPrcFileData("", "sync-video 1")

    # for debug use
    # loadPrcFileData("", "gl-version 3 2")

    def __init__(self, config: dict = None):
        # Setup config and Panda3d
        self.world_config = config
        # self.world_config = merge_config(self.default_config(), config)
        # if config is not None:
        #     self.world_config.merge(config)
        if self.world_config["pstats"]:
            # pstats debug provided by panda3d
            loadPrcFileData("", "want-pstats 1")

        loadPrcFileData("", "win-size {} {}".format(*self.world_config["window_size"]))

        # Setup onscreen render
        if self.world_config["use_render"]:
            self.mode = RENDER_MODE_ONSCREEN
            # Warning it may cause memory leak, Pand3d Official has fixed this in their master branch.
            # You can enable it if your panda version is latest.
            loadPrcFileData("", "threading-model Cull/Draw")  # multi-thread render, accelerate simulation when evaluate
        else:
            if self.world_config["use_image"]:
                self.mode = RENDER_MODE_OFFSCREEN
                loadPrcFileData("", "threading-model Cull/Draw")
            else:
                self.mode = RENDER_MODE_NONE

        if is_mac() and (self.mode == RENDER_MODE_OFFSCREEN):  # Mac don't support offscreen rendering
            self.mode = RENDER_MODE_ONSCREEN

        # Setup some debug options
        if self.world_config["headless_image"]:
            # headless machine support
            loadPrcFileData("", "load-display  pandagles2")
        if self.world_config["debug"]:
            # debug setting
            PGWorld.DEBUG = True
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
            if self.world_config["debug_physics_world"]:
                self.accept('1', self.toggleDebug)
                self.accept('4', self.toggleAnalyze)

        super(PGWorld, self).__init__(windowType=self.mode)

        # Change window size at runtime if screen too small
        # assert int(self.world_config["use_topdown"]) + int(self.world_config["use_image"]) <= 1, (
        #     "Only one of use_topdown and use_image options can be selected."
        # )

        # main_window_position = (0, 0)
        if self.mode == RENDER_MODE_ONSCREEN:
            if self.world_config["fast_launch_window"]:
                pass
            else:
                loadPrcFileData("", "compressed-textures 1")  # Default to compress
            h = self.pipe.getDisplayHeight()
            w = self.pipe.getDisplayWidth()
            if self.world_config["window_size"][0] > 0.9 * w or self.world_config["window_size"][1] > 0.9 * h:
                old_scale = self.world_config["window_size"][0] / self.world_config["window_size"][1]
                new_w = int(min(0.9 * w, 0.9 * h * old_scale))
                new_h = int(min(0.9 * h, 0.9 * w / old_scale))
                self.world_config["window_size"] = tuple([new_w, new_h])
                from panda3d.core import WindowProperties
                props = WindowProperties()
                props.setSize(self.world_config["window_size"][0], self.world_config["window_size"][1])
                self.win.requestProperties(props)
                logging.warning(
                    "Since your screen is too small ({}, {}), we resize the window to {}.".format(
                        w, h, self.world_config["window_size"]
                    )
                )
            # main_window_position = (
            #     (w - self.world_config["window_size"][0]) / 2, (h - self.world_config["window_size"][1]) / 2
            # )

        # self.highway_render = None
        # if self.world_config["use_topdown"]:
        #     self.highway_render = HighwayRender(self.world_config["use_render"], main_window_position)

        # screen scale factor
        self.w_scale = max(self.world_config["window_size"][0] / self.world_config["window_size"][1], 1)
        self.h_scale = max(self.world_config["window_size"][1] / self.world_config["window_size"][0], 1)

        if self.mode == RENDER_MODE_ONSCREEN:
            self.disableMouse()

        if not self.world_config["debug_physics_world"] and (self.mode in [RENDER_MODE_ONSCREEN, RENDER_MODE_OFFSCREEN
                                                                           ]):
            initialize_asset_loader(self)
            gltf.patch_loader(self.loader)

            # Display logo
            if self.mode == RENDER_MODE_ONSCREEN and (not self.world_config["debug"]) \
                        and (not self.world_config["fast_launch_window"]):
                self._loading_logo = OnscreenImage(
                    image=AssetLoader.file_path("PGDrive-large.png"),
                    pos=(0, 0, 0),
                    scale=(self.w_scale, 1, self.h_scale)
                )
                self._loading_logo.setTransparency(True)
                for i in range(20):
                    self.graphicsEngine.renderFrame()
                self.taskMgr.add(self.remove_logo, "remove _loading_logo in first frame")

        self.closed = False

        # add element to render and pbr render, if is exists all the time.
        # these element will not be removed when clear_world() is called
        self.pbr_render = self.render.attachNewNode("pbrNP")

        # attach node to this root root whose children nodes will be clear after calling clear_world()
        self.worldNP = self.render.attachNewNode("world_np")

        # same as worldNP, but this node is only used for render gltf model with pbr material
        self.pbr_worldNP = self.pbr_render.attachNewNode("pbrNP")
        self.debug_node = None

        # some render attribute
        self.pbrpipe = None
        self.light = None

        # physics world
        self.physics_world = PGPhysicsWorld()

        # collision callback
        self.physics_world.dynamic_world.setContactAddedCallback(PythonCallbackObject(pg_collision_callback))

        # for real time simulation
        self.force_fps = ForceFPS(self, start=True)

        # init terrain
        self.terrain = Terrain()
        self.terrain.attach_to_pg_world(self.render, self.physics_world)

        # init other world elements
        if self.mode != RENDER_MODE_NONE:

            from pgdrive.world.our_pbr import OurPipeline
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
            self.cam.node().getDisplayRegion(0).setClearColor(ImageBuffer.BKG_COLOR)
            lens = self.cam.node().getLens()
            lens.setFov(70)
            lens.setAspectRatio(1.2)

            self.sky_box = SkyBox()
            self.sky_box.attach_to_pg_world(self.render, self.physics_world)

            self.light = Light(self.world_config)
            self.light.attach_to_pg_world(self.render, self.physics_world)
            self.render.setLight(self.light.direction_np)
            self.render.setLight(self.light.ambient_np)

            self.render.setShaderAuto()
            self.render.setAntialias(AntialiasAttrib.MAuto)

            # ui and render property
            if self.world_config["show_fps"]:
                self.setFrameRateMeter(True)

            # onscreen message
            self.on_screen_message = PGOnScreenMessage(
                debug=self.DEBUG
            ) if self.mode == RENDER_MODE_ONSCREEN and self.world_config["onscreen_message"] else None
            self._show_help_message = False
            self._episode_start_time = time.time()

            self.accept("h", self.toggle_help_message)
            self.accept("f", self.force_fps.toggle)

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
            self.on_screen_message.update_data(text)
            self.on_screen_message.render()
        if self.mode == RENDER_MODE_ONSCREEN:
            self.sky_box.step()
        # if self.highway_render is not None:
        #     self.highway_render.render()

    def clear_world(self):
        """
        Call me to setup the whole visualization world after _init_
        """
        # attach all node to this node asset_path
        self.worldNP.node().removeAllChildren()
        self.pbr_worldNP.node().removeAllChildren()
        if self.world_config["debug_physics_world"]:
            self.addTask(self.report_body_nums, "report_num")

        self._episode_start_time = time.time()

    def step(self):
        dt = self.world_config["physics_world_step_size"]
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
        # while self.taskMgr.getAllTasks():
        #     time.sleep(0.1)
        self.physics_world.destroy()
        self.destroy()

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
        :param pg_world: pg_world class
        :return:
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


if __name__ == "__main__":
    world = PGWorld({"debug": True})
    world.run()
