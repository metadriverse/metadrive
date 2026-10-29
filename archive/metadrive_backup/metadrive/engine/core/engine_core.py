import os
import gc
from panda3d.core import TexturePool, ModelPool
import sys
import time
from typing import Optional, Union, Tuple
from panda3d.core import Material, LVecBase4
import gltf
from metadrive.third_party.simplepbr import init
from direct.gui.OnscreenImage import OnscreenImage
from direct.showbase import ShowBase
from panda3d.bullet import BulletDebugNode
from panda3d.core import AntialiasAttrib, loadPrcFileData, LineSegs, PythonCallbackObject, Vec3, NodePath
from panda3d.core import GraphicsPipeSelection

from metadrive.component.sensors.base_sensor import BaseSensor
from metadrive.constants import RENDER_MODE_OFFSCREEN, RENDER_MODE_NONE, RENDER_MODE_ONSCREEN, EDITION, CamMask, \
    BKG_COLOR
from metadrive.engine.asset_loader import initialize_asset_loader, close_asset_loader, randomize_cover, get_logo_file
from metadrive.engine.core.collision_callback import collision_callback
from metadrive.engine.core.draw import ColorLineNodePath, ColorSphereNodePath
from metadrive.engine.core.force_fps import ForceFPS
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.engine.core.light import Light
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.engine.core.pssm import PSSM
from metadrive.engine.core.sky_box import SkyBox
from metadrive.engine.core.terrain import Terrain
from metadrive.engine.logger import get_logger
from metadrive.utils.utils import is_mac, setup_logger
import logging
import subprocess
from metadrive.utils.utils import is_port_occupied

logger = get_logger()


def _suppress_warning():
    loadPrcFileData("", "notify-level-glgsg fatal")
    loadPrcFileData("", "notify-level-pgraph fatal")
    loadPrcFileData("", "notify-level-pnmimage fatal")
    loadPrcFileData("", "notify-level-task fatal")
    loadPrcFileData("", "notify-level-thread fatal")
    loadPrcFileData("", "notify-level-device fatal")
    loadPrcFileData("", "notify-level-bullet fatal")
    loadPrcFileData("", "notify-level-display fatal")
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)


def _free_warning():
    loadPrcFileData("", "notify-level-glgsg debug")
    # loadPrcFileData("", "notify-level-pgraph debug")  # press 4 to use toggle analyze to do this
    loadPrcFileData("", "notify-level-display debug")  # press 4 to use toggle analyze to do this
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
    global_config = None  # global config can exist before engine initialization
    loadPrcFileData("", "window-title {}".format(EDITION))
    loadPrcFileData("", "bullet-filter-algorithm groups-mask")
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "model-cache-compressed-textures 1")
    loadPrcFileData("", "textures-power-2 none")

    # loadPrcFileData("", "transform-cache 0")
    # loadPrcFileData("", "state-cache 0")
    loadPrcFileData("", "garbage-collect-states 0")
    loadPrcFileData("", "print-pipe-types 0")

    if is_mac():
        # latest macOS supported openGL version
        loadPrcFileData("", "gl-version 4 1")
        loadPrcFileData("", "framebuffer-multisample 1")
        loadPrcFileData("", "multisamples 4")
    else:
        loadPrcFileData("", "framebuffer-multisample 1")
        loadPrcFileData("", "multisamples 8")

    # loadPrcFileData("", "allow-incomplete-render #t")
    # loadPrcFileData("", "# even-animation #t")

    # loadPrcFileData("", " framebuffer-srgb truein")
    # loadPrcFileData("", "geom-cache-size 50000")

    # v-sync, it seems useless
    # loadPrcFileData("", "sync-video 1")

    # for debug use
    # loadPrcFileData("", "gl-version 3 2")

    def __init__(self, global_config):
        # if EngineCore.global_config is not None:
        #     # assert global_config is EngineCore.global_config, \
        #     #     "No allowed to change ptr of global config, which may cause issue"
        #     pass
        # else:
        config = global_config
        self.main_window_disabled = False
        if "main_camera" not in global_config["sensors"]:
            # reduce size as we don't use the main camera content for improving efficiency
            config["window_size"] = (1, 1)
            self.main_window_disabled = True

        self.pid = os.getpid()
        EngineCore.global_config = global_config
        self.mode = global_config["_render_mode"]
        self.pstats_process = None
        if self.global_config["pstats"]:
            # pstats debug provided by panda3d
            loadPrcFileData("", "want-pstats 1")
            if not is_port_occupied(5185):
                self.pstats_process = subprocess.Popen(['pstats'])
                logger.info(
                    "pstats is launched successfully, tutorial is at: "
                    "https://docs.panda3d.org/1.10/python/optimization/using-pstats"
                )
            else:
                logger.warning(
                    "pstats is already launched! tutorial is at: "
                    "https://docs.panda3d.org/1.10/python/optimization/using-pstats"
                )

        # Setup onscreen render
        if self.global_config["use_render"]:
            assert self.mode == RENDER_MODE_ONSCREEN, "Render mode error"
            # Warning it may cause memory leak, Pand3d Official has fixed this in their master branch.
            # You can enable it if your panda version is latest.
            if self.global_config["multi_thread_render"] and not self.use_render_pipeline:
                # multi-thread render, accelerate simulation
                loadPrcFileData("", "threading-model {}".format(self.global_config["multi_thread_render_mode"]))
        else:
            self.global_config["show_coordinates"] = False
            if self.global_config["image_observation"]:
                assert self.mode == RENDER_MODE_OFFSCREEN, "Render mode error"
                if self.global_config["multi_thread_render"] and not self.use_render_pipeline:
                    # render-pipeline can not work with multi-thread rendering
                    loadPrcFileData("", "threading-model {}".format(self.global_config["multi_thread_render_mode"]))
            else:
                assert self.mode == RENDER_MODE_NONE, "Render mode error"
                if self.global_config["show_interface"]:
                    # Disable useless camera capturing in none mode
                    self.global_config["show_interface"] = False

        loadPrcFileData("", "win-size {} {}".format(*self.global_config["window_size"]))

        if self.use_render_pipeline:
            from metadrive.render_pipeline.rpcore import RenderPipeline
            from metadrive.render_pipeline.rpcore.rpobject import RPObject
            if not self.global_config["debug"]:
                RPObject.set_output_level("warning")  # warning
            self.render_pipeline = RenderPipeline()
            self.render_pipeline.pre_showbase_init()
            # disable it, as some model errors happen!
            loadPrcFileData("", "gl-immutable-texture-storage 0")
        else:
            self.render_pipeline = None

        # Setup some debug options
        # if self.global_config["headless_machine_render"]:
        #     # headless machine support
        #     # loadPrcFileData("", "load-display  pandagles2")
        #     # no further actions will be applied now!
        #     pass
        if self.global_config["debug"]:
            # debug setting
            EngineCore.DEBUG = True
            if self.global_config["debug_panda3d"]:
                _free_warning()
            setup_logger(debug=True)
            self.accept("1", self.toggleDebug)
            self.accept("2", self.toggleWireframe)
            self.accept("3", self.toggleTexture)
            self.accept("4", self.toggleAnalyze)
            self.accept("5", self.reload_shader)
        else:
            # only report fatal error when debug is False
            _suppress_warning()
            # a special debug mode
            if self.global_config["debug_physics_world"]:
                self.accept("1", self.toggleDebug)
                self.accept("4", self.toggleAnalyze)

        if not self.global_config["disable_model_compression"] and not self.use_render_pipeline:
            loadPrcFileData("", "compressed-textures 1")  # Default to compress

        super(EngineCore, self).__init__(windowType=self.mode)
        logger.info("Known Pipes: {}".format(*GraphicsPipeSelection.getGlobalPtr().getPipeTypes()))
        if self.main_window_disabled and self.mode != RENDER_MODE_NONE:
            self.win.setActive(False)

        self._all_panda_tasks = self.taskMgr.getAllTasks()
        if self.use_render_pipeline and self.mode != RENDER_MODE_NONE:
            self.render_pipeline.create(self)

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
                logger.warning(
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

        if not self.global_config["debug_physics_world"] \
                and (self.mode in [RENDER_MODE_ONSCREEN, RENDER_MODE_OFFSCREEN]):
            initialize_asset_loader(self)
            try:
                gltf.patch_loader(self.loader)
            except:
                pass
            if not self.use_render_pipeline:
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

        # attach node to this root whose children nodes will be clear after calling clear_world()
        self.worldNP = self.render.attachNewNode("world_np")
        self.origin = self.worldNP

        self.debug_node = None

        # some render attribute
        self.pbrpipe = None
        self.world_light = None
        self.common_filter = None

        # physics world
        self.physics_world = PhysicsWorld(
            self.global_config["debug_static_world"], disable_collision=self.global_config["disable_collision"]
        )

        # collision callback
        self.physics_world.dynamic_world.setContactAddedCallback(PythonCallbackObject(collision_callback))

        # for real time simulation
        self.force_fps = ForceFPS(self)

        # init terrain
        self.terrain = Terrain(self.global_config["show_terrain"], self)
        # self.terrain.attach_to_world(self.render, self.physics_world)

        self.sky_box = None

        # init other world elements
        if self.mode != RENDER_MODE_NONE:
            if self.use_render_pipeline:
                self.pbrpipe = None
                if self.global_config["daytime"] is not None:
                    self.render_pipeline.daytime_mgr.time = self.global_config["daytime"]
            else:
                if is_mac():
                    self.pbrpipe = init(
                        msaa_samples=4,
                        # use_hardware_skinning=True,
                        use_330=True
                    )
                else:
                    self.pbrpipe = init(
                        msaa_samples=16,
                        use_hardware_skinning=True,
                        # use_normal_maps=True,
                        use_330=False
                    )

                self.sky_box = SkyBox(not self.global_config["show_skybox"])
                self.sky_box.attach_to_world(self.render, self.physics_world)

                self.world_light = Light()
                self.world_light.attach_to_world(self.render, self.physics_world)
                self.render.setLight(self.world_light.direction_np)
                self.render.setLight(self.world_light.ambient_np)
                self.render.setAntialias(AntialiasAttrib.MAuto)

                # setup pssm shadow
                # init shadow if required
                # if not self.global_config["debug_physics_world"]:
                self.pssm = PSSM(self)
                self.pssm.init()

            # set main cam
            self.cam.node().setCameraMask(CamMask.MainCam)
            self.cam.node().getDisplayRegion(0).setClearColorActive(True)
            self.cam.node().getDisplayRegion(0).setClearColor(BKG_COLOR)

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
        self.taskMgr.remove("audioLoop")

        # show coordinate
        self.coordinate_line = []
        if self.global_config["show_coordinates"]:
            self.show_coordinates()

        # create sensors
        self.sensors = {}
        self.setup_sensors()

        # toggle in debug physics world
        if self.global_config["debug_physics_world"]:
            self.toggleDebug()

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
        if self.mode != RENDER_MODE_NONE and self.sky_box is not None:
            self.sky_box.step()

    def step_physics_world(self):
        dt = self.global_config["physics_world_step_size"]
        self.physics_world.dynamic_world.doPhysics(dt, 1, dt)

    def _debug_mode(self):
        debugNode = BulletDebugNode("Debug")
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(True)
        debugNP = self.render.attachNewNode(debugNode)
        self.physics_world.dynamic_world.setDebugNode(debugNP.node())
        self.debug_node = debugNP
        # TODO: debugNP is not registered.

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
        logger.debug(self.physics_world.report_bodies())
        return task.done

    def close_engine(self):
        self.taskMgr.stop()
        logger.debug(
            "Before del taskMgr: task_chain_num={}, all_tasks={}".format(
                self.taskMgr.mgr.getNumTaskChains(), self.taskMgr.getAllTasks()
            )
        )
        for tsk in self.taskMgr.getAllTasks():
            if tsk not in self._all_panda_tasks:
                self.taskMgr.remove(tsk)
        logger.debug(
            "After del taskMgr: task_chain_num={}, all_tasks={}".format(
                self.taskMgr.mgr.getNumTaskChains(), self.taskMgr.getAllTasks()
            )
        )
        if hasattr(self, "_window_logo"):
            self._window_logo.removeNode()
        self.terrain.destroy()
        if self.sky_box:
            self.sky_box.destroy()
        self.physics_world.dynamic_world.clearContactAddedCallback()
        self.physics_world.destroy()
        self.destroy()
        close_asset_loader()
        # EngineCore.global_config.clear()
        EngineCore.global_config = None

        import sys
        if sys.version_info >= (3, 0):
            import builtins
        else:
            import __builtin__ as builtins
        if hasattr(builtins, "base"):
            del builtins.base
        from metadrive.base_class.base_object import BaseObject

        def find_all_subclasses(cls):
            """Find all subclasses of a given class, including subclasses of subclasses."""
            subclasses = cls.__subclasses__()
            for subclass in subclasses:
                # Recursively find subclasses of the current subclass
                subclasses.extend(find_all_subclasses(subclass))
            return subclasses

        gc.collect()
        all_classes = find_all_subclasses(BaseObject)
        for cls in all_classes:
            if hasattr(cls, "MODEL"):
                cls.MODEL = None
            elif hasattr(cls, "model_collections"):
                for node_path in cls.model_collections.values():
                    node_path.removeNode()
                cls.model_collections = {}
            elif hasattr(cls, "_MODEL"):
                for node_path in cls._MODEL.values():
                    node_path.cleanup()
                cls._MODEL = {}
            elif hasattr(cls, "TRAFFIC_LIGHT_MODEL"):
                for node_path in cls.TRAFFIC_LIGHT_MODEL.values():
                    node_path.removeNode()
                cls.TRAFFIC_LIGHT_MODEL = {}
        gc.collect()
        ModelPool.releaseAllModels()
        TexturePool.releaseAllTextures()

    def clear_world(self):
        self.worldNP.removeNode()

    def toggle_help_message(self):
        if self.on_screen_message:
            self.on_screen_message.toggle_help_message()

    def draw_line_2d(self, start_p, end_p, color, thickness: float):
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
        # TODO: line_np is not registered.
        return line_np

    def remove_logo(self, task):
        alpha = self._loading_logo.getColor()[-1]
        if alpha < 0.1:
            self._loading_logo.destroy()
            self._loading_logo = None
            return task.done
        else:
            new_alpha = alpha - 0.08
            self._loading_logo.setColor((1, 1, 1, new_alpha))
            return task.cont

    def _draw_line_3d(self, start_p: Union[Vec3, Tuple], end_p: Union[Vec3, Tuple], color, thickness: float):
        """
        This API is not official
        Args:
            start_p:
            end_p:
            color:
            thickness:

        Returns:

        """
        start_p = [*start_p]
        end_p = [*end_p]
        start_p[1] *= 1
        end_p[1] *= 1
        line_seg = LineSegs("interface")
        line_seg.moveTo(Vec3(*start_p))
        line_seg.drawTo(Vec3(*end_p))
        line_seg.setThickness(thickness)
        np = NodePath(line_seg.create(False))
        material = Material()
        material.setBaseColor(LVecBase4(*color[:3], 1))
        np.setMaterial(material, True)
        return np

    def make_line_drawer(self, parent_node=None, thickness=1.0):
        if parent_node is None:
            parent_node = self.render
        drawer = ColorLineNodePath(parent_node, thickness=thickness)
        return drawer

    def make_point_drawer(self, parent_node=None, scale=1.0):
        if parent_node is None:
            parent_node = self.render
        drawer = ColorSphereNodePath(parent_node, scale=scale)
        return drawer

    def show_coordinates(self):
        if len(self.coordinate_line) > 0:
            return
        # x direction = red
        np_x = self._draw_line_3d(Vec3(0, 0, 0.1), Vec3(100, 0, 0.1), color=[1, 0, 0, 1], thickness=3)
        np_x.reparentTo(self.render)
        # y direction = blue
        np_y = self._draw_line_3d(Vec3(0, 0, 0.1), Vec3(0, 50, 0.1), color=[0, 1, 0, 1], thickness=3)
        np_y.reparentTo(self.render)

        np_y.hide(CamMask.AllOn)
        np_y.show(CamMask.MainCam)

        np_x.hide(CamMask.AllOn)
        np_x.show(CamMask.MainCam)

        self.coordinate_line.append(np_x)
        self.coordinate_line.append(np_y)

    def remove_coordinates(self):
        for line in self.coordinate_line:
            line.detachNode()
            line.removeNode()

    def set_coordinates_indicator_pos(self, pos):
        if len(self.coordinate_line) == 0:
            return
        for line in self.coordinate_line:
            line.setPos(pos[0], pos[1], 0)

    @property
    def use_render_pipeline(self):
        return self.global_config["render_pipeline"] and not self.mode == RENDER_MODE_NONE

    def reload_shader(self):
        if self.render_pipeline is not None:
            self.render_pipeline.reload_shaders()

    def setup_sensors(self):
        for sensor_id, sensor_cfg in self.global_config["sensors"].items():
            if sensor_id == "main_camera":
                # It is added when initializing main_camera
                continue
            if sensor_id in self.sensors:
                raise ValueError("Sensor id {} is duplicated!".format(sensor_id))
            cls = sensor_cfg[0]
            args = sensor_cfg[1:]
            assert issubclass(cls, BaseSensor), "{} is not a subclass of BaseSensor".format(cls.__name__)
            if issubclass(cls, ImageBuffer):
                self.add_image_sensor(sensor_id, cls, args)
            else:
                self.sensors[sensor_id] = cls(*args, self)

    def get_sensor(self, sensor_id):
        if sensor_id not in self.sensors:
            raise ValueError("Can not get {}, available sensors: {}".format(sensor_id, self.sensors.keys()))
        return self.sensors[sensor_id]

    def add_image_sensor(self, name: str, cls, args):
        sensor = cls(*args, engine=self, cuda=self.global_config["image_on_cuda"])
        assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
        self.sensors[name] = sensor
