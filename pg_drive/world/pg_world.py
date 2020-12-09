import logging
import os
import sys
from pg_drive.world.onscreen_message import PgOnScreenMessage
import gltf
from direct.showbase import ShowBase
from panda3d.bullet import BulletDebugNode, BulletWorld
from panda3d.core import Vec3, AntialiasAttrib, NodePath, loadPrcFileData, TextNode, LineSegs

from pg_drive.pg_config.cam_mask import CamMask
from pg_drive.pg_config.pg_config import PgConfig
from pg_drive.utils.asset_loader import AssetLoader
from pg_drive.world.force_fps import ForceFPS
from pg_drive.world.image_buffer import ImageBuffer
from pg_drive.world.light import Light
from pg_drive.world.sky_box import SkyBox
from pg_drive.world.terrain import Terrain
from pg_drive.world.vehicle_panel import VehiclePanel

root_path = os.path.dirname(os.path.dirname(__file__))


class PgWorld(ShowBase.ShowBase):
    loadPrcFileData("", "win-size 1200 900")
    loadPrcFileData("", "framebuffer-multisample 1")
    loadPrcFileData("", "multisamples 8")
    loadPrcFileData("", 'bullet-filter-algorithm groups-mask')
    loadPrcFileData("", "audio-library-name null")

    # loadPrcFileData("", " framebuffer-srgb truein")

    # loadPrcFileData("", "geom-cache-size 50000")

    # v-sync, it seems useless
    # loadPrcFileData("", "sync-video 1")

    # for debug use
    # loadPrcFileData("", "want-pstats 1")
    # loadPrcFileData("", "notify-level-glgsg fatal")

    # loadPrcFileData("", "gl-version 3 2")

    def __init__(self, config: dict = None):
        self.pg_config = self.default_config()
        if config is not None:
            self.pg_config.update(config)
        if self.pg_config["use_render"]:
            mode = "onscreen"
            loadPrcFileData("", "threading-model Cull/Draw")  # multi-thread render, accelerate simulation when evaluate
        else:
            mode = "offscreen" if self.pg_config["use_image"] else "none"
        if self.pg_config["headless_image"]:
            loadPrcFileData("", "load-display  pandagles2")
        super(PgWorld, self).__init__(windowType=mode)
        if not self.pg_config["debug_physics_world"] and (self.pg_config["use_render"] or self.pg_config["use_image"]):
            path = AssetLoader.windows_style2unix_style(root_path) if sys.platform == "win32" else root_path
            AssetLoader.init_loader(self.loader, path)
            gltf.patch_loader(self.loader)
        self.closed = False
        self.exitFunc = self.exitFunc

        # add element to render and pbr render, if is exists all the time
        self.pbr_render = self.render.attachNewNode("pbrNP")

        # add element should be cleared these node asset_path, after reset()
        self.worldNP = self.render.attachNewNode("world_np")
        self.pbr_worldNP = self.pbr_render.attachNewNode("pbrNP")  # This node is only used for render gltf model

        # some render attr
        self.light = None
        self.vehicle_panel = None

        # physics world
        self.physics_world = BulletWorld()
        self.physics_world.setGroupCollisionFlag(0, 1, True)  # detect ego car collide terrain
        self.physics_world.setGravity(Vec3(0, 0, -9.81))  # set gravity

        # init terrain
        self.terrain = Terrain()
        self.terrain.attach_to_pg_world(self.render, self.physics_world)

        # init other world elements
        if self.pg_config["use_image"] or self.pg_config["use_render"]:

            # collision info render
            self.collision_info_np = NodePath(TextNode("collision_info"))
            self._init_collision_info_render()

            from pg_drive.world.our_pbr import OurPipeline
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

            self.sky_box = SkyBox(sys.platform == "darwin")  # openGl shader didn't work for mac...
            self.sky_box.attach_to_pg_world(self.render, self.physics_world)

            self.light = Light(self.pg_config)
            self.light.attach_to_pg_world(self.render, self.physics_world)
            self.render.setLight(self.light.direction_np)
            self.render.setLight(self.light.ambient_np)

            self.render.setShaderAuto()
            self.render.setAntialias(AntialiasAttrib.MAuto)

            # ui and render property
            if self.pg_config["show_message"]:
                self.onScreenDebug.enabled = True  # only show in onscreen mode
            if self.pg_config["show_fps"]:
                self.setFrameRateMeter(True)
            self.force_fps = ForceFPS(self.pg_config["force_fps"])

            # self added display regions and cameras attached to them
            self.my_display_regions = []
            if self.pg_config["use_default_layout"]:
                self._init_display_region()
            self.my_buffers = []

            # first window and display region -- a vehicle panel
            self.vehicle_panel = VehiclePanel(self.win.makeTextureBuffer, self.makeCamera)
            self.vehicle_panel.add_to_display(
                self, [0.67, 1, self.vehicle_panel.display_bottom, self.vehicle_panel.display_top]
            )

        # task manager
        self.taskMgr.remove('audioLoop')
        self.taskMgr.remove("igLoop")

        # onscreen message
        self.on_screen_message = PgOnScreenMessage() \
            if self.pg_config["use_render"] and self.pg_config["onscreen_message"] else None

        # debug setting
        if self.pg_config["debug"] or self.pg_config["debug_physics_world"]:
            self.accept('f1', self.toggleWireframe)
            self.accept('f2', self.toggleTexture)
            self.accept('f3', self.toggleDebug)
            self.accept('f4', self.toggleAnalyze)
            self._debug_mode()

    def _init_display_region(self):
        # TODO maybe decided by the user in the future
        line_seg = LineSegs("interface")
        line_seg.setColor(0.8, 0.8, 0.8, 0)
        line_seg.moveTo(-2, 0, 0.6)
        line_seg.drawTo(2, 0, 0.6)
        line_seg.setThickness(1.5)
        NodePath(line_seg.create(False)).reparentTo(self.aspect2d)

        line_seg.moveTo(-0.455, 0, 1)
        line_seg.drawTo(-0.455, 0, 0.6)
        line_seg.setThickness(1.5)
        NodePath(line_seg.create(False)).reparentTo(self.aspect2d)

        line_seg.moveTo(0.455, 0, 1)
        line_seg.drawTo(0.455, 0, 0.6)
        line_seg.setThickness(1.5)
        NodePath(line_seg.create(False)).reparentTo(self.aspect2d)

    def _init_collision_info_render(self):
        self.collision_info_np.node().setCardActual(-7, 7, -0.3, 1)
        self.collision_info_np.node().setCardDecal(True)
        self.collision_info_np.node().setTextColor(1, 1, 1, 1)
        self.collision_info_np.node().setAlign(TextNode.A_center)
        self.collision_info_np.setScale(0.05)
        self.collision_info_np.setPos(-1, -0.8, -0.8)
        self.collision_info_np.reparentTo(self.aspect2d)

    def render_frame(self, text: dict = None):
        if self.on_screen_message is not None:
            self.on_screen_message.update_data(text)
            self.on_screen_message.render()
        self.graphicsEngine.renderFrame()
        if self.pg_config["use_render"]:
            with self.force_fps:
                self.sky_box.step()

    def clear_world(self):
        """
        Call me to setup the whole world after _init_
        """
        # attach all node to this node asset_path
        self.worldNP.node().removeAllChildren()
        self.pbr_worldNP.node().removeAllChildren()
        if self.pg_config["debug_physics_world"]:
            self.addTask(self.report_body_nums, "report_num")

    def _clear_display_region_and_buffers(self):
        for r in self.my_display_regions:
            self.win.removeDisplayRegion(r)
        for my_buffer in self.my_buffers:
            self.graphicsEngine.removeWindow(my_buffer.buffer)
            if my_buffer.cam in self.camList:
                self.camList.remove(my_buffer.cam)
        self.my_display_regions = []
        self.my_buffers = []

    @staticmethod
    def default_config():
        return PgConfig(
            dict(
                debug=False,
                use_render=False,
                use_image=False,
                physics_world_step_size=2e-2,
                chase_camera=True,
                direction_light=True,
                ambient_light=True,
                show_fps=True,
                show_message=True,  # show message when render is called
                mini_map=True,
                force_fps=None,
                debug_physics_world=False,  # only render physics world without model
                use_default_layout=True,  # decide the layout of white lines
                headless_image=False,  # set to true only when on headless machine and use rgb image!!!!!!
                onscreen_message=True
            )
        )

    def step(self):
        dt = self.pg_config["physics_world_step_size"]
        self.physics_world.doPhysics(dt, 1, dt)

    def _debug_mode(self):
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(True)
        debugNP = self.render.attachNewNode(debugNode)
        debugNP.show()
        self.physics_world.setDebugNode(debugNP.node())
        self.debugnode = debugNP

    def toggleAnalyze(self):
        self.worldNP.analyze()
        # self.worldNP.ls()

    def toggleDebug(self):
        if self.debugnode.isHidden():
            self.debugnode.show()
        else:
            self.debugnode.hide()

    def report_body_nums(self, task):
        logging.debug(
            "Body Nums: {}".format(
                self.physics_world.getNumRigidBodies() + self.physics_world.getNumGhosts() +
                self.physics_world.getNumVehicles()
            )
        )
        return task.done

    def close_world(self):
        if self.pg_config["use_render"] or self.pg_config["use_image"]:
            self._clear_display_region_and_buffers()
        self.destroy()
        self.physics_world.clearDebugNode()
        self.physics_world.clearContactAddedCallback()
        self.physics_world.clearFilterCallback()

        # del self.physics_world  # Will cause error if del it.
        self.physics_world = None


if __name__ == "__main__":
    world = PgWorld()
    world.run()
