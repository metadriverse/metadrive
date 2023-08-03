import cv2
from panda3d.core import Shader, RenderState, ShaderAttrib, GeoMipTerrain

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import RENDER_MODE_NONE
from metadrive.engine.asset_loader import AssetLoader


class DepthCamera(BaseCamera):
    # shape(dim_1, dim_2)
    CAM_MASK = CamMask.DepthCam

    GROUND_HEIGHT = -0.5
    VIEW_GROUND = False
    GROUND = None
    GROUND_MODEL = None

    def __init__(self, engine, width, height, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        self.VIEW_GROUND = True  # default true
        super(DepthCamera, self).__init__(engine, False, cuda)
        cam = self.get_cam()
        lens = self.get_lens()

        # cam.lookAt(0, 2.4, 1.3)
        cam.lookAt(0, 10.4, 1.6)

        lens.setFov(60)
        # lens.setAspectRatio(2.0)
        if self.engine.mode == RENDER_MODE_NONE or not AssetLoader.initialized():
            return
        # add shader for it
        # if get_global_config()["headless_machine_render"]:
        #     vert_path = AssetLoader.file_path("shaders", "depth_cam_gles.vert.glsl")
        #     frag_path = AssetLoader.file_path("shaders", "depth_cam_gles.frag.glsl")
        # else:
        from metadrive.utils import is_mac
        if is_mac():
            vert_path = AssetLoader.file_path("shaders", "depth_cam_mac.vert.glsl")
            frag_path = AssetLoader.file_path("shaders", "depth_cam_mac.frag.glsl")
        else:
            vert_path = AssetLoader.file_path("shaders", "depth_cam.vert.glsl")
            frag_path = AssetLoader.file_path("shaders", "depth_cam.frag.glsl")
        custom_shader = Shader.load(Shader.SL_GLSL, vertex=vert_path, fragment=frag_path)
        cam.node().setInitialState(RenderState.make(ShaderAttrib.make(custom_shader, 1)))

        if self.VIEW_GROUND:
            self.GROUND = GeoMipTerrain("mySimpleTerrain")
            self.GROUND.setHeightfield(AssetLoader.file_path("textures", "height_map.png"))
            self.GROUND.setAutoFlatten(GeoMipTerrain.AFMStrong)
            # terrain.setBruteforce(True)
            # # Since the terrain is a texture, shader will not calculate the depth information, we add a moving terrain
            # # model to enable the depth information of terrain
            self.GROUND_MODEL = self.GROUND.getRoot()
            self.GROUND_MODEL.setPos(-128, -128, self.GROUND_HEIGHT)
            self.GROUND_MODEL.reparentTo(self.engine.render)
            self.GROUND_MODEL.hide(CamMask.AllOn)
            self.GROUND_MODEL.show(CamMask.DepthCam)
            self.GROUND.generate()

    def track(self, base_object):
        if self.VIEW_GROUND:
            pos = base_object.origin.getPos()
            self.GROUND_MODEL.setPos(pos[0], pos[1], self.GROUND_HEIGHT)
            # self.GROUND_MODEL.setP(-base_object.origin.getR())
            # self.GROUND_MODEL.setR(-base_object.origin.getR())
        return super(DepthCamera, self).track(base_object)

    def get_image(self, base_object):
        self.origin.reparentTo(base_object.origin)
        img = super(DepthCamera, self).get_rgb_array()
        self.track(self.attached_object)
        return img

    def save_image(self, base_object, name="debug.png"):
        img = self.get_image(base_object)
        cv2.imwrite(name, img)
