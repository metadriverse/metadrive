import cv2
from panda3d.core import GeoMipTerrain, PNMImage
from panda3d.core import RenderState, LightAttrib, ColorAttrib, ShaderAttrib, TextureAttrib, LVecBase4, MaterialAttrib

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import RENDER_MODE_NONE
from metadrive.engine.asset_loader import AssetLoader


class SemanticCamera(BaseCamera):
    # shape(dim_1, dim_2)
    CAM_MASK = CamMask.SemanticCam

    GROUND_HEIGHT = -0.5
    VIEW_GROUND = False
    GROUND = None
    GROUND_MODEL = None

    # BKG_COLOR = LVecBase4(53 / 255, 81 / 255, 167 / 255, 1)

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        self.VIEW_GROUND = True  # default true
        super(SemanticCamera, self).__init__(engine, False, cuda)
        cam = self.get_cam()
        lens = self.get_lens()

        # cam.lookAt(0, 2.4, 1.3)
        cam.lookAt(0, 10.4, 1.6)

        lens.setFov(60)
        # lens.setAspectRatio(2.0)
        if self.engine.mode == RENDER_MODE_NONE or not AssetLoader.initialized():
            return

        # setup camera
        cam = cam.node()
        cam.setInitialState(RenderState.make(ShaderAttrib.makeOff(),
                                             LightAttrib.makeAllOff(),
                                             TextureAttrib.makeOff(),
                                             ColorAttrib.makeFlat((0, 0, 1, 1)), 1))
        cam.setTagStateKey("type")
        cam.setTagState("vehicle", RenderState.make(ColorAttrib.makeFlat((0, 0, 1, 1)), 1))
        cam.setTagState("ground", RenderState.make(ColorAttrib.makeFlat((1, 0, 0, 1)), 1))

        if self.VIEW_GROUND:
            ground = PNMImage(513, 513, 4)
            ground.fill(1., 1., 1.)

            self.GROUND = GeoMipTerrain("mySimpleTerrain")
            self.GROUND.setHeightfield(ground)
            self.GROUND.setAutoFlatten(GeoMipTerrain.AFMStrong)
            # terrain.setBruteforce(True)
            # # Since the terrain is a texture, shader will not calculate the sematic information, we add a moving terrain
            # # model to enable the sematic information of terrain
            self.GROUND_MODEL = self.GROUND.getRoot()
            self.GROUND_MODEL.setPos(-128, -128, self.GROUND_HEIGHT)
            self.GROUND_MODEL.reparentTo(self.engine.render)
            self.GROUND_MODEL.hide(CamMask.AllOn)
            self.GROUND_MODEL.show(CamMask.SemanticCam)
            self.GROUND_MODEL.setTag("type", "ground")
            self.GROUND.generate()

    def track(self, base_object):
        if self.VIEW_GROUND:
            pos = base_object.origin.getPos()
            self.GROUND_MODEL.setPos(pos[0], pos[1], self.GROUND_HEIGHT)
            self.GROUND_MODEL.setH(base_object.origin.getH())
            # self.GROUND_MODEL.setP(-base_object.origin.getR())
            # self.GROUND_MODEL.setR(-base_object.origin.getR())
        return super(SemanticCamera, self).track(base_object)

    def get_image(self, base_object):
        self.origin.reparentTo(base_object.origin)
        img = super(SemanticCamera, self).get_rgb_array_cpu()
        self.track(self.attached_object)
        return img

    def save_image(self, base_object, name="debug.png"):
        img = self.get_image(base_object)
        cv2.imwrite(name, img)
