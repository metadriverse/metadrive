import cv2
from panda3d.core import GeoMipTerrain, PNMImage
from panda3d.core import RenderState, LightAttrib, ColorAttrib, ShaderAttrib, TextureAttrib, LVecBase4, MaterialAttrib
from metadrive.constants import Semantics
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

    frame_buffer_rgb_bits = (8, 8, 8, 8)

    # BKG_COLOR = LVecBase4(53 / 255, 81 / 255, 167 / 255, 1)

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        self.VIEW_GROUND = True  # default true
        super(SemanticCamera, self).__init__(engine, cuda)
        cam = self.get_cam()
        lens = self.get_lens()

        # cam.lookAt(0, 2.4, 1.3)
        cam.lookAt(0, 10.4, 1.6)

        lens.setFov(60)
        # lens.setAspectRatio(2.0)
        if self.engine.mode == RENDER_MODE_NONE or not AssetLoader.initialized():
            return

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
            self.GROUND_MODEL.setTag("type", Semantics.ROAD.label)
            self.GROUND.generate()

    def track(self, base_object):
        if self.VIEW_GROUND and base_object is not None:
            pos = base_object.origin.getPos()
            self.GROUND_MODEL.setPos(pos[0], pos[1], self.GROUND_HEIGHT)
            self.GROUND_MODEL.setH(base_object.origin.getH())
            # self.GROUND_MODEL.setP(-base_object.origin.getR())
            # self.GROUND_MODEL.setR(-base_object.origin.getR())
        return super(SemanticCamera, self).track(base_object)

    def _setup_effect(self):
        """
        Use tag to apply color to different object class
        Returns: None

        """
        # setup camera
        cam = self.get_cam().node()
        cam.setInitialState(
            RenderState.make(
                ShaderAttrib.makeOff(), LightAttrib.makeAllOff(), TextureAttrib.makeOff(),
                ColorAttrib.makeFlat((0, 0, 1, 1)), 1
            )
        )
        cam.setTagStateKey("type")
        for t in [v for v, m in vars(Semantics).items() if not (v.startswith('_') or callable(m))]:
            label, c = getattr(Semantics, t)
            cam.setTagState(label, RenderState.make(ColorAttrib.makeFlat((c[0] / 255, c[1] / 255, c[2] / 255, 1)), 1))

    def _create_buffer(self, width, height, frame_buffer_property):
        """
        The buffer should be created without frame_buffer_property
        Args:
            width: Image width
            height: Image height
            frame_buffer_property: disabled in Semantic Camera

        Returns: Buffer object

        """
        return self.engine.win.makeTextureBuffer("camera", width, height)
