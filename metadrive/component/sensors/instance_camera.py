from metadrive.component.sensors.semantic_camera import SemanticCamera
import cv2
from panda3d.core import GeoMipTerrain, PNMImage
from panda3d.core import RenderState, LightAttrib, ColorAttrib, ShaderAttrib, TextureAttrib, LVecBase4, MaterialAttrib
from metadrive.constants import Semantics
from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import RENDER_MODE_NONE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import get_engine
import random


class InstanceCamera(SemanticCamera):
    CAM_MASK = CamMask.SemanticCam

    def __init__(self, width, height, engine, *, cuda=False):
        super().__init__(width, height, engine, cuda=cuda)

    def _setup_effect(self):
        """
        Use tag to apply color to different object class
        Returns: None

        """
        # setup camera

        if get_engine() is None:
            super()._setup_effect()
        else:
            mapping = get_engine().id_c
            spawned_objects = get_engine().get_objects()
            for id, obj in spawned_objects.items():
                obj.origin.setTag("id", id)
            cam = self.get_cam().node()
            cam.setTagStateKey("id")
            cam.setInitialState(
                RenderState.make(
                    ShaderAttrib.makeOff(), LightAttrib.makeAllOff(), TextureAttrib.makeOff(),
                    ColorAttrib.makeFlat((0, 0, 0, 1)), 1
                )
            )
            for id, c in mapping.items():
                cam.setTagState(id, RenderState.make(ColorAttrib.makeFlat((c[0], c[1], c[2], 1)), 1))
