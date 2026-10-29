from typing import Union

import numpy as np
from panda3d.core import NodePath
from panda3d.core import RenderState, LightAttrib, ColorAttrib, ShaderAttrib, TextureAttrib

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import CameraTagStateKey
from metadrive.engine.engine_utils import get_engine


class InstanceCamera(BaseCamera):
    CAM_MASK = CamMask.SemanticCam

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W = width
        self.BUFFER_H = height
        super().__init__(engine, cuda)

    def track(self, new_parent_node, position, hpr):
        """
        See BaseCamera.track
        """
        self._setup_effect()
        super().track(new_parent_node, position, hpr)

    def perceive(
        self, to_float=True, new_parent_node: Union[NodePath, None] = None, position=None, hpr=None
    ) -> np.ndarray:
        self._setup_effect()
        return super().perceive(to_float, new_parent_node, position, hpr)

    def _setup_effect(self):
        """
        Use tag to apply color to different objects (instances)
        Returns: None
        """
        # setup camera

        if get_engine() is None:
            super()._setup_effect()
        else:
            mapping = get_engine().id_c
            spawned_objects = get_engine().get_objects()

            ##Ensure consistency between color mapping and the objects actually active in the engine.
            mapping_set, object_set = set(list(mapping.keys())), set(list(spawned_objects.keys()))
            assert (len(mapping_set.difference(object_set)) == 0)

            for id, obj in spawned_objects.items():
                obj.origin.setTag(CameraTagStateKey.ID, id)
            cam = self.get_cam().node()
            cam.setTagStateKey(CameraTagStateKey.ID)
            cam.setInitialState(
                RenderState.make(
                    ShaderAttrib.makeOff(), LightAttrib.makeAllOff(), TextureAttrib.makeOff(),
                    ColorAttrib.makeFlat((0, 0, 0, 1)), 1
                )
            )

            for id in spawned_objects.keys():
                c = mapping[id]
                assert c in self.engine.COLORS_OCCUPIED
                cam.setTagState(id, RenderState.make(ColorAttrib.makeFlat((c[0], c[1], c[2], 1)), 1))
