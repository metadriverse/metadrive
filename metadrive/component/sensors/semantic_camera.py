from panda3d.core import RenderState, LightAttrib, ColorAttrib, ShaderAttrib, TextureAttrib, FrameBufferProperties

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import Semantics, CameraTagStateKey


class SemanticCamera(BaseCamera):
    CAM_MASK = CamMask.SemanticCam

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        buffer_props = FrameBufferProperties()
        buffer_props.set_rgba_bits(8, 8, 8, 8)
        buffer_props.set_depth_bits(8)
        buffer_props.set_force_hardware(True)
        buffer_props.set_multisamples(16)
        buffer_props.set_srgb_color(False)
        buffer_props.set_stereo(False)
        buffer_props.set_stencil_bits(0)
        super(SemanticCamera, self).__init__(engine, cuda, buffer_props)

    def _setup_effect(self):
        """
        Use tag to apply color to different object class
        Returns: None

        """
        # setup camera
        cam = self.get_cam().node()
        cam.setTagStateKey(CameraTagStateKey.Semantic)
        for t in [v for v, m in vars(Semantics).items() if not (v.startswith('_') or callable(m))]:
            label, c = getattr(Semantics, t)
            if label == Semantics.TERRAIN.label:
                from metadrive.engine.core.terrain import Terrain
                cam.setTagState(
                    label, Terrain.make_render_state(self.engine, "terrain.vert.glsl", "terrain_semantics.frag.glsl")
                )
            else:
                cam.setTagState(
                    label,
                    RenderState.make(
                        ShaderAttrib.makeOff(), LightAttrib.makeAllOff(), TextureAttrib.makeOff(),
                        ColorAttrib.makeFlat((c[0] / 255, c[1] / 255, c[2] / 255, 1)), 1
                    )
                )
