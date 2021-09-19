from panda3d.core import Vec3, Shader, RenderState, ShaderAttrib, GeoMipTerrain

from metadrive.component.vehicle_module.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import RENDER_MODE_NONE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import get_global_config, engine_initialized, get_engine


class DepthCamera(BaseCamera):
    # shape(dim_1, dim_2)
    CAM_MASK = CamMask.DepthCam

    GROUND_HEIGHT = -1.2
    VIEW_GROUND = False
    GROUND = None
    GROUND_MODEL = None

    def __init__(self):
        assert engine_initialized(), "You should initialize engine before adding camera to vehicle"
        config = get_global_config()["vehicle_config"]["depth_camera"]
        self.BUFFER_W, self.BUFFER_H = config[0], config[1]
        self.VIEW_GROUND = config[2]
        super(DepthCamera, self).__init__()
        cam = self.get_cam()
        lens = self.get_lens()

        cam.lookAt(0, 2.4, 1.3)
        lens.setFov(60)
        lens.setAspectRatio(2.0)
        if get_engine().mode == RENDER_MODE_NONE or not AssetLoader.initialized() or type(self)._singleton.init_num > 0:
            return
        # add shader for it
        if get_global_config()["headless_machine_render"]:
            vert_path = AssetLoader.file_path("shaders", "depth_cam_gles.vert.glsl")
            frag_path = AssetLoader.file_path("shaders", "depth_cam_gles.frag.glsl")
        else:
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
            # terrain.setBruteforce(True)
            # # Since the terrain is a texture, shader will not calculate the depth information, we add a moving terrain
            # # model to enable the depth information of terrain
            self.GROUND_MODEL = self.GROUND.getRoot()
            self.GROUND_MODEL.setPos(-128, 0, self.GROUND_HEIGHT)
            self.GROUND_MODEL.reparentTo(type(self)._singleton.origin)
            self.GROUND_MODEL.hide(CamMask.AllOn)
            self.GROUND_MODEL.show(CamMask.DepthCam)
            self.GROUND.generate()
