from panda3d.core import Vec3, NodePath, Shader, RenderState, ShaderAttrib, BitMask32, GeoMipTerrain

from pgdrive.constants import CamMask
from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.engine.core.image_buffer import ImageBuffer


class DepthCamera(ImageBuffer):
    # shape(dim_1, dim_2)
    BUFFER_W = 84  # dim 1
    BUFFER_H = 84  # dim 2
    CAM_MASK = CamMask.DepthCam
    GROUND = -1.2
    TASK_NAME = "ground follow"
    default_region = [1 / 3, 2 / 3, ImageBuffer.display_bottom, 1.0]

    def __init__(self, length: int, width: int, view_ground: bool, chassis_np: NodePath):
        """
        :param length: Control resolution of this sensor
        :param width: Control resolution of this sensor
        :param view_ground: Lane line will be invisible when set to True
        :param chassis_np: The vehicle chassis to place this sensor
        """
        self.view_ground = view_ground
        self.BUFFER_W = length
        self.BUFFER_H = width
        super(DepthCamera, self).__init__(
            self.BUFFER_W, self.BUFFER_H, Vec3(0.0, 0.8, 1.5), self.BKG_COLOR, parent_node=chassis_np
        )
        self.add_to_display(self.default_region)
        self.cam.lookAt(0, 2.4, 1.3)
        self.lens.setFov(60)
        self.lens.setAspectRatio(2.0)

        # add shader for it
        if self.engine.global_config["headless_image"]:
            vert_path = AssetLoader.file_path("shaders", "depth_cam_gles.vert.glsl")
            frag_path = AssetLoader.file_path("shaders", "depth_cam_gles.frag.glsl")
        else:
            from pgdrive.utils import is_mac
            if is_mac():
                vert_path = AssetLoader.file_path("shaders", "depth_cam_mac.vert.glsl")
                frag_path = AssetLoader.file_path("shaders", "depth_cam_mac.frag.glsl")
            else:
                vert_path = AssetLoader.file_path("shaders", "depth_cam.vert.glsl")
                frag_path = AssetLoader.file_path("shaders", "depth_cam.frag.glsl")
        custom_shader = Shader.load(Shader.SL_GLSL, vertex=vert_path, fragment=frag_path)
        self.cam.node().setInitialState(RenderState.make(ShaderAttrib.make(custom_shader, 1)))

        if self.view_ground:
            self.ground = GeoMipTerrain("mySimpleTerrain")

            self.ground.setHeightfield(AssetLoader.file_path("textures", "height_map.png"))
            # terrain.setBruteforce(True)
            # # Since the terrain is a texture, shader will not calculate the depth information, we add a moving terrain
            # # model to enable the depth information of terrain
            self.ground_model = self.ground.getRoot()
            self.ground_model.reparentTo(chassis_np)
            self.ground_model.setPos(-128, 0, self.GROUND)
            self.ground_model.hide(BitMask32.allOn())
            self.ground_model.show(CamMask.DepthCam)
            self.ground.Generate()
            self.engine.task_manager.add(
                self.renew_pos_of_ground_mode, self.TASK_NAME, extraArgs=[chassis_np], appendTask=True
            )

    def renew_pos_of_ground_mode(self, chassis_np: Vec3, task):
        self.ground_model.setPos(-128, 0, self.GROUND)
        self.ground_model.setP(-chassis_np.getP())
        return task.cont

    def destroy(self):
        super(DepthCamera, self).destroy()
        self.engine.task_manager.remove(self.TASK_NAME)
