"""

Test-script to view the baked GI

"""

import os
import sys

os.chdir(os.path.realpath(os.path.dirname(__file__)))
sys.path.insert(0, "../../")

from _bake_params import *

from panda3d.core import *
from direct.showbase.ShowBase import ShowBase

from rpcore.globals import Globals
from rpcore.render_target import RenderTarget


class Application(ShowBase):
    def __init__(self):

        load_prc_file_data("", """
            textures-power-2 none
            win-size 1600 900
        """)

        ShowBase.__init__(self)
        Globals.load(self)
        Globals.resolution = LVecBase2i(1600, 900)
        sun_vector = Vec3(BAKE_SUN_VECTOR).normalized()

        diameter = (BAKE_MESH_END - BAKE_MESH_START).length()
        model_center = (BAKE_MESH_START + BAKE_MESH_END) * 0.5
        sun_shadow_map_resolution = 8192

        # model = loader.load_model("resources/test-scene.bam")
        model = loader.load_model("scene/scene.bam")
        model.reparent_to(render)
        model.flatten_strong()

        print("Rendering sun shadow map ..")
        sun_shadow_cam = Camera("SunShadowCamera")
        sun_shadow_lens = OrthographicLens()
        sun_shadow_lens.set_film_size(diameter * 2, diameter * 2)
        sun_shadow_lens.set_near_far(0, 2 * diameter)
        sun_shadow_cam.set_lens(sun_shadow_lens)
        sun_shadow_cam_np = render.attach_new_node(sun_shadow_cam)
        sun_shadow_cam_np.set_pos(model_center + sun_vector * diameter)
        sun_shadow_cam_np.look_at(model_center)

        sun_shadow_target = RenderTarget()
        sun_shadow_target.size = sun_shadow_map_resolution
        sun_shadow_target.add_depth_attachment(bits=32)
        sun_shadow_target.prepare_render(sun_shadow_cam_np)

        self.graphicsEngine.render_frame()
        sun_shadow_target.active = False
        shadow_mvp = self.get_mvp(sun_shadow_cam_np)

        # Load the dataset
        dataset = loader.load_texture("raw-bake.png")
        render.set_shader_input("GIDataTexture", dataset)

        # Load the display shader
        shader = Shader.load(Shader.SL_GLSL, "resources/display.vert.glsl", "resources/display.frag.glsl")
        render.set_shader(shader)

        render.set_shader_inputs(ShadowMap=sun_shadow_target.depth_tex, shadowMVP=shadow_mvp, sunVector=sun_vector)

        # Render spheres distributed over the mesh
        mesh_size = BAKE_MESH_END - BAKE_MESH_START

        for i in range(11):
            for j in range(11):
                for k in range(11):
                    offs_x = i / 10.0 * mesh_size.x + BAKE_MESH_START.x
                    offs_y = j / 10.0 * mesh_size.y + BAKE_MESH_START.y
                    offs_z = k / 10.0 * mesh_size.z + BAKE_MESH_START.z

                    sphere = loader.load_model("resources/sphere.bam")
                    sphere.reparent_to(render)
                    sphere.set_scale(0.02)
                    sphere.set_pos(offs_x, offs_y, offs_z)

    def get_mvp(self, cam_node):
        """ Computes the view-projection matrix of a camera """
        return render.get_transform(cam_node).get_mat() * cam_node.node().get_lens().get_projection_mat()


Application().run()
