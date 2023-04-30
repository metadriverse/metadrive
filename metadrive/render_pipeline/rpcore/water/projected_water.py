from panda3d.core import OmniBoundingVolume, Texture, Mat4, SamplerState, Vec3, PTALMatrix4f, PTAVecBase3f

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.water.water_manager import WaterManager


class ProjectedWater:
    def __init__(self, water_options):
        self.water_options = water_options
        self.water_level = 0.0
        self.model = Globals.base.loader.load_model("/$$rp/data/builtin_models/water/water_grid.bam")
        self.model.reparent_to(Globals.base.render)
        self.model.node().set_final(True)
        self.model.node().set_bounds(OmniBoundingVolume())
        self.model.set_two_sided(True)
        self.model.set_shader_input("waterHeight", self.water_level)
        self.model.set_mat(Mat4.identMat())
        self.model.clear_transform()

    def setup_water(self, pipeline, water_level):
        if (pipeline and self.model and isinstance(water_level, float) and self.water_options):

            self.pipeline = pipeline
            foam = Globals.base.loader.load_texture("/$$rp/data/builtin_models/water/water_foam.png")
            self.model.set_shader_input("waterFoam", foam)

            self.manager = WaterManager(self.water_options)
            self.manager.setup()
            self.manager.update()

            self.model.set_shader_input("waterHeightfield", self.manager.get_displacement_texture())
            self.model.set_shader_input("waterNormal", self.manager.get_normal_texture())

            # Set texture filter modes
            for tex in [foam, self.manager.get_displacement_texture(), self.manager.get_normal_texture()]:
                tex.set_wrap_u(SamplerState.WMRepeat)
                tex.set_wrap_u(SamplerState.WMRepeat)
                tex.set_minfilter(SamplerState.FTLinearMipmapLinear)
                tex.set_magfilter(SamplerState.FTLinearMipmapLinear)

            self.pipeline.set_effect(self.model, "/$$rp/effects/projected_water.yaml", {"render_shadow": False})

            self.water_level = water_level
            self.model.set_shader_input("waterHeight", self.water_level)

            Globals.base.add_task(self.update_task, "update_water")

    def clear_water(self):
        Globals.base.remove_task("update_water")
        if self.model:
            self.model.remove_node()

    def update_task(self, task):
        # Update water displacement
        self.manager.update()
        # Update the camera position and the current model view projection matrix
        self.model.set_shader_input("cameraPosition", Globals.base.camera.get_pos(render))
        self.model.set_shader_input("currentMVP", self.model.get_mat())
        return task.cont
