"""

Renders the sphere using the render pipeline

"""

import _tmp_material as material

import sys
from panda3d.core import load_prc_file_data, Vec4
from direct.showbase.ShowBase import ShowBase


class Application(ShowBase):
    def __init__(self):
        sys.path.insert(0, "../../")
        load_prc_file_data("", "win-size 512 512")
        load_prc_file_data("", "textures-power-2 none")
        load_prc_file_data("", "print-pipe-types #f")
        load_prc_file_data("", "notify-level-glgsg error")
        # load_prc_file_data("", "win-size 1024 1024")

        from rpcore import RenderPipeline, PointLight

        self.render_pipeline = RenderPipeline()
        self.render_pipeline.mount_mgr.config_dir = "config/"
        self.render_pipeline.create(self)

        sphere = self.loader.loadModel("res/sphere.bam")
        sphere.reparent_to(self.render)

        self.disableMouse()
        self.camLens.setFov(40)
        self.camLens.setNearFar(0.03, 2000.0)
        self.camera.set_pos(0, -3.5, 0)
        self.camera.look_at(0, -2.5, 0)

        self.render2d.hide()
        self.aspect2d.hide()

        light = PointLight()
        light.pos = 10, -10, 10
        light.radius = 1e20
        light.color = (1, 1, 1)
        light.inner_radius = 4.0
        light.energy = 3
        self.render_pipeline.add_light(light)

        light = PointLight()
        light.pos = -10, -10, 10
        light.radius = 1e20
        light.color = (1, 1, 1)
        light.inner_radius = 4.0
        light.energy = 3
        self.render_pipeline.add_light(light)

        for mat in sphere.find_all_materials():
            mat.roughness = material.roughness
            mat.base_color = Vec4(*(list(material.basecolor) + [1]))
            mat.refractive_index = material.ior

            mat.metallic = 1.0 if material.mat_type == "metallic" else 0.0

            if material.mat_type == "clearcoat":
                mat.emission = (2, 0, 0, 0)
                mat.metallic = 1.0
                mat.refractive_index = 1.51

            if material.mat_type == "foliage":
                mat.emission = (5, 0, 0, 0)
                mat.metallic = 0.0
                mat.refractive_index = 1.51

        for i in range(10):
            self.taskMgr.step()

        self.win.save_screenshot("scene-rp.png")

        self.accept("r", self.reload)

    def reload(self):
        print("Reloading")
        self.render_pipeline.reload_shaders()

        for i in range(4):
            self.taskMgr.step()

        self.win.save_screenshot("scene-rp.png")


if len(sys.argv) <= 1:
    Application().run()
else:
    Application()
