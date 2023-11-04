# Author: Epihaius
# Date: 2023-04-29
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *


# @time_me
from metadrive.utils.vertex import make_polygon_model


def calculate_normal(p1, p2, p3):
    # These are three points (numpy arrays) on the face
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    return normal


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # set up a light source
        p_light = PointLight("point_light")
        p_light.set_color((1., 1., 1., 1.))
        self.light = self.camera.attach_new_node(p_light)
        self.light.set_pos(0., 10., 0.)
        self.render.set_light(self.light)
        self.disable_mouse()
        self.camera.set_pos(0., 15., 15.)
        self.camera.look_at(0., 0., 0.)

        coords = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (3, 5), (2, 4), (-1, 1), (1, 2)]
        poly = make_polygon_model(coords, 1, force_anticlockwise=True)
        poly.reparentTo(self.render)
        poly.setColor(0.8, 0.5, 0.8, 0)
        # poly.set_render_mode_wireframe()
        poly.hprInterval(6, (360, 0, 0)).loop()
        # self.camera.look_at(poly)


app = MyApp()
app.run()
