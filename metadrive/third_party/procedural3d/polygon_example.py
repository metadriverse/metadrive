# Author: Epihaius
# Date: 2023-04-29
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from shapely import geometry
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
        self.camera.set_pos(0., 15., 100.)
        self.camera.look_at(0., 0., 0.)
        #
        points = [
            [236.70966602, -79.94476767], [235.48608421, -81.79984158], [234.17903486, -71.70488153],
            [233.22807516, -64.56937753], [232.70516199, -60.43428241], [230.06360069, -58.76593274],
            [226.86714349, -57.3599948], [222.64706585, -53.10916339], [218.20545033, -48.79186],
            [219.96329603, -46.58656909], [224.27795345, -50.88911069], [227.92423276, -54.4945615],
            [231.56956222, -57.75740367], [235.59826439, -61.20357757], [235.19689215, -62.43117443],
            [234.1779322, -61.86125077], [234.68204325, -64.88105623], [235.55523041, -71.68052],
            [236.70966602, -79.94476767]
        ]
        poly = make_polygon_model([[point[0] - 220, point[1] + 60] for point in points[:-1]][::-1], 1)
        poly = make_polygon_model([[point[0] - 220, point[1] + 60] for point in points], 1)
        # poly = make_polygon_model([[point[0]-220, point[1]+60] for point in points[:-1]], 1)

        # coords = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (3, 5), (2, 4), (-1, 1), (1, 2), (0, 0)]
        # poly = make_polygon_model(coords, 1, force_anticlockwise=True)
        poly.reparentTo(self.render)
        poly.setColor(0.8, 0.5, 0.8, 0)
        # poly.set_render_mode_wireframe()
        poly.hprInterval(6, (360, 360, 360)).loop()
        self.camera.look_at(poly)


app = MyApp()
app.run()
