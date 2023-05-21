# Author: Epihaius
# Date: 2023-04-29

import array

from direct.showbase.ShowBase import ShowBase
from panda3d.core import *


class MyApp(ShowBase):
    def __init__(self):

        ShowBase.__init__(self)

        # set up a light source
        p_light = PointLight("point_light")
        p_light.set_color((1., 1., 1., 1.))
        self.light = self.camera.attach_new_node(p_light)
        self.light.set_pos(5., -10., 7.)
        self.render.set_light(self.light)
        self.disable_mouse()
        self.camera.set_pos(0., -50., 50.)
        self.camera.look_at(-10., 0., 0.)

        triangulator = Triangulator()
        values = array.array("f", [])
        vertex_data = GeomVertexData("poly", GeomVertexFormat.get_v3n3(), Geom.UH_static)

        with open("poly_coords.txt", "r") as coords_file:
            coords = coords_file.readlines()

        for i, coord in enumerate(coords):
            x, y = [float(c) for c in coord.replace("\n", "").split(" ")]
            values.extend((x, y, 0., 0., 0., 1.))
            triangulator.add_vertex(x, y)
            triangulator.add_polygon_vertex(i)

        triangulator.triangulate()
        prim = GeomTriangles(Geom.UH_static)

        for i in range(triangulator.get_num_triangles()):
            index0 = triangulator.get_triangle_v0(i)
            index1 = triangulator.get_triangle_v1(i)
            index2 = triangulator.get_triangle_v2(i)
            prim.add_vertices(index0, index1, index2)

        # add the values to the vertex table using a memoryview;
        # since the size of a memoryview cannot change, the vertex data table
        # already needs to have the right amount of rows before creating
        # memoryviews from its array(s)
        vertex_data.unclean_set_num_rows(len(coords))
        # retrieve the data array for modification
        data_array = vertex_data.modify_array(0)
        memview = memoryview(data_array).cast("B").cast("f")
        memview[:] = values

        geom = Geom(vertex_data)
        geom.add_primitive(prim)
        node = GeomNode("poly_node")
        node.add_geom(geom)
        poly = self.render.attach_new_node(node)
        poly.set_render_mode_wireframe()


app = MyApp()
app.run()
