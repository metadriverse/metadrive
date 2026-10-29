"""

RenderPipeline

Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

from metadrive.render_pipeline.rpcore.rpobject import RPObject

from panda3d.core import GeomVertexFormat, GeomVertexData, Geom, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode, NodePath, Camera, OmniBoundingVolume
from panda3d.core import TransparencyAttrib, OrthographicLens, Vec4


class PostProcessRegion(RPObject):
    """ Simple wrapper class to create fullscreen triangles and quads """
    @classmethod
    def make(cls, internal_buffer, *args):
        return cls(internal_buffer, *args)

    def __init__(self, internal_buffer, *args):
        RPObject.__init__(self)
        self._buffer = internal_buffer
        self._region = self._buffer.make_display_region(*args)
        self._node = NodePath("RTRoot")

        self._make_fullscreen_tri()
        self._make_fullscreen_cam()
        self._init_function_pointers()

    def _init_function_pointers(self):
        self.set_sort = self._region.set_sort
        self.set_instance_count = self._tri.set_instance_count
        self.disable_clears = self._region.disable_clears
        self.set_active = self._region.set_active
        self.set_clear_depth_active = self._region.set_clear_depth_active
        self.set_clear_depth = self._region.set_clear_depth
        self.set_shader = self._tri.set_shader
        self.set_camera = self._region.set_camera
        self.set_clear_color_active = self._region.set_clear_color_active
        self.set_clear_color = self._region.set_clear_color
        self.set_attrib = self._tri.set_attrib

    def _make_fullscreen_tri(self):
        """ Creates the oversized triangle used for rendering """
        vformat = GeomVertexFormat.get_v3()
        vdata = GeomVertexData("vertices", vformat, Geom.UH_static)
        vdata.set_num_rows(3)
        vwriter = GeomVertexWriter(vdata, "vertex")
        vwriter.add_data3f(-1, 0, -1)
        vwriter.add_data3f(3, 0, -1)
        vwriter.add_data3f(-1, 0, 3)
        gtris = GeomTriangles(Geom.UH_static)
        gtris.add_next_vertices(3)
        geom = Geom(vdata)
        geom.add_primitive(gtris)
        geom_node = GeomNode("gn")
        geom_node.add_geom(geom)
        geom_node.set_final(True)
        geom_node.set_bounds(OmniBoundingVolume())
        tri = NodePath(geom_node)
        tri.set_depth_test(False)
        tri.set_depth_write(False)
        tri.set_attrib(TransparencyAttrib.make(TransparencyAttrib.M_none), 10000)
        tri.set_color(Vec4(1))
        tri.set_bin("unsorted", 10)
        tri.reparent_to(self._node)
        self._tri = tri

    def set_shader_input(self, *args, **kwargs):
        if kwargs.get("override", False):
            self._node.set_shader_input(*args, priority=100000)
        else:
            self._tri.set_shader_input(*args)

    def set_shader_inputs(self, **kwargs):
        self._tri.set_shader_inputs(**kwargs)

    def _make_fullscreen_cam(self):
        """ Creates an orthographic camera for the buffer """
        buffer_cam = Camera("BufferCamera")
        lens = OrthographicLens()
        lens.set_film_size(2, 2)
        lens.set_film_offset(0, 0)
        lens.set_near_far(-100, 100)
        buffer_cam.set_lens(lens)
        buffer_cam.set_cull_bounds(OmniBoundingVolume())
        self._camera = self._node.attach_new_node(buffer_cam)
        self._region.set_camera(self._camera)
