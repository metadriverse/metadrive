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

from panda3d.core import Vec4, SamplerState, BoundingVolume

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.image import Image


class ProbeManager(RPObject):
    """ Manages all environment probes """
    def __init__(self):
        """ Initializes a new probe manager """
        RPObject.__init__(self)
        self.probes = []
        self.max_probes = 3
        self.resolution = 128
        self.diffuse_resolution = 4

    def init(self):
        """ Creates the cubemap storage """

        # Storage for the specular components (with mipmaps)
        self.cubemap_storage = Image.create_cube_array("EnvmapStorage", self.resolution, self.max_probes, "RGBA16")
        self.cubemap_storage.set_minfilter(SamplerState.FT_linear_mipmap_linear)
        self.cubemap_storage.set_magfilter(SamplerState.FT_linear)
        self.cubemap_storage.set_clear_color(Vec4(1.0, 0.0, 0.1, 1.0))
        self.cubemap_storage.clear_image()

        # Storage for the diffuse component
        self.diffuse_storage = Image.create_cube_array(
            "EnvmapDiffStorage", self.diffuse_resolution, self.max_probes, "RGBA16"
        )
        self.diffuse_storage.set_clear_color(Vec4(1, 0, 0.2, 1.0))
        self.diffuse_storage.clear_image()

        # Data-storage to store all cubemap properties
        self.dataset_storage = Image.create_buffer("EnvmapData", self.max_probes * 5, "RGBA32")
        self.dataset_storage.set_clear_color(Vec4(0))
        self.dataset_storage.clear_image()

    def add_probe(self, probe):
        """ Adds a new probe """
        if len(self.probes) >= self.max_probes:
            self.error("Cannot attach probe, out of slots!")
            return False
        probe.last_update = -1
        probe.index = len(self.probes)
        self.probes.append(probe)
        return True

    def update(self):
        """ Updates the manager, updating all probes """
        ptr = self.dataset_storage.modify_ram_image()
        for probe in self.probes:
            if probe.modified:
                probe.write_to_buffer(ptr)

    @property
    def num_probes(self):
        return len(self.probes)

    def find_probe_to_update(self):
        """ Finds the next probe which requires an update, or returns None """
        if not self.probes:
            return None

        view_frustum = Globals.base.camLens.make_bounds()
        view_frustum.xform(Globals.base.cam.get_transform(Globals.base.render).get_mat())

        def rating(probe):
            return probe.last_update

        for candidate in sorted(self.probes, key=rating):
            if view_frustum.contains(candidate.bounds) == BoundingVolume.IF_no_intersection:
                continue
            return candidate
        return None
