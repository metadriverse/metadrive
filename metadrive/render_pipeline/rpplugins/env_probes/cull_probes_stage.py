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

import math

from metadrive.render_pipeline.rpcore.render_stage import RenderStage
from metadrive.render_pipeline.rpcore.image import Image


class CullProbesStage(RenderStage):
    """ This stage takes the list of used cells and creates a list of environment
    probes for each cell """

    required_inputs = ["EnvProbes"]
    required_pipes = ["CellListBuffer"]

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.max_probes_per_cell = 4
        self.slice_width = pipeline.settings["lighting.culling_slice_width"]

    @property
    def produced_pipes(self):
        return {"PerCellProbes": self.per_cell_probes}

    @property
    def produced_defines(self):
        return {"MAX_PROBES_PER_CELL": self.max_probes_per_cell}

    def create(self):
        # TODO: Use no oversized triangle in this stage
        self.target = self.create_target("CullProbes")
        self.target.size = 0, 0
        self.target.prepare_buffer()

        self.per_cell_probes = Image.create_buffer("PerCellProbes", 0, "R32I")
        self.per_cell_probes.clear_image()
        self.target.set_shader_inputs(PerCellProbes=self.per_cell_probes, threadCount=1)

    def set_dimensions(self):
        max_cells = self._pipeline.light_mgr.total_tiles
        num_rows = int(math.ceil(max_cells / float(self.slice_width)))
        self.per_cell_probes.set_x_size(max_cells * self.max_probes_per_cell)
        self.target.size = self.slice_width, num_rows

    def reload_shaders(self):
        self.target.shader = self.load_plugin_shader("/$$rp/shader/tiled_culling.vert.glsl", "cull_probes.frag.glsl")
