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


class CullLightsStage(RenderStage):
    """ This stage takes the list of used cells and creates a list of lights
    for each cell """

    required_pipes = ["CellListBuffer"]
    required_inputs = ["AllLightsData", "maxLightIndex"]

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.max_lights_per_cell = pipeline.settings["lighting.max_lights_per_cell"]

        if self.max_lights_per_cell > 2**16:
            self.fatal("lighting.max_lights_per_cell must be <=", 2**16, "!")

        self.slice_width = pipeline.settings["lighting.culling_slice_width"]
        self.cull_threads = 32

        # Amount of light classes. Has to match the ones in LightClassification.inc.glsl
        self.num_light_classes = 4

    @property
    def produced_pipes(self):
        return {"PerCellLights": self.grouped_cell_lights, "PerCellLightsCounts": self.grouped_cell_lights_counts}

    @property
    def produced_defines(self):
        return {
            "LC_SLICE_WIDTH": self.slice_width,
            "LC_CULL_THREADS": self.cull_threads,
            "LC_LIGHT_CLASS_COUNT": self.num_light_classes
        }

    def create(self):
        self.target_visible = self.create_target("GetVisibleLights")
        self.target_visible.size = 16, 16
        self.target_visible.prepare_buffer()

        # TODO: Use no oversized triangle in this stage
        self.target_cull = self.create_target("CullLights")
        self.target_cull.size = 0, 0
        self.target_cull.prepare_buffer()

        # TODO: Use no oversized triangle in this stage
        self.target_group = self.create_target("GroupLightsByClass")
        self.target_group.size = 0, 0
        self.target_group.prepare_buffer()

        self.frustum_lights_ctr = Image.create_counter("VisibleLightCount")
        self.frustum_lights = Image.create_buffer("FrustumLights", self._pipeline.light_mgr.MAX_LIGHTS, "R16UI")
        self.per_cell_lights = Image.create_buffer("PerCellLights", 0, "R16UI")
        self.per_cell_light_counts = Image.create_buffer(
            "PerCellLightCounts", 0, "R32I"
        )  # Needs to be R32 for atomic add in cull stage
        self.grouped_cell_lights = Image.create_buffer("GroupedPerCellLights", 0, "R16UI")
        self.grouped_cell_lights_counts = Image.create_buffer("GroupedPerCellLightsCount", 0, "R16UI")

        self.target_visible.set_shader_inputs(
            FrustumLights=self.frustum_lights, FrustumLightsCount=self.frustum_lights_ctr
        )

        self.target_cull.set_shader_inputs(
            PerCellLightsBuffer=self.per_cell_lights,
            PerCellLightCountsBuffer=self.per_cell_light_counts,
            FrustumLights=self.frustum_lights,
            FrustumLightsCount=self.frustum_lights_ctr,
            threadCount=self.cull_threads
        )

        self.target_group.set_shader_inputs(
            PerCellLightsBuffer=self.per_cell_lights,
            PerCellLightCountsBuffer=self.per_cell_light_counts,
            GroupedCellLightsBuffer=self.grouped_cell_lights,
            GroupedPerCellLightsCountBuffer=self.grouped_cell_lights_counts,
            threadCount=1
        )

    def reload_shaders(self):
        self.target_cull.shader = self.load_shader("tiled_culling.vert.glsl", "cull_lights.frag.glsl")
        self.target_group.shader = self.load_shader("tiled_culling.vert.glsl", "group_lights.frag.glsl")
        self.target_visible.shader = self.load_shader("view_frustum_cull.frag.glsl")

    def update(self):
        self.frustum_lights_ctr.clear_image()

    def set_dimensions(self):
        max_cells = self._pipeline.light_mgr.total_tiles
        num_rows_threaded = int(math.ceil((max_cells * self.cull_threads) / float(self.slice_width)))
        num_rows = int(math.ceil(max_cells / float(self.slice_width)))
        self.per_cell_lights.set_x_size(max_cells * self.max_lights_per_cell)
        self.per_cell_light_counts.set_x_size(max_cells)
        self.grouped_cell_lights.set_x_size(max_cells * (self.max_lights_per_cell + self.num_light_classes))
        self.target_cull.size = self.slice_width, num_rows_threaded
        self.target_group.size = self.slice_width, num_rows

        self.grouped_cell_lights_counts.set_x_size(max_cells * (1 + self.num_light_classes))
