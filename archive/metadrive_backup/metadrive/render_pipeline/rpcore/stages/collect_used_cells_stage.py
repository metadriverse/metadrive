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

from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class CollectUsedCellsStage(RenderStage):
    """ This stage collects the flagged cells from the FlagUsedCellsStage and
    makes a list of them """

    required_pipes = ["FlaggedCells"]

    @property
    def produced_pipes(self):
        return {
            "CellListBuffer": self.cell_list_buffer,
            "CellIndices": self.cell_index_buffer,
        }

    def create(self):
        self.target = self.create_target("CollectUsedCells")
        self.target.size = 0, 0
        self.target.prepare_buffer()

        self.cell_list_buffer = Image.create_buffer("CellList", 0, "R32I")
        self.cell_index_buffer = Image.create_2d_array("CellIndices", 0, 0, 0, "R32I")

        self.target.set_shader_inputs(CellListBuffer=self.cell_list_buffer, CellListIndices=self.cell_index_buffer)

    def update(self):
        self.cell_list_buffer.clear_image()
        self.cell_index_buffer.clear_image()

    def set_dimensions(self):
        tile_amount = self._pipeline.light_mgr.num_tiles
        num_slices = self._pipeline.settings["lighting.culling_grid_slices"]
        max_cells = tile_amount.x * tile_amount.y * num_slices

        self.cell_list_buffer.set_x_size(1 + max_cells)
        self.cell_index_buffer.set_x_size(tile_amount.x)
        self.cell_index_buffer.set_y_size(tile_amount.y)
        self.cell_index_buffer.set_z_size(num_slices)

        self.cell_list_buffer.clear_image()
        self.cell_index_buffer.clear_image()

        self.target.size = tile_amount.x, tile_amount.y

    def reload_shaders(self):
        self.target.shader = self.load_shader("collect_used_cells.frag.glsl")
