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

from metadrive.render_pipeline.rpcore.render_stage import RenderStage
from metadrive.render_pipeline.rpcore.image import Image


class FlagUsedCellsStage(RenderStage):
    """ This stage flags all used cells based on the depth buffer """

    required_pipes = ["GBuffer"]
    required_inputs = []

    @property
    def produced_pipes(self):
        return {"FlaggedCells": self.cell_grid_flags}

    def create(self):
        self.target = self.create_target("FlagUsedCells")
        self.target.prepare_buffer()

        self.cell_grid_flags = Image.create_2d_array(
            "CellGridFlags", 0, 0, self._pipeline.settings["lighting.culling_grid_slices"], "R8"
        )
        self.target.set_shader_input("cellGridFlags", self.cell_grid_flags)

    def update(self):
        self.cell_grid_flags.clear_image()

    def set_dimensions(self):
        tile_amount = self._pipeline.light_mgr.num_tiles
        self.cell_grid_flags.set_x_size(tile_amount.x)
        self.cell_grid_flags.set_y_size(tile_amount.y)

    def reload_shaders(self):
        self.target.shader = self.load_shader("flag_used_cells.frag.glsl")
