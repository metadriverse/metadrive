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

from __future__ import division

import collections

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.pluginbase.base_plugin import BasePlugin

from .voxelization_stage import VoxelizationStage
from .vxgi_stage import VXGIStage


class Plugin(BasePlugin):

    name = "Voxel Global Illumination"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = (
        "Provides Global Illumination using Voxel Cone Tracing. This "
        "technique is still very unoptimized and experimental!"
    )
    version = "alpha (!)"

    def on_stage_setup(self):
        self._voxel_stage = self.create_stage(VoxelizationStage)
        self._vxgi_stage = self.create_stage(VXGIStage)

        self._voxel_stage.voxel_resolution = self.get_setting("grid_resolution")
        self._voxel_stage.voxel_world_size = self.get_setting("grid_ws_size")

        if self.is_plugin_enabled("pssm"):
            # Add shadow map as requirement
            self._voxel_stage.required_pipes.append("PSSMSceneSunShadowMapPCF")
            self._voxel_stage.required_inputs.append("PSSMSceneSunShadowMVP")

    def on_pre_render_update(self):
        task = self._queue[0]
        self._queue.rotate(-1)
        task()

    def on_pipeline_created(self):
        self._queue = collections.deque()
        self._queue.extend([self._voxelize_x, self._voxelize_y, self._voxelize_z])
        self._queue.extend([self._generate_mipmaps])

    def _set_grid_pos(self):
        """ Finds the new voxel grid position """
        grid_pos = Globals.base.camera.get_pos(Globals.base.render)

        # Snap the voxel grid
        voxel_size = 2.0 * self.get_setting("grid_ws_size") / self.get_setting("grid_resolution")
        snap_size = voxel_size * 2**4

        for dimension in range(3):
            cell_val = grid_pos.get_cell(dimension)
            grid_pos.set_cell(dimension, cell_val - cell_val % snap_size)

        self._voxel_stage.set_grid_position(grid_pos)

    def _voxelize_x(self):
        """ Voxelizes the scene from the x axis """
        self._set_grid_pos()
        self._voxel_stage.state = VoxelizationStage.S_voxelize_x

    def _voxelize_y(self):
        """ Voxelizes the scene from the y axis """
        self._voxel_stage.state = VoxelizationStage.S_voxelize_y

    def _voxelize_z(self):
        """ Voxelizes the scene from the z axis """
        self._voxel_stage.state = VoxelizationStage.S_voxelize_z

    def _generate_mipmaps(self):
        """ Generates the mipmaps for the voxel grid """
        self._voxel_stage.state = VoxelizationStage.S_gen_mipmaps
