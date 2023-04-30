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

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.render_stage import RenderStage

from panda3d.core import Camera, OrthographicLens, NodePath, CullFaceAttrib
from panda3d.core import DepthTestAttrib, Vec4, PTALVecBase3, Vec3, SamplerState
from panda3d.core import ColorWriteAttrib


class VoxelizationStage(RenderStage):
    """ This stage voxelizes the whole scene """

    required_inputs = ["DefaultEnvmap", "AllLightsData", "maxLightIndex"]
    required_pipes = []

    # The different states of voxelization
    S_disabled = 0
    S_voxelize_x = 1
    S_voxelize_y = 2
    S_voxelize_z = 3
    S_gen_mipmaps = 4

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.voxel_resolution = 256
        self.voxel_world_size = -1
        self.state = self.S_disabled
        self.create_ptas()

    def set_grid_position(self, pos):
        self.pta_next_grid_pos[0] = pos

    def create_ptas(self):
        self.pta_next_grid_pos = PTALVecBase3.empty_array(1)
        self.pta_grid_pos = PTALVecBase3.empty_array(1)

    @property
    def produced_inputs(self):
        return {"voxelGridPosition": self.pta_grid_pos}

    @property
    def produced_pipes(self):
        return {"SceneVoxels": self.voxel_grid}

    def create(self):
        # Create the voxel grid used to generate the voxels
        self.voxel_temp_grid = Image.create_3d(
            "VoxelsTemp", self.voxel_resolution, self.voxel_resolution, self.voxel_resolution, "RGBA8"
        )
        self.voxel_temp_grid.set_clear_color(Vec4(0))
        self.voxel_temp_nrm_grid = Image.create_3d(
            "VoxelsTemp", self.voxel_resolution, self.voxel_resolution, self.voxel_resolution, "R11G11B10"
        )
        self.voxel_temp_nrm_grid.set_clear_color(Vec4(0))

        # Create the voxel grid which is a copy of the temporary grid, but stable
        self.voxel_grid = Image.create_3d(
            "Voxels", self.voxel_resolution, self.voxel_resolution, self.voxel_resolution, "RGBA8"
        )
        self.voxel_grid.set_clear_color(Vec4(0))
        self.voxel_grid.set_minfilter(SamplerState.FT_linear_mipmap_linear)

        # Create the camera for voxelization
        self.voxel_cam = Camera("VoxelizeCam")
        self.voxel_cam.set_camera_mask(self._pipeline.tag_mgr.get_mask("voxelize"))
        self.voxel_cam_lens = OrthographicLens()
        self.voxel_cam_lens.set_film_size(-2.0 * self.voxel_world_size, 2.0 * self.voxel_world_size)
        self.voxel_cam_lens.set_near_far(0.0, 2.0 * self.voxel_world_size)
        self.voxel_cam.set_lens(self.voxel_cam_lens)
        self.voxel_cam_np = Globals.base.render.attach_new_node(self.voxel_cam)
        self._pipeline.tag_mgr.register_camera("voxelize", self.voxel_cam)

        # Create the voxelization target
        self.voxel_target = self.create_target("VoxelizeScene")
        self.voxel_target.size = self.voxel_resolution
        self.voxel_target.prepare_render(self.voxel_cam_np)

        # Create the target which copies the voxel grid
        self.copy_target = self.create_target("CopyVoxels")
        self.copy_target.size = self.voxel_resolution
        self.copy_target.prepare_buffer()

        # TODO! Does not work with the new render target yet - maybe add option
        # to post process region for instances?
        self.copy_target.instance_count = self.voxel_resolution
        self.copy_target.set_shader_inputs(SourceTex=self.voxel_temp_grid, DestTex=self.voxel_grid)

        # Create the target which generates the mipmaps
        self.mip_targets = []
        mip_size, mip = self.voxel_resolution, 0
        while mip_size > 1:
            mip_size, mip = mip_size // 2, mip + 1
            mip_target = self.create_target("GenMipmaps:" + str(mip))
            mip_target.size = mip_size
            mip_target.prepare_buffer()
            mip_target.instance_count = mip_size
            mip_target.set_shader_inputs(SourceTex=self.voxel_grid, sourceMip=(mip - 1))
            mip_target.set_shader_input("DestTex", self.voxel_grid, False, True, -1, mip, 0)
            self.mip_targets.append(mip_target)

        # Create the initial state used for rendering voxels
        initial_state = NodePath("VXGIInitialState")
        initial_state.set_attrib(CullFaceAttrib.make(CullFaceAttrib.M_cull_none), 100000)
        initial_state.set_attrib(DepthTestAttrib.make(DepthTestAttrib.M_none), 100000)
        initial_state.set_attrib(ColorWriteAttrib.make(ColorWriteAttrib.C_off), 100000)
        self.voxel_cam.set_initial_state(initial_state.get_state())

        Globals.base.render.set_shader_inputs(
            voxelGridPosition=self.pta_next_grid_pos, VoxelGridDest=self.voxel_temp_grid
        )

    def update(self):
        self.voxel_cam_np.show()
        self.voxel_target.active = True
        self.copy_target.active = False

        for target in self.mip_targets:
            target.active = False

        # Voxelization disable
        if self.state == self.S_disabled:
            self.voxel_cam_np.hide()
            self.voxel_target.active = False

        # Voxelization from X-Axis
        elif self.state == self.S_voxelize_x:
            # Clear voxel grid
            self.voxel_temp_grid.clear_image()
            self.voxel_cam_np.set_pos(self.pta_next_grid_pos[0] + Vec3(self.voxel_world_size, 0, 0))
            self.voxel_cam_np.look_at(self.pta_next_grid_pos[0])

        # Voxelization from Y-Axis
        elif self.state == self.S_voxelize_y:
            self.voxel_cam_np.set_pos(self.pta_next_grid_pos[0] + Vec3(0, self.voxel_world_size, 0))
            self.voxel_cam_np.look_at(self.pta_next_grid_pos[0])

        # Voxelization from Z-Axis
        elif self.state == self.S_voxelize_z:
            self.voxel_cam_np.set_pos(self.pta_next_grid_pos[0] + Vec3(0, 0, self.voxel_world_size))
            self.voxel_cam_np.look_at(self.pta_next_grid_pos[0])

        # Generate mipmaps
        elif self.state == self.S_gen_mipmaps:
            self.voxel_target.active = False
            self.copy_target.active = True
            self.voxel_cam_np.hide()

            for target in self.mip_targets:
                target.active = True

            # As soon as we generate the mipmaps, we need to update the grid position
            # as well
            self.pta_grid_pos[0] = self.pta_next_grid_pos[0]

    def reload_shaders(self):
        self.copy_target.shader = self.load_plugin_shader(
            "/$$rp/shader/default_post_process_instanced.vert.glsl", "copy_voxels.frag.glsl"
        )
        mip_shader = self.load_plugin_shader(
            "/$$rp/shader/default_post_process_instanced.vert.glsl", "generate_mipmaps.frag.glsl"
        )
        for target in self.mip_targets:
            target.shader = mip_shader

    def set_shader_input(self, *args):
        Globals.render.set_shader_input(*args)

    def set_shader_inputs(self, **kwargs):
        Globals.render.set_shader_inputs(**kwargs)
