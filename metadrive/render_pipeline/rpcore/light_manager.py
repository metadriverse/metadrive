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

from panda3d.core import LVecBase2i, PTAInt

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.gpu_command_queue import GPUCommandQueue
from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.native import InternalLightManager, PointLight, ShadowManager
from metadrive.render_pipeline.rpcore.rpobject import RPObject

from metadrive.render_pipeline.rpcore.stages.apply_lights_stage import ApplyLightsStage
from metadrive.render_pipeline.rpcore.stages.collect_used_cells_stage import CollectUsedCellsStage
from metadrive.render_pipeline.rpcore.stages.cull_lights_stage import CullLightsStage
from metadrive.render_pipeline.rpcore.stages.flag_used_cells_stage import FlagUsedCellsStage
from metadrive.render_pipeline.rpcore.stages.shadow_stage import ShadowStage


class LightManager(RPObject):
    """ This class is a wrapper around the InternalLightManager, and provides
    additional functionality like setting up all required stages and defines."""

    # Maximum amount of lights, has to match the definitions in the native code
    MAX_LIGHTS = 65535

    # Maximum amount of shadow sources
    MAX_SOURCES = 2048

    def __init__(self, pipeline):
        """ Constructs the light manager """
        RPObject.__init__(self)
        self.pipeline = pipeline
        self.compute_tile_size()
        self.init_internal_manager()
        self.init_command_queue()
        self.init_shadow_manager()
        self.init_stages()

    @property
    def total_tiles(self):
        """ Returns the total amount of tiles """
        return self.num_tiles.x * self.num_tiles.y * \
            self.pipeline.settings["lighting.culling_grid_slices"]

    @property
    def num_lights(self):
        """ Returns the amount of stored lights """
        return self.internal_mgr.num_lights

    @property
    def num_shadow_sources(self):
        """ Returns the amount of stored shadow sources """
        return self.internal_mgr.num_shadow_sources

    @property
    def shadow_atlas_coverage(self):
        """ Returns the shadow atlas coverage in percentage  """
        return self.internal_mgr.shadow_manager.atlas.coverage * 100.0

    def add_light(self, light):
        """ Adds a new light """
        self.internal_mgr.add_light(light)
        self.pta_max_light_index[0] = self.internal_mgr.max_light_index

    def remove_light(self, light):
        """ Removes a light """
        self.internal_mgr.remove_light(light)
        self.pta_max_light_index[0] = self.internal_mgr.max_light_index

    def update(self):
        """ Main update method to process the GPU commands """
        self.internal_mgr.set_camera_pos(Globals.base.camera.get_pos(Globals.base.render))
        self.internal_mgr.update()
        self.shadow_manager.update()
        self.cmd_queue.process_queue()

    def reload_shaders(self):
        """ Reloads all assigned shaders """
        self.cmd_queue.reload_shaders()

    def compute_tile_size(self):
        """ Computes how many tiles there are on screen """
        self.tile_size = LVecBase2i(
            self.pipeline.settings["lighting.culling_grid_size_x"],
            self.pipeline.settings["lighting.culling_grid_size_y"]
        )
        num_tiles_x = int(math.ceil(Globals.resolution.x / float(self.tile_size.x)))
        num_tiles_y = int(math.ceil(Globals.resolution.y / float(self.tile_size.y)))
        self.debug(
            "Tile size =", self.tile_size.x, "x", self.tile_size.y, ", Num tiles =", num_tiles_x, "x", num_tiles_y
        )
        self.num_tiles = LVecBase2i(num_tiles_x, num_tiles_y)

    def init_command_queue(self):
        """ Inits the command queue """
        self.cmd_queue = GPUCommandQueue(self.pipeline)
        self.cmd_queue.register_input("LightData", self.img_light_data)
        self.cmd_queue.register_input("SourceData", self.img_source_data)
        self.internal_mgr.set_command_list(self.cmd_queue.command_list)

    def init_shadow_manager(self):
        """ Inits the shadow manager """
        self.shadow_manager = ShadowManager()
        self.shadow_manager.set_max_updates(self.pipeline.settings["shadows.max_updates"])
        self.shadow_manager.set_scene(Globals.base.render)
        self.shadow_manager.set_tag_state_manager(self.pipeline.tag_mgr)
        self.shadow_manager.atlas_size = self.pipeline.settings["shadows.atlas_size"]
        self.internal_mgr.shadow_manager = self.shadow_manager

    def init_shadows(self):
        """ Inits the shadows, this has to get called after the stages were
        created, because we need the GraphicsOutput of the shadow atlas, which
        is not available earlier """
        self.shadow_manager.set_atlas_graphics_output(self.shadow_stage.atlas_buffer)
        self.shadow_manager.init()

    def init_internal_manager(self):
        """ Creates the light storage manager and the buffer to store the light data """
        self.internal_mgr = InternalLightManager()
        self.internal_mgr.set_shadow_update_distance(self.pipeline.settings["shadows.max_update_distance"])

        # Storage for the Lights
        per_light_vec4s = 4
        self.img_light_data = Image.create_buffer("LightData", self.MAX_LIGHTS * per_light_vec4s, "RGBA16")
        self.img_light_data.clear_image()

        self.pta_max_light_index = PTAInt.empty_array(1)
        self.pta_max_light_index[0] = 0

        # Storage for the shadow sources
        per_source_vec4s = 5

        # IMPORTANT: RGBA32 is really required here. Otherwise artifacts and bad
        # shadow filtering occur due to precision issues
        self.img_source_data = Image.create_buffer("ShadowSourceData", self.MAX_SOURCES * per_source_vec4s, "RGBA32")
        self.img_light_data.clear_image()

        # Register the buffer
        inputs = self.pipeline.stage_mgr.inputs
        inputs["AllLightsData"] = self.img_light_data
        inputs["ShadowSourceData"] = self.img_source_data
        inputs["maxLightIndex"] = self.pta_max_light_index

    def init_stages(self):
        """ Inits all required stages for the lighting """

        add_stage = self.pipeline.stage_mgr.add_stage

        self.flag_cells_stage = FlagUsedCellsStage(self.pipeline)
        add_stage(self.flag_cells_stage)

        self.collect_cells_stage = CollectUsedCellsStage(self.pipeline)
        add_stage(self.collect_cells_stage)

        self.cull_lights_stage = CullLightsStage(self.pipeline)
        add_stage(self.cull_lights_stage)

        self.apply_lights_stage = ApplyLightsStage(self.pipeline)
        add_stage(self.apply_lights_stage)

        self.shadow_stage = ShadowStage(self.pipeline)
        self.shadow_stage.size = self.shadow_manager.get_atlas_size()
        add_stage(self.shadow_stage)

    def init_defines(self):
        """ Inits the common defines """
        defines = self.pipeline.stage_mgr.defines
        defines["LC_TILE_SIZE_X"] = self.tile_size.x
        defines["LC_TILE_SIZE_Y"] = self.tile_size.y
        defines["LC_TILE_SLICES"] = self.pipeline.settings["lighting.culling_grid_slices"]
        defines["LC_MAX_DISTANCE"] = self.pipeline.settings["lighting.culling_max_distance"]
        defines["LC_CULLING_SLICE_WIDTH"] = self.pipeline.settings["lighting.culling_slice_width"]
        defines["LC_MAX_LIGHTS_PER_CELL"] = self.pipeline.settings["lighting.max_lights_per_cell"]
        defines["SHADOW_ATLAS_SIZE"] = self.pipeline.settings["shadows.atlas_size"]

        # Register all light types as defines
        for attr in dir(PointLight):
            if attr.startswith("LT_"):
                defines[attr.upper()] = getattr(PointLight, attr)
