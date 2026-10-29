from __future__ import print_function

import os
import sys
import struct
from panda3d.core import Vec3, load_prc_file_data, Texture, GeomEnums, Vec2
from panda3d.core import OmniBoundingVolume
from direct.showbase.ShowBase import ShowBase

# Change to the current directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Insert the pipeline path to the system path, this is required to be
# able to import the pipeline classes
pipeline_path = "../../"

# Just a special case for my development setup, so I don't accidentally
# commit a wrong path. You can remove this in your own programs.
if not os.path.isfile(os.path.join(pipeline_path, "setup.py")):
    pipeline_path = "../../RenderPipeline/"

sys.path.insert(0, pipeline_path)

# Import the render pipeline class
from metadrive.render_pipeline.rpcore import RenderPipeline

from metadrive.render_pipeline.rpcore.water.projected_water import ProjectedWater

# This is a helper class for better camera movement - see below.
from metadrive.render_pipeline.rpcore.util.movement_controller import MovementController


class WaterOptions:
    """ This class stores all options required for the WaterDisplacement """
    size = 128

    wind_dir = Vec2(0.8, 0.6)
    wind_speed = 1600.0
    wind_dependency = 0.2
    choppy_scale = 1.3
    patch_length = 2000.0
    wave_amplitude = 0.35

    time_scale = 0.8
    normalization_factor = 150.0


class Application(ShowBase):
    def __init__(self):
        # Setup window size and title
        load_prc_file_data(
            "", """
            # win-size 1600 900
            window-title Render Pipeline - Projected Water Example
        """
        )

        # Construct the render pipeline
        self.render_pipeline = RenderPipeline()
        self.render_pipeline.create(self)
        self.render_pipeline.daytime_mgr.time = "8:17"
        # self.render_pipeline.daytime_mgr.time = "12:00"

        # Initialize the projected water
        self.water = ProjectedWater(WaterOptions)
        self.water.setup_water(pipeline=self.render_pipeline, water_level=-3.0)

        # Initialize movement controller, this is a convenience class
        # to provide an improved camera control compared to Panda3Ds default
        # mouse controller.
        self.controller = MovementController(self)
        self.controller.set_initial_position_hpr(Vec3(-23.2, -32.5, 5.3), Vec3(-33.8, -8.3, 0.0))
        self.controller.setup()


Application().run()
