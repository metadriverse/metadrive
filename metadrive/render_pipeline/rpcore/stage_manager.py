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

from metadrive.render_pipeline.rplibs.six import iteritems
from metadrive.render_pipeline.rplibs.yaml import load_yaml_file

from direct.stdpy.file import open

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.gui.pipe_viewer import PipeViewer
from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.util.shader_input_blocks import SimpleInputBlock, GroupedInputBlock
from metadrive.render_pipeline.rpcore.stages.update_previous_pipes_stage import UpdatePreviousPipesStage


class StageManager(RPObject):
    """ This manager takes a list of RenderStages and puts them into an order,
    while connecting the different pipes, inputs, ubos and defines. """
    def __init__(self, pipeline):
        """ Constructs the stage manager """
        RPObject.__init__(self)
        self.stages = []
        self.inputs = {}
        self.pipes = {}
        self.input_blocks = []
        self.previous_pipes = {}
        self.future_bindings = []
        self.defines = {}
        self.pipeline = pipeline
        self.created = False

        self._load_stage_order()

        # Register the manager so the pipe viewer can read our data
        PipeViewer.register_stage_mgr(self)

    def _load_stage_order(self):
        """ Loads the order of all stages from the stages.yaml configuration
        file """
        orders = load_yaml_file("/$$rpconfig/stages.yaml")
        if "global_stage_order" not in orders:
            self.error("Could not load stage order, root key does not exist!")
            return
        self._stage_order = orders["global_stage_order"]

    def add_stage(self, stage):
        """ Adds a new stage """
        if stage.stage_id not in self._stage_order:
            self.error("The stage type", stage.debug_name, "is not registered yet! Please add it to the StageManager!")
            return

        if self.created:
            self.error("Cannot attach stage, stages are already created!")
            return

        self.stages.append(stage)

    def get_stage(self, stage_class):
        """ Returns a handle to an instantiated stage """
        for stage in self.stages:
            if stage.__class__.__name__ == stage_class:
                return stage

    def _prepare_stages(self):
        """ Prepares all stages by removing disabled stages and sorting stages
        by order """
        self.debug("Preparing stages ..")

        # Remove all disabled stages
        to_remove = []
        for stage in self.stages:
            if stage.disabled:
                to_remove.append(stage)

        for stage in to_remove:
            self.stages.remove(stage)

        self.stages.sort(key=lambda stage: self._stage_order.index(stage.stage_id))

    def _bind_pipes_to_stage(self, stage):
        """ Sets all required pipes on a stage """
        for pipe in stage.required_pipes:

            # Check if there is an input block named like the pipe
            if pipe in self.input_blocks:
                self.input_blocks[pipe].bind_to(stage)
                continue

            if pipe.startswith("PreviousFrame::"):
                # Special case: Pipes from the previous frame. We assume those
                # pipes have the same size as the window and a format of
                # F_rgba16. Could be subject to change.
                pipe_name = pipe.split("::")[-1]
                if pipe_name not in self.previous_pipes:
                    tex_format = "RGBA16"

                    # XXX: Assuming we have a depth texture whenever "depth"
                    # occurs in the textures name
                    if "depth" in pipe_name.lower():
                        tex_format = "R32"

                    pipe_tex = Image.create_2d("Prev-" + pipe_name, 0, 0, tex_format)
                    pipe_tex.clear_image()
                    self.previous_pipes[pipe_name] = pipe_tex
                stage.set_shader_input("Previous_" + pipe_name, self.previous_pipes[pipe_name])
                continue

            elif pipe.startswith("FuturePipe::"):
                # Special case: Future Pipes which are not available yet.
                # They will contain the unmodified data from the last
                # frame.
                pipe_name = pipe.split("::")[-1]
                self.debug("Awaiting future pipe", pipe_name)
                self.future_bindings.append((pipe_name, stage))
                continue

            if pipe not in self.pipes:
                self.fatal("Pipe '" + pipe + "' is missing for", stage)
                return False

            pipe_value = self.pipes[pipe]
            if isinstance(pipe_value, list) or isinstance(pipe_value, tuple):
                stage.set_shader_input(pipe, *pipe_value)
            else:
                stage.set_shader_input(pipe, pipe_value)
        return True

    def _bind_inputs_to_stage(self, stage):
        """ Binds all inputs including common inputs to the given stage """
        common_inputs = ["mainCam", "mainRender", "MainSceneData", "TimeOfDay"]
        for input_binding in stage.required_inputs + common_inputs:
            if input_binding not in self.inputs and \
               input_binding not in self.input_blocks:
                self.error("Input", input_binding, "is missing for", stage)
                continue

            if input_binding in self.inputs:
                stage.set_shader_input(input_binding, self.inputs[input_binding])
            elif input_binding in self.input_blocks:
                self.input_blocks[input_binding].bind_to(stage)
            else:
                assert False, "Input binding not in inputs and not in blocks!"
        return True

    def _register_stage_result(self, stage):
        """ Registers all produced pipes, inputs and defines from the given
        stage, so they can be used by later stages. """
        for pipe_name, pipe_data in (iteritems)(stage.produced_pipes):
            if isinstance(pipe_data, (SimpleInputBlock, GroupedInputBlock)):
                self.input_blocks[pipe_name] = pipe_data
                continue
            self.pipes[pipe_name] = pipe_data

        for define_name, data in iteritems(stage.produced_defines):
            if define_name in self.defines:
                self.warn("Stage", stage, "overrides define", define_name)
            self.defines[define_name] = data

        for input_name, data in iteritems(stage.produced_inputs):
            if input_name in self.inputs:
                self.warn("Stage", stage, "overrides input", input_name)

            if isinstance(data, (SimpleInputBlock, GroupedInputBlock)):
                self.input_blocks[input_name] = data
                continue

            self.inputs[input_name] = data

    def _create_previous_pipes(self):
        """ Creates a target for each last-frame's pipe, any pipe starting
        with the prefix 'Previous::' has to be stored and copied each frame. """
        if self.previous_pipes:
            self._prev_stage = UpdatePreviousPipesStage(self.pipeline)
            for prev_pipe, prev_tex in iteritems(self.previous_pipes):

                if prev_pipe not in self.pipes:
                    self.error(
                        "Attempted to use previous frame data from pipe", prev_pipe,
                        "- however, that pipe was never created!"
                    )
                    return False

                # Tell the stage to transfer the data from the current pipe to
                # the current texture
                self._prev_stage.add_transfer(self.pipes[prev_pipe], prev_tex)
            self._prev_stage.create()
            self._prev_stage.set_dimensions()
            self.stages.append(self._prev_stage)

    def _apply_future_bindings(self):
        """ Applies all future bindings. At this point all pipes and
        inputs should be present """
        for pipe, stage in self.future_bindings:
            if pipe not in self.pipes:
                self.error("Could not bind future pipe:", pipe, "not present!")
                continue
            stage.set_shader_input(pipe, self.pipes[pipe])
        self.future_bindings = []

    def setup(self):
        """ Setups the stages """
        self.debug("Setup stages ..")
        self.created = True

        # Convert input blocks so we can access them in a better way
        self.input_blocks = {block.name: block for block in self.input_blocks}
        self._prepare_stages()

        for stage in self.stages:
            stage.create()
            stage.handle_window_resize()

            # Rely on the methods to print an appropriate error message
            if not self._bind_pipes_to_stage(stage):
                continue
            if not self._bind_inputs_to_stage(stage):
                continue

            self._register_stage_result(stage)
        self._create_previous_pipes()
        self._apply_future_bindings()

    def reload_shaders(self):
        """ This pass sets the shaders to all passes and also generates the
        shader configuration """
        self.write_autoconfig()
        for stage in self.stages:
            stage.reload_shaders()

    def update(self):
        """ Calls the update method for each registered stage. Inactive stages
        are skipped. """
        for stage in self.stages:
            if stage.active:
                stage.update()

    def handle_window_resize(self):
        """ Method to get called when the window got resized. Propagates the
        resize event to all registered stages """
        for stage in self.stages:
            stage.handle_window_resize()

    def write_autoconfig(self):
        """ Writes the shader auto config, based on the defines specified by the
        different stages """
        self.debug("Writing shader config")

        # Generate autoconfig as string
        output = "#pragma once\n\n"
        output += "// Autogenerated by the render pipeline\n"
        output += "// Do not edit! Your changes will be lost.\n\n"

        for key, value in sorted(iteritems(self.defines)):
            if isinstance(value, bool):
                value = 1 if value else 0
            output += "#define " + key + " " + str(value) + "\n"

        try:
            with open("/$$rptemp/$$pipeline_shader_config.inc.glsl", "w") as handle:
                handle.write(output)
        except IOError as msg:
            self.error("Error writing shader autoconfig:", msg)
