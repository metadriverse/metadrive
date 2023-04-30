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

from panda3d.core import Texture, Vec3

from direct.gui.DirectScrolledFrame import DirectScrolledFrame
from direct.gui.DirectFrame import DirectFrame

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.util.generic import rgb_from_string
from metadrive.render_pipeline.rpcore.util.display_shader_builder import DisplayShaderBuilder
from metadrive.render_pipeline.rpcore.util.shader_input_blocks import SimpleInputBlock, GroupedInputBlock

from metadrive.render_pipeline.rpcore.gui.draggable_window import DraggableWindow
from metadrive.render_pipeline.rpcore.gui.text import Text
from metadrive.render_pipeline.rpcore.gui.sprite import Sprite


class PipeViewer(DraggableWindow):
    """ Small tool which displays the order of the graphic pipes """

    _STAGE_MGR = None

    @classmethod
    def register_stage_mgr(cls, mgr):
        """ Sets the stage manager, this is a workaround to prevent
        circular imports, since the pipe viewer is already included
        from the StageManager """
        cls._STAGE_MGR = mgr

    def __init__(self, pipeline, parent):
        """ Constructs the pipe viewer """
        DraggableWindow.__init__(self, width=1300, height=900, parent=parent, title="Pipeline Visualizer")
        self._pipeline = pipeline
        self._scroll_width = 8000
        self._scroll_height = 3000
        self._created = False
        self._create_components()
        self.hide()

    def toggle(self):
        """ Toggles the pipe viewer """
        if self._visible:
            Globals.base.taskMgr.remove("RP_GUI_UpdatePipeViewer")
            self.hide()
        else:
            Globals.base.taskMgr.add(self._update_task, "RP_GUI_UpdatePipeViewer")
            if not self._created:
                self._populate_content()
            self.show()

    def _update_task(self, task=None):
        """ Updates the viewer """
        scroll_value = self._content_frame.horizontalScroll["value"]
        scroll_value *= 2.45
        self._pipe_descriptions.set_x(scroll_value * 2759.0)
        return task.cont

    def _populate_content(self):  # pylint: disable=too-many-branches,too-many-statements
        """ Reads the pipes and stages from the stage manager and renders those
        into the window """
        self._created = True
        self._pipe_node = self._content_node.attach_new_node("pipes")
        self._pipe_node.set_scale(1, 1, -1)
        self._stage_node = self._content_node.attach_new_node("stages")
        current_pipes = []
        pipe_pixel_size = 3
        pipe_height = 100

        # Generate stages
        for offs, stage in enumerate(self._STAGE_MGR.stages):
            node = self._content_node.attach_new_node("stage")
            node.set_pos(220 + offs * 200.0, 0, 20)
            node.set_scale(1, 1, -1)
            DirectFrame(parent=node, frameSize=(10, 150, 0, -3600), frameColor=(0.2, 0.2, 0.2, 1))
            Text(text=str(stage.debug_name.replace("Stage", "")), parent=node, x=20, y=25, size=15)

            for output_pipe, pipe_tex in iteritems(stage.produced_pipes):
                pipe_idx = 0
                r, g, b = rgb_from_string(output_pipe)
                if output_pipe in current_pipes:
                    pipe_idx = current_pipes.index(output_pipe)
                else:
                    current_pipes.append(output_pipe)
                    pipe_idx = len(current_pipes) - 1
                    DirectFrame(
                        parent=node,
                        frameSize=(0, 8000, pipe_pixel_size / 2, -pipe_pixel_size / 2),
                        frameColor=(r, g, b, 1),
                        pos=(10, 1, -95 - pipe_idx * pipe_height)
                    )
                w = 160
                h = Globals.native_resolution.y /\
                    float(Globals.native_resolution.x) * w
                DirectFrame(
                    parent=node,
                    frameSize=(
                        -pipe_pixel_size, w + pipe_pixel_size, h / 2 + pipe_pixel_size, -h / 2 - pipe_pixel_size
                    ),
                    frameColor=(r, g, b, 1),
                    pos=(0, 1, -95 - pipe_idx * pipe_height)
                )

                if isinstance(pipe_tex, (list, tuple)):
                    pipe_tex = pipe_tex[0]

                if isinstance(pipe_tex, (SimpleInputBlock, GroupedInputBlock)):
                    icon_file = "/$$rp/data/gui/icon_ubo.png"
                elif pipe_tex.get_z_size() > 1:
                    icon_file = "/$$rp/data/gui/icon_texture.png"
                elif pipe_tex.get_texture_type() == Texture.TT_buffer_texture:
                    icon_file = "/$$rp/data/gui/icon_buffer_texture.png"
                else:
                    icon_file = None
                    preview = Sprite(
                        image=pipe_tex,
                        parent=node,
                        x=0,
                        y=50 + pipe_idx * pipe_height,
                        w=w,
                        h=h,
                        any_filter=False,
                        transparent=False
                    )

                    preview_shader = DisplayShaderBuilder.build(pipe_tex, int(w), int(h))
                    preview.set_shader(preview_shader)

                    preview.set_shader_inputs(mipmap=0, slice=0, brightness=1, tonemap=False)

                if icon_file:
                    Sprite(
                        image=icon_file,
                        parent=node,
                        x=55,
                        y=65 + pipe_idx * pipe_height,
                        w=48,
                        h=48,
                        near_filter=False,
                        transparent=True
                    )

                    if isinstance(pipe_tex, (SimpleInputBlock, GroupedInputBlock)):
                        tex_desc = "UBO"
                    else:
                        tex_desc = pipe_tex.format_texture_type(pipe_tex.get_texture_type())
                        tex_desc += " - " + pipe_tex.format_format(pipe_tex.get_format()).upper()

                    Text(
                        text=tex_desc,
                        parent=node,
                        x=55 + 48 / 2,
                        y=130 + pipe_idx * pipe_height,
                        color=Vec3(0.2),
                        size=12,
                        align="center"
                    )

            for input_pipe in stage.required_pipes:
                if input_pipe not in current_pipes:
                    self.warn("Pipe not found:", input_pipe)
                    continue
                idx = current_pipes.index(input_pipe)
                r, g, b = rgb_from_string(input_pipe)
                DirectFrame(
                    parent=node,
                    frameSize=(0, 10, 40, -40),
                    frameColor=(r, g, b, 1),
                    pos=(5, 1, -95 - idx * pipe_height)
                )

        self._pipe_descriptions = self._content_node.attach_new_node("PipeDescriptions")
        self._pipe_descriptions.set_scale(1, 1, -1)

        DirectFrame(parent=self._pipe_descriptions, frameSize=(0, 190, 0, -5000), frameColor=(0.1, 0.1, 0.1, 1.0))

        # Generate the pipe descriptions
        for idx, pipe in enumerate(current_pipes):
            r, g, b = rgb_from_string(pipe)
            DirectFrame(
                parent=self._pipe_descriptions,
                frameSize=(0, 180, -95, -135),
                frameColor=(r, g, b, 1.0),
                pos=(0, 1, -idx * pipe_height)
            )
            Text(parent=self._pipe_descriptions, text=pipe, x=42, y=121 + idx * pipe_height, size=15, color=Vec3(0.1))
            Sprite(
                parent=self._pipe_descriptions,
                x=9,
                y=103 + idx * pipe_height,
                image="/$$rp/data/gui/icon_pipe.png",
                transparent=True,
                near_filter=False
            )

    def _create_components(self):
        """ Internal method to create the window components """
        DraggableWindow._create_components(self)

        self._content_frame = DirectScrolledFrame(
            frameSize=(0, self._width - 40, 0, self._height - 80),
            canvasSize=(0, self._scroll_width, 0, self._scroll_height),
            autoHideScrollBars=False,
            scrollBarWidth=20.0,
            frameColor=(0, 0, 0, 0),
            verticalScroll_relief=False,
            horizontalScroll_relief=False,
            parent=self._node,
            pos=(20, 1, -self._height + 20)
        )
        self._content_node = self._content_frame.getCanvas().attach_new_node("PipeComponents")
        self._content_node.set_scale(1, 1, -1)
        self._content_node.set_z(self._scroll_height)
