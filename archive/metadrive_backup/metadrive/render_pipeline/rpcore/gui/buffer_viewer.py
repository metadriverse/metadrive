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
from functools import partial

from panda3d.core import Texture, Vec3

from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectGui import DGG
from direct.gui.DirectScrolledFrame import DirectScrolledFrame

from metadrive.render_pipeline.rplibs.six import itervalues

from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.util.display_shader_builder import DisplayShaderBuilder
from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.render_target import RenderTarget
from metadrive.render_pipeline.rpcore.gui.texture_preview import TexturePreview
from metadrive.render_pipeline.rpcore.gui.sprite import Sprite
from metadrive.render_pipeline.rpcore.gui.labeled_checkbox import LabeledCheckbox
from metadrive.render_pipeline.rpcore.gui.text import Text
from metadrive.render_pipeline.rpcore.gui.draggable_window import DraggableWindow


class BufferViewer(DraggableWindow):
    """ This class provides a view into the buffers to inspect them """
    def __init__(self, pipeline, parent):
        """ Constructs the buffer viewer """
        DraggableWindow.__init__(self, width=1400, height=800, parent=parent, title="Buffer- and Image-Browser")
        self._pipeline = pipeline
        self._scroll_height = 3000
        self._display_images = False
        self._stages = []
        self._create_components()
        self._tex_preview = TexturePreview(self._pipeline, parent)
        self._tex_preview.hide()
        self.hide()

    def toggle(self):
        """ Updates all the buffers and then toggles the buffer viewer """
        if self._visible:
            self._remove_components()
            self.hide()
        else:
            self._perform_update()
            self.show()

    @property
    def entries(self):
        """ Returns a list of all registered entries """
        return RenderTarget.REGISTERED_TARGETS + Image.REGISTERED_IMAGES

    @property
    def stage_information(self):
        """ Returns the amount of attached stages, and also the memory consumed
        in MiB in a tuple. """
        count, memory = 0, 0
        for entry in self.entries:
            if isinstance(entry, Texture):
                memory += entry.estimate_texture_memory()
                count += 1
            elif entry.__class__.__name__ == "RenderTarget":
                for target in itervalues(entry.targets):
                    memory += target.estimate_texture_memory()
                    count += 1
            else:
                self.warn("Unkown type:", entry.__class__.__name__)
        return memory, count

    def _create_components(self):
        """ Creates the window components """
        DraggableWindow._create_components(self)

        self._content_frame = DirectScrolledFrame(
            frameSize=(0, self._width - 15, 0, self._height - 70),
            canvasSize=(0, self._width - 80, 0, self._scroll_height),
            autoHideScrollBars=False,
            scrollBarWidth=12.0,
            frameColor=(0, 0, 0, 0),
            verticalScroll_relief=DGG.FLAT,
            verticalScroll_incButton_relief=DGG.FLAT,
            verticalScroll_decButton_relief=DGG.FLAT,
            verticalScroll_thumb_relief=DGG.FLAT,
            verticalScroll_frameColor=(0.05, 0.05, 0.05, 1),
            verticalScroll_thumb_frameColor=(0.8, 0.8, 0.8, 1),
            verticalScroll_incButton_frameColor=(0.6, 0.6, 0.6, 1),
            verticalScroll_decButton_frameColor=(0.6, 0.6, 0.6, 1),
            horizontalScroll_frameColor=(0, 0, 0, 0),
            horizontalScroll_relief=False,
            horizontalScroll_thumb_relief=False,
            horizontalScroll_incButton_relief=False,
            horizontalScroll_decButton_relief=False,
            parent=self._node,
            pos=(0, 1, -self._height)
        )
        self._content_node = self._content_frame.getCanvas().attach_new_node("BufferComponents")
        self._content_node.set_scale(1, 1, -1)
        self._content_node.set_z(self._scroll_height)
        self._chb_show_images = LabeledCheckbox(
            parent=self._node,
            x=10,
            y=43,
            chb_callback=self._set_show_images,
            chb_checked=False,
            text="Display image resources",
            text_color=Vec3(0.4),
            expand_width=330
        )

    def _set_show_images(self, arg):
        """ Sets whether images and textures will be shown """
        self._display_images = arg
        self._perform_update()

    def _set_scroll_height(self, height):
        """ Sets the maximum scroll height in the content frame """
        self._scroll_height = height
        self._content_frame["canvasSize"] = (0, self._width - 80, 0, self._scroll_height)
        self._content_node.set_z(self._scroll_height)

    def _remove_components(self):
        """ Removes all components of the buffer viewer """
        self._content_node.node().remove_all_children()
        self._tex_preview.hide()

    def _perform_update(self):
        """ Collects all entries, extracts their images and re-renders the
        window """

        # Collect texture stages
        self._stages = []
        for entry in sorted(self.entries, key=lambda entry: entry.sort):
            if isinstance(entry, Texture):
                if self._display_images:
                    self._stages.append(entry)
            # Can not use isinstance or we get circular import references
            elif entry.__class__.__name__ == "RenderTarget":
                for target in itervalues(entry.targets):
                    self._stages.append(target)
            else:
                self.warn("Unrecognized instance!", entry.__class__)

        self._render_stages()

    def _on_texture_hovered(self, hover_frame, evt=None):  # pylint: disable=unused-argument
        """ Internal method when a texture is hovered """
        hover_frame["frameColor"] = (0, 0, 0, 0.1)

    def _on_texture_blurred(self, hover_frame, evt=None):  # pylint: disable=unused-argument
        """ Internal method when a texture is blurred """
        hover_frame["frameColor"] = (0, 0, 0, 0)

    def _on_texture_clicked(self, tex_handle, evt=None):  # pylint: disable=unused-argument
        """ Internal method when a texture is blurred """
        self._tex_preview.present(tex_handle)

    def _render_stages(self):
        """ Renders the stages to the window """

        self._remove_components()
        entries_per_row = 6
        aspect = Globals.native_resolution.y / Globals.native_resolution.x
        entry_width = 235
        entry_height = (entry_width - 20) * aspect + 55

        # Store already processed images
        processed = set()
        index = -1
        # Iterate over all stages
        for stage_tex in self._stages:
            if stage_tex in processed:
                continue
            processed.add(stage_tex)
            index += 1
            stage_name = stage_tex.get_name()

            xoffs = index % entries_per_row
            yoffs = index // entries_per_row
            node = self._content_node.attach_new_node("Preview")
            node.set_sz(-1)
            node.set_pos(10 + xoffs * (entry_width - 14), 1, yoffs * (entry_height - 14 + 10))

            r, g, b = 0.2, 0.2, 0.2
            if isinstance(stage_tex, Image):
                r, g, b = 0.2, 0.4, 0.6

            stage_name = stage_name.replace("render_pipeline_internal:", "")
            parts = stage_name.split(":")
            stage_name = parts[-1]
            DirectFrame(
                parent=node,
                frameSize=(7, entry_width - 17, -7, -entry_height + 17),
                frameColor=(r, g, b, 1.0),
                pos=(0, 0, 0)
            )

            frame_hover = DirectFrame(
                parent=node,
                frameSize=(0, entry_width - 10, 0, -entry_height + 10),
                frameColor=(0, 0, 0, 0),
                pos=(0, 0, 0),
                state=DGG.NORMAL
            )
            frame_hover.bind(DGG.ENTER, partial(self._on_texture_hovered, frame_hover))
            frame_hover.bind(DGG.EXIT, partial(self._on_texture_blurred, frame_hover))
            frame_hover.bind(DGG.B1PRESS, partial(self._on_texture_clicked, stage_tex))

            Text(text=stage_name, x=15, y=29, parent=node, size=12, color=Vec3(0.8))

            # Scale image so it always fits
            w, h = stage_tex.get_x_size(), stage_tex.get_y_size()
            padd_x, padd_y = 24, 57
            scale_x = (entry_width - padd_x) / max(1, w)
            scale_y = (entry_height - padd_y) / max(1, h)
            scale_factor = min(scale_x, scale_y)

            if stage_tex.get_texture_type() == Image.TT_buffer_texture:
                scale_factor = 1
                w = entry_width - padd_x
                h = entry_height - padd_y

            preview = Sprite(
                image=stage_tex,
                w=scale_factor * w,
                h=scale_factor * h,
                any_filter=False,
                parent=node,
                x=7,
                y=40,
                transparent=False
            )

            preview.set_shader_inputs(mipmap=0, slice=0, brightness=1, tonemap=False)

            preview_shader = DisplayShaderBuilder.build(stage_tex, scale_factor * w, scale_factor * h)
            preview.set_shader(preview_shader)

        num_rows = (index + entries_per_row) // entries_per_row

        self._set_scroll_height(50 + (entry_height - 14 + 10) * num_rows)
