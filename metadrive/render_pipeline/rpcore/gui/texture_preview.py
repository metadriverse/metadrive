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

from panda3d.core import Vec3

from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.gui.draggable_window import DraggableWindow
from metadrive.render_pipeline.rpcore.gui.sprite import Sprite
from metadrive.render_pipeline.rpcore.gui.text import Text
from metadrive.render_pipeline.rpcore.gui.slider import Slider
from metadrive.render_pipeline.rpcore.gui.labeled_checkbox import LabeledCheckbox

from metadrive.render_pipeline.rpcore.util.display_shader_builder import DisplayShaderBuilder


class TexturePreview(DraggableWindow):
    """ Small window which provides a preview of a texture """
    def __init__(self, pipeline, parent):
        DraggableWindow.__init__(self, width=1600, height=900, parent=parent, title="Texture Viewer")
        self._pipeline = pipeline
        self._current_tex = None
        self._mip_slider = None
        self._mip_text = None
        self._slice_slider = None
        self._slice_text = None
        self._preview_image = None
        self._create_components()

    def present(self, tex):
        """ "Presents" a given texture and shows the window """
        self._current_tex = tex

        self.set_title(tex.get_name())

        # tex.write(tex.get_name() + ".png")

        # Remove old content
        self._content_node.node().remove_all_children()

        w, h = tex.get_x_size(), tex.get_y_size()
        if h > 1:
            scale_x = (self._width - 40.0) / w
            scale_y = (self._height - 110.0) / h
            scale_f = min(scale_x, scale_y)
            display_w = scale_f * w
            display_h = scale_f * h

        else:
            display_w = self._width - 40
            display_h = self._height - 110

        image = Sprite(
            image=tex,
            parent=self._content_node,
            x=20,
            y=90,
            w=display_w,
            h=display_h,
            any_filter=False,
            transparent=False
        )
        description = ""

        # Image size
        description += "{:d} x {:d} x {:d}".format(tex.get_x_size(), tex.get_y_size(), tex.get_z_size())

        # Image type
        description += ", {:s}, {:s}".format(
            Image.format_format(tex.get_format()).upper(),
            Image.format_component_type(tex.get_component_type()).upper()
        )

        Text(text=description, parent=self._content_node, x=17, y=70, size=16, color=Vec3(0.6, 0.6, 0.6))

        estimated_bytes = tex.estimate_texture_memory()
        size_desc = "Estimated memory: {:2.2f} MB".format(estimated_bytes / (1024.0**2))

        Text(
            text=size_desc,
            parent=self._content_node,
            x=self._width - 20.0,
            y=70,
            size=18,
            color=Vec3(0.34, 0.564, 0.192),
            align="right"
        )

        x_pos = len(size_desc) * 9 + 140

        # Slider for viewing different mipmaps
        if tex.uses_mipmaps():
            max_mips = tex.get_expected_num_mipmap_levels() - 1
            self._mip_slider = Slider(
                parent=self._content_node,
                size=140,
                min_value=0,
                max_value=max_mips,
                callback=self._set_mip,
                x=x_pos,
                y=65,
                value=0
            )
            x_pos += 140 + 5

            self._mip_text = Text(
                text="MIP: 5", parent=self._content_node, x=x_pos, y=72, size=18, color=Vec3(1, 0.4, 0.4), may_change=1
            )
            x_pos += 50 + 30

        # Slider for viewing different Z-layers
        if tex.get_z_size() > 1:
            self._slice_slider = Slider(
                parent=self._content_node,
                size=250,
                min_value=0,
                max_value=tex.get_z_size() - 1,
                callback=self._set_slice,
                x=x_pos,
                y=65,
                value=0
            )
            x_pos += 250 + 5

            self._slice_text = Text(
                text="Z: 5", parent=self._content_node, x=x_pos, y=72, size=18, color=Vec3(0.4, 1, 0.4), may_change=1
            )

            x_pos += 50 + 30

        # Slider to adjust brightness
        self._bright_slider = Slider(
            parent=self._content_node,
            size=140,
            min_value=-14,
            max_value=14,
            callback=self._set_brightness,
            x=x_pos,
            y=65,
            value=0
        )
        x_pos += 140 + 5
        self._bright_text = Text(
            text="Bright: 1", parent=self._content_node, x=x_pos, y=72, size=18, color=Vec3(0.4, 0.4, 1), may_change=1
        )
        x_pos += 100 + 30

        # Slider to enable reinhard tonemapping
        self._tonemap_box = LabeledCheckbox(
            parent=self._content_node,
            x=x_pos,
            y=60,
            text="Tonemap",
            text_color=Vec3(1, 0.4, 0.4),
            chb_checked=False,
            chb_callback=self._set_enable_tonemap,
            text_size=18,
            expand_width=90
        )
        x_pos += 90 + 30

        image.set_shader_inputs(slice=0, mipmap=0, brightness=1, tonemap=False)

        preview_shader = DisplayShaderBuilder.build(tex, display_w, display_h)
        image.set_shader(preview_shader)

        self._preview_image = image
        self.show()

    def _set_slice(self):
        idx = int(self._slice_slider.value)
        self._preview_image.set_shader_input("slice", idx)
        self._slice_text.set_text("Z: " + str(idx))

    def _set_mip(self):
        idx = int(self._mip_slider.value)
        self._preview_image.set_shader_input("mipmap", idx)
        self._mip_text.set_text("MIP " + str(idx))

    def _set_brightness(self):
        val = self._bright_slider.value
        scale = 2**val
        self._bright_text.set_text("Bright: " + str(round(scale, 3)))
        self._preview_image.set_shader_input("brightness", scale)

    def _set_enable_tonemap(self, enable_tonemap):
        self._preview_image.set_shader_input("tonemap", enable_tonemap)

    def _create_components(self):
        """ Internal method to init the components """
        DraggableWindow._create_components(self)
        self._content_node = self._node.attach_new_node("content")
