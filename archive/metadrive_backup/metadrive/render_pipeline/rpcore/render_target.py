"""

RenderTarget

Copyright (c) 2015 tobspr <tobias.springer1@gmail.com>

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

from __future__ import print_function, division

from panda3d.core import GraphicsOutput, Texture, AuxBitplaneAttrib, NodePath
from panda3d.core import Vec4, TransparencyAttrib, ColorWriteAttrib, SamplerState
from panda3d.core import WindowProperties, FrameBufferProperties, GraphicsPipe
from panda3d.core import LVecBase2i

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error
from metadrive.render_pipeline.rplibs.six import iterkeys, itervalues

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.util.post_process_region import PostProcessRegion

__all__ = "RenderTarget",
__version__ = "2.0"


class setter(object):  # noqa # pylint: disable=invalid-name,too-few-public-methods
    """ Setter only property """
    def __init__(self, func):
        self.__func = func
        self.__doc__ = func.__doc__

    def __set__(self, name, value):
        return self.__func(name, value)


class RenderTarget(RPObject):
    """ Second version of the RenderTarget library, provides functions
    to easily setup buffers in Panda3D. """

    NUM_ALLOCATED_BUFFERS = 0
    USE_R11G11B10 = True
    REGISTERED_TARGETS = []
    CURRENT_SORT = -300

    def __init__(self, name="target"):
        RPObject.__init__(self, name)
        self._targets = {}
        self._color_bits = (0, 0, 0, 0)
        self._aux_bits = 8
        self._aux_count = 0
        self._depth_bits = 0
        self._size = LVecBase2i(-1)
        self._size_constraint = LVecBase2i(-1)
        self._source_window = Globals.base.win
        self._source_region = None
        self._active = False
        self._internal_buffer = None
        self.sort = None

        # Public attributes
        self.engine = Globals.base.graphicsEngine
        self.support_transparency = False
        self.create_default_region = True

        # Disable all global clears, since they are not required
        for region in Globals.base.win.get_display_regions():
            region.disable_clears()

    def add_color_attachment(self, bits=8, alpha=False):
        """ Adds a new color attachment with the given amount of bits, bits can
        be either a single int or a tuple determining the bits. If bits is a
        single int, alpha determines whether alpha bits are requested """
        self._targets["color"] = Texture(self.debug_name + "_color")
        if isinstance(bits, (list, tuple)):
            self._color_bits = (bits[0], bits[1], bits[2], bits[3] if len(bits) == 4 else 0)
        else:
            self._color_bits = ((bits, bits, bits, (bits if alpha else 0)))

    def add_depth_attachment(self, bits=32):
        """ Adds a depth attachment wit the given amount of bits """
        self._targets["depth"] = Texture(self.debug_name + "_depth")
        self._depth_bits = bits

    def add_aux_attachment(self, bits=8):
        """ Adds a new aux attachment with the given amount of bits. The amount
        of bits passed overrides all previous bits set, since all aux textures
        have to have the same amount of bits. """
        self._aux_bits = bits
        self._aux_count += 1

    def add_aux_attachments(self, bits=8, count=1):
        """ Adds n new aux attachments, with the given amount of bits. All
        previously set aux bits are overriden, since all aux textures have to
        have the same amount of bits """
        self._aux_bits = bits
        self._aux_count += count

    @setter
    def size(self, *args):
        """ Sets the render target size. This can be either a single integer,
        in which case it applies to both dimensions. Negative integers cause
        the render target to be proportional to the screen size, i.e. a value
        of -4 produces a quarter resolution target, a value of -2 a half
        resolution target, and a value of -1 a full resolution target
        (the default). """
        self._size_constraint = LVecBase2i(*args)

    @property
    def active(self):
        """ Returns whether the target is currently active """
        return self._active

    @active.setter
    def active(self, flag):
        """ Sets whether the target is active, this just propagates the active
        flag to all display regions """
        for region in self._internal_buffer.get_display_regions():
            region.set_active(flag)

    @property
    def color_tex(self):
        """ Returns the color attachment if present """
        return self._targets["color"]

    @property
    def depth_tex(self):
        """ Returns the depth attachment if present """
        return self._targets["depth"]

    @property
    def aux_tex(self):
        """ Returns a list of aux textures, can be used like target.aux_tex[2],
        notice the indices start at zero, so the first target has the index 0. """
        return [self._targets[i] for i in sorted(iterkeys(self._targets)) if i.startswith("aux_")]

    def set_shader_input(self, *args, **kwargs):
        """ Sets a shader input available to the target """
        if self.create_default_region:
            self._source_region.set_shader_input(*args, **kwargs)

    def set_shader_inputs(self, **kwargs):
        """ Sets shader inputs available to the target """
        if self.create_default_region:
            self._source_region.set_shader_inputs(**kwargs)

    @setter
    def shader(self, shader_obj):
        """ Sets a shader on the target """
        if not shader_obj:
            self.error("shader must not be None!")
            return
        self._source_region.set_shader(shader_obj)

    @property
    def internal_buffer(self):
        """ Returns a handle to the internal GraphicsBuffer object """
        return self._internal_buffer

    @property
    def targets(self):
        """ Returns the dictionary of attachments, whereas the key is the name
        of the attachment and the value is the Texture handle of the attachment """
        return self._targets

    @property
    def region(self):
        """ Returns the internally used PostProcessRegion """
        return self._source_region

    def prepare_render(self, camera_np):
        """ Prepares to render a scene """
        self.create_default_region = False
        self._create_buffer()
        self._source_region = self._internal_buffer.get_display_region(0)

        if camera_np:
            initial_state = NodePath("rtis")
            initial_state.set_state(camera_np.node().get_initial_state())

            if self._aux_count:
                initial_state.set_attrib(AuxBitplaneAttrib.make(self._aux_bits), 20)
            initial_state.set_attrib(TransparencyAttrib.make(TransparencyAttrib.M_none), 20)

            if max(self._color_bits) == 0:
                initial_state.set_attrib(ColorWriteAttrib.make(ColorWriteAttrib.C_off), 20)

            # Disable existing regions of the camera
            for region in camera_np.node().get_display_regions():
                region.set_active(False)

            # Remove the existing display region of the camera
            for region in self._source_window.get_display_regions():
                if region.get_camera() == camera_np:
                    self._source_window.remove_display_region(region)

            camera_np.node().set_initial_state(initial_state.get_state())
            self._source_region.set_camera(camera_np)

        self._internal_buffer.disable_clears()
        self._source_region.disable_clears()
        self._source_region.set_active(True)
        self._source_region.set_sort(20)

        # Reenable depth-clear, usually desireable
        self._source_region.set_clear_depth_active(True)
        self._source_region.set_clear_depth(1.0)
        self._active = True

    def prepare_buffer(self):
        """ Prepares the target to render to an offscreen buffer """
        self._create_buffer()
        self._active = True

    def present_on_screen(self):
        """ Prepares the target to render on the main window, to present the
        final rendered image """
        self._source_region = PostProcessRegion.make(self._source_window)
        self._source_region.set_sort(5)

    def remove(self):
        """ Deletes this buffer, restoring the previous state """
        self._internal_buffer.clear_render_textures()
        self.engine.remove_window(self._internal_buffer)
        self._active = False
        for target in itervalues(self._targets):
            target.release_all()
        RenderTarget.REGISTERED_TARGETS.remove(self)

    def set_clear_color(self, *args):
        """ Sets the  clear color """
        self._internal_buffer.set_clear_color_active(True)
        self._internal_buffer.set_clear_color(Vec4(*args))

    @setter
    def instance_count(self, count):
        """ Sets the instance count """
        self._source_region.set_instance_count(count)

    def _create_buffer(self):
        """ Internal method to create the buffer object """
        self._compute_size_from_constraint()
        if not self._create():
            self.error("Failed to create buffer!")
            return False

        if self.create_default_region:
            self._source_region = PostProcessRegion.make(self._internal_buffer)

            if max(self._color_bits) == 0:
                self._source_region.set_attrib(ColorWriteAttrib.make(ColorWriteAttrib.M_none), 1000)

    def _compute_size_from_constraint(self):
        """ Computes the actual size in pixels from the targets size constraint """
        w, h = Globals.resolution.x, Globals.resolution.y
        self._size = LVecBase2i(self._size_constraint)
        if self._size_constraint.x < 0:
            self._size.x = (w - self._size_constraint.x - 1) // (-self._size_constraint.x)
        if self._size_constraint.y < 0:
            self._size.y = (h - self._size_constraint.y - 1) // (-self._size_constraint.y)

    def _setup_textures(self):
        """ Prepares all bound textures """
        for i in range(self._aux_count):
            self._targets["aux_{}".format(i)] = Texture(self.debug_name + "_aux{}".format(i))
        for tex in itervalues(self._targets):
            tex.set_wrap_u(SamplerState.WM_clamp)
            tex.set_wrap_v(SamplerState.WM_clamp)
            tex.set_anisotropic_degree(0)
            tex.set_x_size(self._size.x)
            tex.set_y_size(self._size.y)
            tex.set_minfilter(SamplerState.FT_linear)
            tex.set_magfilter(SamplerState.FT_linear)

    def _make_properties(self):
        """ Creates the window and buffer properties """
        window_props = WindowProperties.size(self._size.x, self._size.y)
        buffer_props = FrameBufferProperties()

        if self._size_constraint.x == 0 or self._size_constraint.y == 0:
            window_props = WindowProperties.size(1, 1)

        if self._color_bits == (16, 16, 16, 0):
            if RenderTarget.USE_R11G11B10:
                buffer_props.set_rgba_bits(11, 11, 10, 0)
            else:
                buffer_props.set_rgba_bits(*self._color_bits)
        elif 8 in self._color_bits:
            # When specifying 8 bits, specify 1 bit, this is a workarround
            # to a legacy logic in panda
            buffer_props.set_rgba_bits(*[i if i != 8 else 1 for i in self._color_bits])
        else:
            buffer_props.set_rgba_bits(*self._color_bits)

        buffer_props.set_accum_bits(0)
        buffer_props.set_stencil_bits(0)
        buffer_props.set_back_buffers(0)
        buffer_props.set_coverage_samples(0)
        buffer_props.set_depth_bits(self._depth_bits)

        if self._depth_bits == 32:
            buffer_props.set_float_depth(True)

        buffer_props.set_float_color(max(self._color_bits) > 8)

        buffer_props.set_force_hardware(True)
        buffer_props.set_multisamples(0)
        buffer_props.set_srgb_color(False)
        buffer_props.set_stereo(False)
        buffer_props.set_stencil_bits(0)

        if self._aux_bits == 8:
            buffer_props.set_aux_rgba(self._aux_count)
        elif self._aux_bits == 16:
            buffer_props.set_aux_hrgba(self._aux_count)
        elif self._aux_bits == 32:
            buffer_props.set_aux_float(self._aux_count)
        else:
            self.error("Invalid aux bits")

        return window_props, buffer_props

    def _create(self):
        """ Creates the internally used buffer """
        self._setup_textures()
        window_props, buffer_props = self._make_properties()

        self._internal_buffer = self.engine.make_output(
            self._source_window.get_pipe(), self.debug_name, 1, buffer_props, window_props,
            GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_resizeable, self._source_window.gsg, self._source_window
        )

        if not self._internal_buffer:
            self.error("Failed to create buffer")
            return

        if self._depth_bits:
            self._internal_buffer.add_render_texture(
                self.depth_tex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_depth
            )

        if max(self._color_bits) > 0:
            self._internal_buffer.add_render_texture(
                self.color_tex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_color
            )

        aux_prefix = {
            8: "RTP_aux_rgba_{}",
            16: "RTP_aux_hrgba_{}",
            32: "RTP_aux_float_{}",
        }[self._aux_bits]

        for i in range(self._aux_count):
            target_mode = getattr(GraphicsOutput, aux_prefix.format(i))
            self._internal_buffer.add_render_texture(self.aux_tex[i], GraphicsOutput.RTM_bind_or_copy, target_mode)

        if not self.sort:
            RenderTarget.CURRENT_SORT += 20
            self.sort = RenderTarget.CURRENT_SORT

        RenderTarget.NUM_ALLOCATED_BUFFERS += 1
        self._internal_buffer.set_sort(self.sort)
        self._internal_buffer.disable_clears()
        self._internal_buffer.get_display_region(0).disable_clears()
        self._internal_buffer.get_overlay_display_region().disable_clears()
        self._internal_buffer.get_overlay_display_region().set_active(False)

        RenderTarget.REGISTERED_TARGETS.append(self)
        return True

    def consider_resize(self):
        """ Checks if the target has to get resized, and if this is the case,
        performs the resize. This should be called when the window resolution
        changed. """
        current_size = LVecBase2i(self._size)
        self._compute_size_from_constraint()
        if current_size != self._size:
            if self._internal_buffer:
                self._internal_buffer.set_size(self._size.x, self._size.y)
