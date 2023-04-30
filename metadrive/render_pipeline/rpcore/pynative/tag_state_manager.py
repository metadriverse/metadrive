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

from __future__ import print_function
from panda3d.core import RenderState, ColorWriteAttrib, ShaderAttrib, BitMask32

from metadrive.render_pipeline.rplibs.six import itervalues


class TagStateManager(object):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """
    class StateContainer(object):  # pylint: disable=too-few-public-methods
        def __init__(self, tag_name, mask, write_color):
            self.cameras = []
            self.tag_states = {}
            self.tag_name = tag_name
            self.mask = BitMask32.bit(mask)
            self.write_color = write_color

    def __init__(self, main_cam_node):
        self._main_cam_node = main_cam_node
        self._main_cam_node.node().set_camera_mask(BitMask32.bit(1))
        self.containers = {
            "shadow": self.StateContainer("Shadows", 2, False),
            "voxelize": self.StateContainer("Voxelize", 3, False),
            "envmap": self.StateContainer("Envmap", 4, True),
            "forward": self.StateContainer("Forward", 5, True),
        }

    def get_mask(self, container_name):
        if container_name == "gbuffer":
            return BitMask32.bit(1)
        return self.containers[container_name].mask

    def apply_state(self, container_name, np, shader, name, sort):
        assert shader
        state = RenderState.make_empty()
        container = self.containers[container_name]

        if not container.write_color:
            state = state.set_attrib(ColorWriteAttrib.make(ColorWriteAttrib.C_off), 10000)

        state = state.set_attrib(ShaderAttrib.make(shader, sort), sort)
        container.tag_states[name] = state
        np.set_tag(container.tag_name, name)

        for camera in container.cameras:
            camera.set_tag_state(name, state)

    def cleanup_states(self):
        self._main_cam_node.node().clear_tag_states()
        for container in itervalues(self.containers):
            for camera in container.cameras:
                camera.clear_tag_states()
            container.tag_states = {}

    def register_camera(self, container_name, source):
        container = self.containers[container_name]
        source.set_tag_state_key(container.tag_name)
        source.set_camera_mask(container.mask)
        state = RenderState.make_empty()
        if not container.write_color:
            state = state.set_attrib(ColorWriteAttrib.make(ColorWriteAttrib.C_off), 10000)
        source.set_initial_state(state)
        container.cameras.append(source)

    def unregister_camera(self, container_name, source):
        container = self.containers[container_name]
        if source not in container.cameras:
            print("Could not remove source, was never attached!")
            return

        container.cameras.remove(source)
        source.clear_tag_states()
        source.set_initial_state(RenderState.make_empty())
