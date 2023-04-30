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

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error

from panda3d.core import Camera, MatrixLens

from metadrive.render_pipeline.rpcore.pynative.shadow_atlas import ShadowAtlas


class ShadowManager(object):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """
    def __init__(self):
        self._max_updates = 10
        self._atlas = None
        self._atlas_size = 4096
        self._tag_state_mgr = None
        self._atlas_graphics_output = None
        self._display_regions = []
        self._queued_updates = []
        self._cameras = []
        self._camera_nps = []

    def set_max_updates(self, max_updates):
        if max_updates == 0:
            print("Warning: max_updates set to 0, no shadow updates will happen")
        self._max_updates = max_updates

    def set_atlas_size(self, atlas_size):
        self._atlas_size = atlas_size

    def get_atlas_size(self):
        return self._atlas_size

    atlas_size = property(get_atlas_size, set_atlas_size)

    def set_scene(self, scene_parent):
        self._scene_parent = scene_parent

    def set_tag_state_manager(self, tag_mgr):
        self._tag_state_mgr = tag_mgr

    def set_atlas_graphics_output(self, graphics_output):
        self._atlas_graphics_output = graphics_output

    def get_num_update_slots_left(self):
        return self._max_updates - len(self._queued_updates)

    num_update_slots_left = property(get_num_update_slots_left)

    def get_atlas(self):
        return self._atlas

    atlas = property(get_atlas)

    def init(self):
        for i in range(self._max_updates):
            camera = Camera("ShadowCam-" + str(i))
            camera.set_lens(MatrixLens())
            camera.set_active(False)
            camera.set_scene(self._scene_parent)
            self._tag_state_mgr.register_camera("shadow", camera)
            self._camera_nps.append(self._scene_parent.attach_new_node(camera))
            self._cameras.append(camera)

            region = self._atlas_graphics_output.make_display_region()
            region.set_sort(1000)
            region.set_clear_depth_active(True)
            region.set_clear_depth(1.0)
            region.set_clear_color_active(False)
            region.set_camera(self._camera_nps[i])
            region.set_active(False)
            self._display_regions.append(region)

        self._atlas = ShadowAtlas(self._atlas_size)

    def update(self):
        for i in range(len(self._queued_updates), self._max_updates):
            self._cameras[i].set_active(False)
            self._display_regions[i].set_active(False)

        for i, source in enumerate(self._queued_updates):
            self._cameras[i].set_active(True)
            self._display_regions[i].set_active(True)
            self._cameras[i].get_lens().set_user_mat(source.get_mvp())
            uv = source.get_uv_region()
            self._display_regions[i].set_dimensions(uv.x, uv.x + uv.z, uv.y, uv.y + uv.w)

        self._queued_updates = []

    def add_update(self, source):
        if len(self._queued_updates) >= self._max_updates:
            return False
        self._queued_updates.append(source)
        return True
