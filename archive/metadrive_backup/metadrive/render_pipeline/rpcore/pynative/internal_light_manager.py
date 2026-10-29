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

from panda3d.core import Vec3

from metadrive.render_pipeline.rpcore.pynative.pointer_slot_storage import PointerSlotStorage
from metadrive.render_pipeline.rpcore.pynative.gpu_command import GPUCommand

MAX_LIGHT_COUNT = 65535
MAX_SHADOW_SOURCES = 2048


class InternalLightManager(object):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """
    def __init__(self):
        self._lights = PointerSlotStorage(MAX_LIGHT_COUNT)
        self._shadow_sources = PointerSlotStorage(MAX_SHADOW_SOURCES)
        self._cmd_list = None
        self._shadow_manager = None
        self._camera_pos = Vec3(0)
        self._shadow_update_distance = 100.0

    def get_max_light_index(self):
        return self._lights.get_max_index()

    max_light_index = property(get_max_light_index)

    def get_num_lights(self):
        return self._lights.get_num_entries()

    num_lights = property(get_num_lights)

    def get_num_shadow_sources(self):
        return self._shadow_sources.get_num_entries()

    num_shadow_sources = property(get_num_shadow_sources)

    def set_shadow_manager(self, shadow_manager):
        self._shadow_manager = shadow_manager

    def get_shadow_manager(self):
        return self._shadow_manager

    shadow_manager = property(get_shadow_manager, set_shadow_manager)

    def set_command_list(self, cmd_list):
        self._cmd_list = cmd_list

    def set_camera_pos(self, pos):
        self._camera_pos = pos

    def set_shadow_update_distance(self, dist):
        self._shadow_update_distance = dist

    def add_light(self, light):
        if light.has_slot():
            print("ERROR: Cannot add light since it already has a slot!")
            return

        slot = self._lights.find_slot()
        if slot < 0:
            print("ERROR: Could not find a free slot for a new light!")
            return

        light.assign_slot(slot)
        self._lights.reserve_slot(slot, light)

        if light.get_casts_shadows():
            self.setup_shadows(light)

        self.gpu_update_light(light)

    def setup_shadows(self, light):
        light.init_shadow_sources()
        light.update_shadow_sources()

        num_sources = light.get_num_shadow_sources()
        base_slot = self._shadow_sources.find_consecutive_slots(num_sources)
        if base_slot < 0:
            print("ERROR: Failed to find slot for shadow sources!")
            return

        for i in range(num_sources):
            source = light.get_shadow_source(i)
            source.set_needs_update(True)
            slot = base_slot + i
            self._shadow_sources.reserve_slot(slot, source)
            source.set_slot(slot)

    def remove_light(self, light):
        assert light is not None
        if not light.has_slot():
            print("ERROR: Could not detach light, light was not attached!")
            return

        self._lights.free_slot(light.get_slot())
        self.gpu_remove_light(light)
        light.remove_slot()

        if light.get_casts_shadows():

            for i in range(light.get_num_shadow_sources()):
                source = light.get_shadow_source(i)
                if source.has_slot():
                    self._shadow_sources.free_slot(source.get_slot())
                if source.has_region():
                    self._shadow_manager.get_atlas().free_region(source.get_region())
                    source.clear_region()

            self.gpu_remove_consecutive_sources(light.get_shadow_source(0), light.get_num_shadow_sources())

            light.clear_shadow_sources()

    def gpu_remove_consecutive_sources(self, first_source, num_sources):
        cmd_remove = GPUCommand(GPUCommand.CMD_remove_sources)
        cmd_remove.push_int(first_source.get_slot())
        cmd_remove.push_int(num_sources)
        self._cmd_list.add_command(cmd_remove)

    def gpu_remove_light(self, light):
        cmd_remove = GPUCommand(GPUCommand.CMD_remove_light)
        cmd_remove.push_int(light.get_slot())
        self._cmd_list.add_command(cmd_remove)

    def gpu_update_light(self, light):
        cmd_update = GPUCommand(GPUCommand.CMD_store_light)
        cmd_update.push_int(light.get_slot())
        light.write_to_command(cmd_update)
        light.set_needs_update(False)
        self._cmd_list.add_command(cmd_update)

    def gpu_update_source(self, source):
        cmd_update = GPUCommand(GPUCommand.CMD_store_source)
        cmd_update.push_int(source.get_slot())
        source.write_to_command(cmd_update)
        self._cmd_list.add_command(cmd_update)

    def update_lights(self):
        for light in self._lights.begin():
            if light.get_needs_update():
                if light.casts_shadows:
                    light.update_shadow_sources()
            self.gpu_update_light(light)

    def update_shadow_sources(self):
        sources_to_update = []

        for source in self._shadow_sources.begin():
            # if source and source.get_needs_update():
            # sources_to_update.append(source)
            if source:
                bounds = source.get_bounds()
                distance_to_camera = (self._camera_pos - bounds.get_center()) - bounds.get_radius()
                if distance_to_camera < self._shadow_update_distance:
                    sources_to_update.append(source)
                else:
                    if source.has_region():
                        self._shadow_manager.get_atlas().free_region(source.get_region())
                        source.clear_region()

        def get_source_score(source):
            dist = (source.get_bounds().get_center() - self._camera_pos).length()
            return -dist + (10**10 if source.has_region() else 0)

        sorted_sources = list(sorted(sources_to_update, key=get_source_score))

        atlas = self._shadow_manager.get_atlas()
        update_slots = min(len(sorted_sources), self._shadow_manager.get_num_update_slots_left())

        for i in range(update_slots):
            if sorted_sources[i].has_region():
                atlas.free_region(sorted_sources[i].get_region())

        for i in range(update_slots):
            source = sorted_sources[i]

            if not self._shadow_manager.add_update(source):
                print("ERROR: Shadow manager ensured update slot, but slot is taken!")
                break

            region_size = atlas.get_required_tiles(source.get_resolution())
            new_region = atlas.find_and_reserve_region(region_size, region_size)
            new_uv_region = atlas.region_to_uv(new_region)
            source.set_region(new_region, new_uv_region)
            source.set_needs_update(False)
            self.gpu_update_source(source)

    def update(self):
        self.update_lights()
        self.update_shadow_sources()
