# Copyright (c) 2020 kitsune.ONE team.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import math

from panda3d.core import CS_linear, PNMImage, PNMFileTypeRegistry, LMatrix4d
from panda3d.egg import (EggData, EggComment, EggGroup, EggPolygon, EggVertex, EggVertexPool, EggTexture)


class CardMaker(object):
    def __init__(self, animations_frames, images, empty=None, prefix=None, fps=30, scale=1, type='png'):
        self.animations = {}
        for i, frames in enumerate(animations_frames):
            self.animations[i] = frames
        self.images = images
        if empty:
            self.images.insert(0, empty)
        self.prefix = prefix or ''
        self.fps = fps
        self.scale = scale
        self.type = type
        self.empty = empty

        self.index_poly = 0
        self.index_vertex = 0

    def _get_frames_num(self):
        frames_total = sum(self.animations.values())
        if self.empty:
            frames_total += 1
        return frames_total

    def _add_polygon(self, name=None, group_main=None, egg_texture=None, egg_vertex_pool=None, is_empty=False):
        frames_total = self._get_frames_num()
        frames_rows = math.ceil(math.sqrt(frames_total))  # rows == columns

        # vertical mirror UV, because spritesheet goes from top to bottom
        y = (frames_rows - 1) - math.floor(self.index_poly / frames_rows)
        x = self.index_poly % frames_rows

        size_tile = 1 / frames_rows
        coords_uvs = (
            ((-1, 1, 0), (x * size_tile, (y + 1) * size_tile)),
            ((-1, -1, 0), (x * size_tile, y * size_tile)),
            ((1, -1, 0), ((x + 1) * size_tile, y * size_tile)),
            ((1, 1, 0), ((x + 1) * size_tile, (y + 1) * size_tile)),
        )
        if not name:
            self.empty_uvs = coords_uvs
            return

        if is_empty:
            coords_uvs = self.empty_uvs

        vertices = []
        for coord, uv in coords_uvs:
            vertex = EggVertex()
            vertex.set_pos(coord)
            vertex.set_uv(uv)

            egg_vertex_pool.add_vertex(vertex, self.index_vertex)
            self.index_vertex += 1

            vertices.append(vertex)

        poly = EggPolygon(name)
        poly.add_texture(egg_texture)
        for vertex in vertices:
            poly.add_vertex(vertex)

        group_poly = EggGroup(name)
        group_poly.add_child(poly)
        group_main.add_child(group_poly)

        self.index_poly += 1

    def _add_geom(self, name, count, egg_texture, egg_vertex_pool):
        egg_group = EggGroup(name)
        egg_group.set_switch_flag(True)  # enable animation
        egg_group.set_switch_fps(self.fps)  # animation fps

        for i in range(count):
            self._add_polygon('{}_frame_{:04d}'.format(name, self.index_poly), egg_group, egg_texture, egg_vertex_pool)

        if self.empty:
            self._add_polygon(
                '{}_frame_{:04d}'.format(name, self.index_poly), egg_group, egg_texture, egg_vertex_pool, is_empty=True
            )

        return egg_group

    def _make_spritesheet(self, path):
        reg = PNMFileTypeRegistry.get_global_ptr()
        ftype = reg.get_type_from_extension(os.path.basename(path))
        frames_total = self._get_frames_num()
        frames_rows = math.ceil(math.sqrt(frames_total))  # rows == columns
        frame_size = 0

        spritesheet = None

        i = 0
        while True:
            if i >= len(self.images):
                break
            ipath = self.images[i]

            image = PNMImage()
            image.read(ipath)

            if not frame_size:
                frame_size = image.get_read_x_size()

            if spritesheet is None:
                size = int(frames_rows * frame_size)
                spritesheet = PNMImage(size, size, 4, 255, ftype, CS_linear)
                spritesheet.add_alpha()

            y = i // frames_rows
            x = i % frames_rows

            spritesheet.blend_sub_image(
                image,
                x * frame_size,
                y * frame_size,  # (x, y) to
                0,
                0,  # (x, y) from
                frame_size,
                frame_size,
                1
            )

            i += 1

        if spritesheet is not None:
            spritesheet.write(path)

    def make(self, path):
        egg_root = EggData()

        egg_comment = EggComment(
            '', 'KITSUNETSUKI Asset Tools by kitsune.ONE - '
            'https://github.com/kitsune-ONE-team/KITSUNETSUKI-Asset-Tools'
        )

        spritesheet = path.replace('.egg', '.{}'.format(self.type))
        self._make_spritesheet(spritesheet)

        name = os.path.basename(spritesheet).rpartition('.')[0]
        egg_texture = EggTexture(name, '{}{}'.format(self.prefix, spritesheet))
        egg_vertex_pool = EggVertexPool('vpool')

        egg_root.add_child(egg_comment)
        egg_root.add_child(egg_texture)
        egg_root.add_child(egg_vertex_pool)

        if self.empty:
            self._add_polygon()

        for animation, frames in self.animations.items():
            egg_group = self._add_geom('animation_{:04d}'.format(animation), frames, egg_texture, egg_vertex_pool)
            egg_group.add_matrix4(LMatrix4d.scale_mat(self.scale))
            egg_root.add_child(egg_group)

        # print(egg_root)
        egg_root.write_egg(path)
