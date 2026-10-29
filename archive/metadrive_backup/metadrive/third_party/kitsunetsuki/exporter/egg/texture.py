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

from panda3d.core import PNMImage
from panda3d.egg import EggTexture


class TextureMixin(object):
    def get_num_channels(self, filepath):
        image = PNMImage()
        image.read(filepath)
        return image.get_num_channels()

    def get_images(self, material, shader):
        if self._render_type == 'rp':  # custom texture order for RP
            return (
                ((0, 'Diffuse'), self.get_diffuse(material, shader)),
                ((1, 'Normal'), self.get_normal_map(material, shader)),
                ((2, 'Specular'), self.get_specular_map(material, shader)),
                ((3, 'Roughness'), self.get_roughness_map(material, shader)),
            )
        else:
            return (
                ((0, 'Diffuse'), self.get_diffuse(material, shader)),
                ((1, 'Normal'), self.get_normal_map(material, shader)),
            )

    def make_texture(self, type_, image_texture):
        filepath = image_texture.image.filepath.lstrip('/')
        path = os.path.dirname(self._output)

        egg_texture = EggTexture(image_texture.image.name, os.path.join(path, filepath))

        if image_texture.extension == 'CLIP':
            egg_texture.set_wrap_mode(EggTexture.WM_clamp)
        elif image_texture.extension == 'REPEAT':
            egg_texture.set_wrap_mode(EggTexture.WM_repeat)

        # check if we have patched panda3d
        have_srgb = hasattr(EggTexture, 'F_srgb_alpha')
        num_channels = self.get_num_channels(os.path.join(path, filepath))

        if type_[1] == 'Diffuse':
            if have_srgb:
                if num_channels == 4:
                    egg_texture.set_format(EggTexture.F_srgb_alpha)
                elif num_channels == 3:
                    egg_texture.set_format(EggTexture.F_srgb)
            else:
                if num_channels == 4:
                    egg_texture.set_format(EggTexture.F_rgba)
                elif num_channels == 3:
                    egg_texture.set_format(EggTexture.F_rgb)

        else:
            if num_channels == 4:
                egg_texture.set_format(EggTexture.F_rgba)
            elif num_channels == 3:
                egg_texture.set_format(EggTexture.F_rgb)

        if type_[1] == 'Diffuse':
            egg_texture.set_env_type(EggTexture.ET_modulate)
        elif type_[1] == 'Normal':
            egg_texture.set_env_type(EggTexture.ET_normal)
        else:
            egg_texture.set_env_type(EggTexture.ET_decal)

        egg_texture.set_priority(type_[0])
        egg_texture.set_stage_name(type_[1])

        return None, egg_texture  # sampler, image

    def make_empty_texture(self, type_):
        filepath = 'textures/{}.png'.format(type_.lower())
        path = os.path.dirname(self._output)

        egg_texture = EggTexture(os.path.basename(filepath), os.path.join(path, filepath))

        egg_texture.set_wrap_mode(EggTexture.WM_clamp)

        # check if we have patched panda3d
        have_srgb = hasattr(EggTexture, 'F_srgb_alpha')

        if type_[1] == 'Diffuse':
            if have_srgb:
                egg_texture.set_format(EggTexture.F_srgb_alpha)
            else:
                egg_texture.set_format(EggTexture.F_rgba)
        else:
            egg_texture.set_format(EggTexture.F_rgba)

        if type_[1] == 'Diffuse':
            egg_texture.set_env_type(EggTexture.ET_modulate)
        elif type_[1] == 'Normal':
            egg_texture.set_env_type(EggTexture.ET_normal)
        else:
            egg_texture.set_env_type(EggTexture.ET_decal)

        egg_texture.set_priority(type_[0])
        egg_texture.set_stage_name(type_[1])

        return None, egg_texture  # sampler, image
