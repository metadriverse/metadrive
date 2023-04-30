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

from panda3d.core import SamplerState, Vec4

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.image import Image


class CubemapFilter(RPObject):
    """ Util class for filtering cubemaps, provides funcionality to generate
    a specular and diffuse IBL cubemap. """

    # Fixed size for the diffuse cubemap, since it does not contain much detail
    DIFFUSE_CUBEMAP_SIZE = 10
    PREFILTER_CUBEMAP_SIZE = 32

    def __init__(self, parent_stage, name="Cubemap", size=128):
        """ Inits the filter from a given stage """
        RPObject.__init__(self)
        self._stage = parent_stage
        self._name = name
        self._size = size
        self._make_maps()

    @property
    def specular_cubemap(self):
        """ Returns the generated specular cubemap. The specular cubemap is
        mipmapped and provides the specular IBL components of the input cubemap. """
        return self._specular_map

    @property
    def diffuse_cubemap(self):
        """ Returns the generated diffuse cubemap. The diffuse cubemap has no
        mipmaps and contains the filtered diffuse component of the input cubemap. """
        return self._diffuse_map

    @property
    def target_cubemap(self):
        """ Returns the target where the caller should write the initial cubemap
        data to """
        return self._specular_map

    @property
    def size(self):
        """ Returns the size of the created cubemap, previously passed to the
        constructor of the filter """
        return self._size

    def create(self):
        """ Creates the filter. The input cubemap should be mipmapped, and will
        get reused for the specular cubemap. """
        self._make_specular_targets()
        self._make_diffuse_target()

    def _make_maps(self):
        """ Internal method to create the cubemap storage """

        # Create the cubemaps for the diffuse and specular components
        self._prefilter_map = Image.create_cube(
            self._name + "IBLPrefDiff", CubemapFilter.PREFILTER_CUBEMAP_SIZE, "R11G11B10"
        )
        self._diffuse_map = Image.create_cube(self._name + "IBLDiff", CubemapFilter.DIFFUSE_CUBEMAP_SIZE, "R11G11B10")
        self._spec_pref_map = Image.create_cube(self._name + "IBLPrefSpec", self._size, "R11G11B10")
        self._specular_map = Image.create_cube(self._name + "IBLSpec", self._size, "R11G11B10")

        # Set the correct filtering modes
        for tex in [self._diffuse_map, self._specular_map, self._prefilter_map, self._spec_pref_map]:
            tex.set_minfilter(SamplerState.FT_linear)
            tex.set_magfilter(SamplerState.FT_linear)
            tex.set_clear_color(Vec4(0))
            tex.clear_image()

        # Use mipmaps for the specular cubemap
        self._spec_pref_map.set_minfilter(SamplerState.FT_linear_mipmap_linear)
        self._specular_map.set_minfilter(SamplerState.FT_linear_mipmap_linear)

    def _make_specular_targets(self):
        """ Internal method to create the specular mip chain """
        self._targets_spec = []
        self._targets_spec_filter = []
        mip, mipsize = 0, self._size
        while mipsize >= 2:

            # Assuming an initial size of a power of 2, its safe to do this
            mipsize = mipsize // 2

            # Create the target which downsamples the mipmap
            target = self._stage.create_target("CF:SpecIBL-PreFilter-" + str(mipsize))
            target.size = mipsize * 6, mipsize
            target.prepare_buffer()

            if mip == 0:
                target.set_shader_input("SourceTex", self._specular_map)
            else:
                target.set_shader_input("SourceTex", self._spec_pref_map)
            target.set_shader_input("DestMipmap", self._spec_pref_map, False, True, -1, mip + 1, 0)
            target.set_shader_input("currentMip", mip)

            mip += 1

            # Create the target which filters the mipmap and removes the noise
            target_filter = self._stage.create_target("CF:SpecIBL-PostFilter-" + str(mipsize))
            target_filter.size = mipsize * 6, mipsize
            target_filter.prepare_buffer()
            target_filter.set_shader_inputs(currentMip=mip, SourceTex=self._spec_pref_map)
            target_filter.set_shader_input("DestMipmap", self._specular_map, False, True, -1, mip, 0)

            self._targets_spec.append(target)
            self._targets_spec_filter.append(target_filter)

    def _make_diffuse_target(self):
        """ Internal method to create the diffuse cubemap """

        # Create the target which integrates the lambert brdf
        self._diffuse_target = self._stage.create_target("CF:DiffuseIBL")
        self._diffuse_target.size = (CubemapFilter.PREFILTER_CUBEMAP_SIZE * 6, CubemapFilter.PREFILTER_CUBEMAP_SIZE)
        self._diffuse_target.prepare_buffer()

        self._diffuse_target.set_shader_inputs(
            SourceCubemap=self._specular_map,
            DestCubemap=self._prefilter_map,
            cubeSize=CubemapFilter.PREFILTER_CUBEMAP_SIZE
        )

        # Create the target which removes the noise from the previous target,
        # which is introduced with importance sampling
        self._diff_filter_target = self._stage.create_target("CF:DiffPrefIBL")
        self._diff_filter_target.size = (CubemapFilter.DIFFUSE_CUBEMAP_SIZE * 6, CubemapFilter.DIFFUSE_CUBEMAP_SIZE)
        self._diff_filter_target.prepare_buffer()

        self._diff_filter_target.set_shader_inputs(
            SourceCubemap=self._prefilter_map,
            DestCubemap=self._diffuse_map,
            cubeSize=CubemapFilter.DIFFUSE_CUBEMAP_SIZE
        )

    def reload_shaders(self):
        """ Sets all required shaders on the filter. """

        # Set diffuse filter shaders
        self._diffuse_target.shader = self._stage.load_shader("ibl/cubemap_diffuse.frag.glsl")
        self._diff_filter_target.shader = self._stage.load_shader("ibl/cubemap_diffuse_filter.frag.glsl")

        # Set specular prefilter shaders
        mip_shader = self._stage.load_shader("ibl/cubemap_specular_prefilter.frag.glsl")
        for target in self._targets_spec:
            target.shader = mip_shader

        # Special shader for the first prefilter target
        self._targets_spec[0].shader = self._stage.load_shader("ibl/cubemap_specular_prefilter_first.frag.glsl")

        # Set specular filter sampling shaders
        mip_filter_shader = self._stage.load_shader("ibl/cubemap_specular_filter.frag.glsl")
        for target in self._targets_spec_filter:
            target.shader = mip_filter_shader

        # Special shader for the first filter target
        self._targets_spec_filter[0].shader = self._stage.load_shader("ibl/cubemap_specular_filter_first.frag.glsl")
