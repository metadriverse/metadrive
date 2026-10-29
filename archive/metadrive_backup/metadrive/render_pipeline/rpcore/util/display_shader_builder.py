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

# pylint: disable=line-too-long

from direct.stdpy.file import isfile, open

from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.loader import RPLoader


class DisplayShaderBuilder(object):  # pylint: disable=too-few-public-methods
    """ Utility class to generate shaders on the fly to display texture previews
    and also buffers """
    @classmethod
    def build(cls, texture, view_width, view_height):
        """ Builds a shader to display <texture> in a view port with the size
        <view_width> * <view_height> """
        view_width, view_height = int(view_width), int(view_height)

        cache_key = "/$$rptemp/$$TEXDISPLAY-X{}-Y{}-Z{}-TT{}-CT{}-VW{}-VH{}.frag.glsl".format(
            texture.get_x_size(), texture.get_y_size(), texture.get_z_size(), texture.get_texture_type(),
            texture.get_component_type(), view_width, view_height
        )

        # Only regenerate the file when there is no cache entry for it
        if not isfile(cache_key) or True:
            fragment_shader = cls._build_fragment_shader(texture, view_width, view_height)

            with open(cache_key, "w") as handle:
                handle.write(fragment_shader)

        return RPLoader.load_shader("/$$rp/shader/default_gui_shader.vert.glsl", cache_key)

    @classmethod
    def _build_fragment_shader(cls, texture, view_width, view_height):
        """ Internal method to build a fragment shader displaying the texture """

        sampling_code, sampler_type = cls._generate_sampling_code(texture, view_width, view_height)

        # Build actual shader
        built = """
            #version 400
            #extension GL_ARB_shading_language_420pack : enable
            #pragma include "render_pipeline_base.inc.glsl"
            in vec2 texcoord;
            out vec3 result;
            uniform int mipmap;
            uniform int slice;
            uniform float brightness;
            uniform bool tonemap;
            uniform """ + sampler_type + """ p3d_Texture0;
            void main() {
                int view_width = """ + str(view_width) + """;
                int view_height = """ + str(
            view_height
        ) + """;
                ivec2 display_coord = ivec2(texcoord * vec2(view_width, view_height));
                int int_index = display_coord.x + display_coord.y * view_width;
                """ + sampling_code + """
                result *= brightness;
                if (tonemap)
                    result = result / (1 + result);
            }
        """

        # Strip trailing spaces
        built = '\n'.join([i.strip() for i in built.split("\n")])

        return built

    @classmethod
    def _generate_sampling_code(cls, texture, view_width, view_height):  # noqa # pylint: disable=unused-argument,too-many-branches
        """ Generates the GLSL code to sample a texture and also returns the
        GLSL sampler type """

        texture_type = texture.get_texture_type()
        comp_type = texture.get_component_type()

        # Useful snippets
        int_coord = "ivec2 int_coord = ivec2(texcoord * textureSize(p3d_Texture0, mipmap).xy);"
        slice_count = "int slice_count = textureSize(p3d_Texture0, 0).z;"

        float_types = [Image.T_float, Image.T_unsigned_byte]
        int_types = [Image.T_int, Image.T_unsigned_short, Image.T_unsigned_int_24_8]

        result = "result = vec3(1, 0, 1);", "sampler2D"

        if comp_type not in float_types + int_types:
            RPObject.global_warn("DisplayShaderBuilder", "Unkown texture component type:", comp_type)

        # 2D Textures
        if texture_type == Image.TT_2d_texture:

            if comp_type in float_types:
                result = "result = textureLod(p3d_Texture0, texcoord, mipmap).xyz;", "sampler2D"

            elif comp_type in int_types:
                result = int_coord + "result = texelFetch(p3d_Texture0, int_coord, mipmap).xyz / 10.0;", "isampler2D"  # noqa

        # Buffer Textures
        elif texture_type == Image.TT_buffer_texture:

            def range_check(code):
                return "if (int_index < textureSize(p3d_Texture0)) {" + code + "} else { result = vec3(1.0, 0.6, 0.2);};"  # noqa

            if comp_type in float_types:
                result = range_check("result = texelFetch(p3d_Texture0, int_index).xyz;"), "samplerBuffer"  # noqa

            elif comp_type in int_types:
                result = range_check(
                    "result = texelFetch(p3d_Texture0, int_index).xyz / 10.0;"
                ), "isamplerBuffer"  # noqa

        # 3D Textures
        elif texture_type == Image.TT_3d_texture:

            if comp_type in float_types:
                result = slice_count + "result = textureLod(p3d_Texture0, vec3(texcoord, (0.5 + slice) / slice_count), mipmap).xyz;", "sampler3D"  # noqa

            elif comp_type in int_types:
                result = int_coord + "result = texelFetch(p3d_Texture0, ivec3(int_coord, slice), mipmap).xyz / 10.0;", "isampler3D"  # noqa

        # 2D Texture Array
        elif texture_type == Image.TT_2d_texture_array:

            if comp_type in float_types:
                result = "result = textureLod(p3d_Texture0, vec3(texcoord, slice), mipmap).xyz;", "sampler2DArray"  # noqa

            elif comp_type in int_types:
                result = int_coord + "result = texelFetch(p3d_Texture0, ivec3(int_coord, slice), mipmap).xyz / 10.0;", "isampler2DArray"  # noqa

        # Cubemap
        elif texture_type == Image.TT_cube_map:

            code = "vec3 sample_dir = get_cubemap_coordinate(slice, texcoord*2-1);\n"
            code += "result = textureLod(p3d_Texture0, sample_dir, mipmap).xyz;"
            result = code, "samplerCube"

        # Cubemap array
        elif texture_type == Image.TT_cube_map_array:
            code = "vec3 sample_dir = get_cubemap_coordinate(slice % 6, texcoord*2-1);\n"
            code += "result = textureLod(p3d_Texture0, vec4(sample_dir, slice / 6), mipmap).xyz;"
            result = code, "samplerCubeArray"

        else:
            print("WARNING: Unhandled texture type", texture_type, "in display shader builder")

        return result
