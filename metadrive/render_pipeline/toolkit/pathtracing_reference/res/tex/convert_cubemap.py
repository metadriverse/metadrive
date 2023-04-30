"""
Converts the cubemap to a spherical one
"""

from __future__ import division, print_function

from panda3d.core import *  # noqa
load_prc_file_data("", "textures-power-2 none")

import direct.directbase.DirectStart  # noqa

cubemap = loader.load_cube_map("../../../../data/default_cubemap/source/#.jpg")
w, h = 4096, 2048

cshader = Shader.make_compute(
    Shader.SL_GLSL, """
#version 430
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

uniform samplerCube SourceTex;
uniform writeonly image2D DestTex;
#define M_PI 3.1415926535897932384626433
#define TWO_PI 6.2831853071795864769252867

// Converts a normalized spherical coordinate (r = 1) to cartesian coordinates
vec3 spherical_to_vector(float theta, float phi) {
    float sin_theta = sin(theta);
    return normalize(vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos(theta)
    ));
}

// Fixes the cubemap direction
vec3 fix_cubemap_coord(vec3 coord) {
    return normalize(coord.xzy * vec3(1,-1,1));
}

void main() {
    ivec2 dimensions = imageSize(DestTex).xy;
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    float theta = (coord.x + 0.5) / float(dimensions.x) * TWO_PI;
    float phi = (dimensions.y - coord.y - 0.5) / float(dimensions.y) * M_PI;
    vec3 v = spherical_to_vector(phi, theta);
    v = fix_cubemap_coord(v);
    vec4 color = texture(SourceTex, v);
    imageStore(DestTex, coord, vec4(color));
}
"""
)

dest_tex = Texture("")
dest_tex.setup_2d_texture(w, h, Texture.T_float, Texture.F_rgba16)

print("Converting to spherical coordinates ..")
np = NodePath("np")
np.set_shader(cshader)
np.set_shader_input("SourceTex", cubemap)
np.set_shader_input("DestTex", dest_tex)
attr = np.get_attrib(ShaderAttrib)
base.graphicsEngine.dispatch_compute((w // 16, h // 16, 1), attr, base.win.gsg)

print("Extracting data ..")
base.graphicsEngine.extract_texture_data(dest_tex, base.win.gsg)

print("Writing texture ..")
dest_tex.write("envmap.png")
