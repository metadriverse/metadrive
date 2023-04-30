/**
 *
 * RenderPipeline
 *
 * Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#version 430

// Shader to generate the cloud noise textures

layout (local_size_x = 8, local_size_y = 8, local_size_z = 4) in;
uniform writeonly image3D DestTex;

const int tex_size = 32;

#pragma include "noise.inc.glsl"

void main() {
    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);
    vec3 flt_coord = coord / float(tex_size);

    float worley1 = sqrt(fbm_worley(flt_coord, 8, 0.6, 3, 0.76632)) * 1.0;
    float worley2 = sqrt(fbm_worley(flt_coord, 8, 0.4, 3, 0.26386)) * 1.0;
    float worley3 = sqrt(fbm_worley(flt_coord, 16, 0.4, 2, 0.64243)) * 1.0;

    // imageStore(DestTex, coord, vec4(worley1, worley2, worley3, 1));
    imageStore(DestTex, coord, vec4(worley1, worley2, worley3, 1));
}
