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

#pragma include "scattering_common.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform sampler3D deltaSSampler;

#if SCATTERING_USE_32_BIT
layout(rgba32f) uniform image3D RESTRICT dest;
#else
layout(rgba16f) uniform image3D RESTRICT dest;
#endif

void main() {

    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);

    float r;
    vec4 dhdH;
    get_r_dhdh(coord.z, r, dhdH);

    float mu, muS, nu;
    getMuMuSNu(r, dhdH, mu, muS, nu);

    vec4 orig_val = imageLoad(dest, coord);
    vec4 new_val = texelFetch(deltaSSampler, coord, 0) / phaseFunctionR(nu);
    imageStore(dest, coord, vec4(orig_val + new_val));

}
