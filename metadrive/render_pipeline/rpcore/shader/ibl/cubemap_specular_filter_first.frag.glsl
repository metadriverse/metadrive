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

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/poisson_disk.inc.glsl"
#pragma include "includes/noise.inc.glsl"

#pragma optionNV (unroll all)

uniform samplerCube SourceTex;
uniform writeonly imageCube RESTRICT DestMipmap;
uniform int currentMip;

void main() {

    const int num_samples = 12;

    // Get cubemap coordinate
    int texsize = textureSize(SourceTex, currentMip).x;
    ivec2 coord = ivec2(gl_FragCoord.xy);
    ivec2 clamped_coord; int face;
    vec3 n = texcoord_to_cubemap(texsize, coord, clamped_coord, face);

    // Get tangent and binormal
    vec3 tangent, binormal;
    find_arbitrary_tangent(n, tangent, binormal);

    const float filter_radius = 0.05;
    vec3 accum = vec3(0.0);
    for (int i = 0; i < num_samples; ++i) {
        vec2 offset = poisson_2D_12[i];
        vec3 sample_vec = normalize(n +
            filter_radius * offset.x * tangent +
            filter_radius * offset.y * binormal +
            (rand(vec2(coord + 23 * i))-0.5) * 0.01);
        accum += textureLod(SourceTex, sample_vec, currentMip).xyz;
    }

    accum /= num_samples;
    imageStore(DestMipmap, ivec3(clamped_coord, face), vec4(accum, 1.0));
}
