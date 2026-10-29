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
#pragma include "includes/importance_sampling.inc.glsl"
#pragma include "includes/poisson_disk.inc.glsl"
#pragma include "includes/noise.inc.glsl"

#pragma optionNV (unroll all)

uniform samplerCube SourceCubemap;
uniform writeonly imageCube RESTRICT DestCubemap;
uniform int cubeSize;

void main() {
    const int sample_count = 64;

    // Get cubemap coordinate
    ivec2 coord = ivec2(gl_FragCoord.xy);
    ivec2 clamped_coord; int face;
    vec3 n = texcoord_to_cubemap(cubeSize, coord, clamped_coord, face);

    // Convert normal to spherical coordinates
    float theta, phi;
    vector_to_spherical(n, theta, phi);

    // Get tangent and binormal
    vec3 tangent, binormal;
    find_arbitrary_tangent(n, tangent, binormal);

    vec3 accum = vec3(0);
    for (int i = 0; i < sample_count; ++i)
    {
        vec2 xi = hammersley(i, sample_count);
        vec3 offset = importance_sample_lambert(xi);
        offset = normalize(tangent * offset.x + binormal * offset.y + n * offset.z);
        accum += textureLod(SourceCubemap, offset, 0).xyz;
    }

    accum /= sample_count;
    imageStore(DestCubemap, ivec3(clamped_coord, face), vec4(accum, 1));
}
