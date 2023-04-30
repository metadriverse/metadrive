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
#pragma include "includes/brdf.inc.glsl"
#pragma include "includes/noise.inc.glsl"

#pragma optionNV (unroll all)

uniform int currentMip;
uniform samplerCube SourceTex;
layout(r11f_g11f_b10f) uniform imageCube RESTRICT DestMipmap;

void main() {
    const int num_samples = 64;

    // Get cubemap coordinate
    int texsize = imageSize(DestMipmap).x;
    ivec2 coord = ivec2(gl_FragCoord.xy);
    ivec2 clamped_coord; int face;
    vec3 n = texcoord_to_cubemap(texsize, coord, clamped_coord, face);

    // Determine target roughenss
    float sample_roughness = 1e-10 + (currentMip + 1) / 7.0 - 0.04;

    vec3 tangent, binormal;
    find_arbitrary_tangent(n, tangent, binormal);

    vec3 accum = vec3(0);
    float accum_weights = 0.0;

    // Importance sampling
    for (int i = 0; i < num_samples; ++i) {
        vec2 Xi = hammersley(i, num_samples);
        vec3 h = importance_sample_ggx(Xi, sample_roughness);
        h = normalize(h.x * tangent + h.y * binormal + h.z * n);
        accum += textureLod(SourceTex, h, 1).xyz;
    }

    // Energy conservation
    accum /= num_samples;
    imageStore(DestMipmap, ivec3(clamped_coord, face), vec4(accum, 1.0));
}
