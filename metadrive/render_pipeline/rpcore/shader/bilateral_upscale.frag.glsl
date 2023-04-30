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

// Shader which upscales a half resolution image to full resolution,
// respecting the normals and depth

#pragma optionNV (unroll all)

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"

// x: Max depth difference, y: Max normal difference
uniform vec2 upscaleWeights;
uniform sampler2D SourceTex;
uniform GBufferData GBuffer;

out vec4 result;

void main() {
    // Get sample coordinates
    ivec2 coord = ivec2(gl_FragCoord.xy);
    ivec2 bil_start_coord = get_bilateral_coord(coord);

    // Get current pixel data
    float mid_depth = get_gbuffer_depth(GBuffer, coord);
    vec3 mid_nrm = get_gbuffer_normal(GBuffer, coord);

    const float max_depth_diff = upscaleWeights.x;
    const float max_nrm_diff = upscaleWeights.y;

    float weights = 0.0;
    vec4 accum = vec4(0);

    // Controls how many other pixels should be taken into account, besides
    // of the 4 direct neighbors.
    const int search_radius = 0;

    // Accumulate all samples
    for (int x = -search_radius; x < 2 + search_radius; ++x) {
        for (int y = -search_radius; y < 2 + search_radius; ++y) {

            ivec2 source_coord = bil_start_coord + ivec2(x, y);
            vec4 source_sample = texelFetch(SourceTex, source_coord, 0);

            // Check how much information those pixels share, and if it is
            // enough, use that sample
            float sample_depth = get_gbuffer_depth(GBuffer, source_coord * 2);
            vec3 sample_nrm = get_gbuffer_normal(GBuffer, source_coord * 2);
            float depth_diff = abs(sample_depth - mid_depth) / max_depth_diff;
            float nrm_diff = max(0, dot(sample_nrm, mid_nrm));
            float weight = 1.0 - saturate(depth_diff);
            weight *= pow(nrm_diff, 1.0 / max_nrm_diff);

            // Make sure we don't have a null-weight, but instead only a very
            // small weight, so that in case no pixel matches, we still have
            // data to work with.
            // weight = max(1e-5, weight);

            accum += source_sample * weight;
            weights += weight;
        }
    }

    if (weights < 1e-5) {
        // When no sample was valid, take the center sample - this is still
        // better than invalid pixels
        result = texelFetch(SourceTex, coord / 2, 0);
    } else {
        result = accum / weights;
    }
}
