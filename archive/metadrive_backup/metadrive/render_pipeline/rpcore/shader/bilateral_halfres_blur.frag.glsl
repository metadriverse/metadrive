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

// This shader provides a bilateral blur at half-resolution

#pragma optionNV (unroll all)

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/gaussian_weights.inc.glsl"

uniform ivec2 blur_direction;
uniform sampler2D SourceTex;
uniform GBufferData GBuffer;

out vec4 result;

float get_lin_z(vec2 ccoord) {
    return get_linear_z_from_z(get_gbuffer_depth(GBuffer, ccoord));
}

void main() {
    vec2 texcoord = get_half_native_texcoord();
    vec2 pixel_size = 2.0 / SCREEN_SIZE;

    // Store accumulated color
    vec4 accum = vec4(0);
    float accum_w = 0.0;

    // Amount of samples, don't forget to change the weights array in case you change this.
    const int blur_size = 4;

    // Get the mid pixel normal and depth
    vec3 pixel_nrm = get_gbuffer_normal(GBuffer, texcoord);
    float pixel_depth = get_lin_z(texcoord);

    // Blur to the right
    for (int i = -blur_size + 1; i < blur_size; ++i) {
        vec2 offcoord = texcoord + pixel_size * i * blur_direction;
        vec4 sampled = textureLod(SourceTex, offcoord, 0);
        vec3 nrm = get_gbuffer_normal(GBuffer, offcoord);
        float depth = get_lin_z(offcoord);

        float weight = gaussian_weights_4[abs(i)];
        weight *= 1.0 - saturate(GET_SETTING(ao, blur_normal_factor) *
            distance(nrm, pixel_nrm) / 0.01);
        weight *= 1.0 - saturate(GET_SETTING(ao, blur_depth_factor) *
            abs(depth - pixel_depth) / 0.001);

        accum += sampled * weight;
        accum_w += weight;
    }

    accum /= max(0.04, accum_w);
    result = accum;
}
