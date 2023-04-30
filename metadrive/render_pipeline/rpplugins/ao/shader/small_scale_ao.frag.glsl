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

#pragma optionNV (unroll all)

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/noise.inc.glsl"
#pragma include "includes/poisson_disk.inc.glsl"

out float result;

uniform sampler2D DownscaledDepth;
uniform sampler2D AOResult;

// use the extended gbuffer api
#define USE_GBUFFER_EXTENSIONS 1
#pragma include "includes/gbuffer.inc.glsl"

float get_linear_depth_at(vec2 coord) {
    return textureLod(DownscaledDepth, coord, 0).x;
}

void main() {

    ivec2 coord = ivec2(gl_FragCoord.xy);

    // Provide some variables to the kernel
    vec2 screen_size = vec2(WINDOW_WIDTH, WINDOW_HEIGHT);
    vec2 pixel_size = vec2(1.0) / screen_size;
    vec2 texcoord = get_texcoord();
    float pixel_z = get_linear_depth_at(texcoord);

    // Merge with previous ao result
    float prev_result = textureLod(AOResult, texcoord, 0).x;

    // Fade out small scale ao
    if (pixel_z > 150.0) {
        result = prev_result;
        return;
    }

    vec3 noise_vec = rand_rgb(coord % 4 +
        0.01 * (MainSceneData.frame_index % (GET_SETTING(ao, clip_length))));

    vec3 pixel_view_normal = get_view_normal(texcoord);
    vec3 pixel_view_pos = get_view_pos_at(texcoord);

    float kernel_scale = min(5.0, 10.0 / pixel_z);
    const float sample_radius = 6.0;

    const int num_samples = 4;
    const float bias = 0.0005 + 0.01 / kernel_scale;
    float max_range = 0.2;

    float sample_offset = sample_radius * pixel_size.x * 30.0;
    float range_accum = 0.0;
    float accum = 0.0;

    sample_offset /= kernel_scale;

    for (int i = 0; i < num_samples; ++i) {
        vec3 offset = poisson_3D_16[4 * i];
        // offset = mix(offset, noise_vec, 0.5);
        offset = normalize(offset + noise_vec);

        // Flip offset in case it faces away from the normal
        offset = face_forward(offset, pixel_view_normal);

        // Compute offset position in view space
        vec3 offset_pos = pixel_view_pos + offset * sample_offset;

        // Project offset position to screen space
        vec3 projected = view_to_screen(offset_pos);

        // Fetch the expected depth
        float sample_depth = get_linear_depth_at(projected.xy);
        // float sample_depth = get_linear_z_from_z(get_depth_at(projected.xy));

        // Linearize both depths
        float linz_a = get_linear_z_from_z(projected.z);
        float linz_b = sample_depth;

        // Compare both depths by distance to find the AO factor
        float modifier = step(distance(linz_a, linz_b), max_range);
        range_accum += fma(modifier, 0.5, 0.5);
        modifier *= step(linz_b + bias, linz_a);
        accum += modifier;
    }

    // Normalize samples
    accum /= max(0.1, range_accum);
    accum = 1 - accum;
    accum = pow(accum, 5.0);
    prev_result *= accum;
    // prev_result = pow(accum, 9.0);
    result = prev_result;
}
