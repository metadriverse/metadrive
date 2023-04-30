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

// Copies the reflection velocity vector

#define USE_GBUFFER_EXTENSIONS
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/color_spaces.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/transforms.inc.glsl"

uniform sampler2D TraceResult;
out vec2 velocity;

void main() {
    vec2 texcoord = get_texcoord();

    // Get bilateral start coordinates
    ivec2 coord = ivec2(gl_FragCoord.xy);
    ivec2 bil_start_coord = get_bilateral_coord(coord);
    float mid_depth = get_depth_at(texcoord);

    // Find the position of the closest fragment
    const int kernel_size = 0;
    vec2 best_result = vec2(0);
    float best_score = 0.0;

    for (int i = -kernel_size; i < 2 + kernel_size; ++i) {
        for (int j = -kernel_size; j < 2 + kernel_size; ++j) {

            ivec2 source_coord = bil_start_coord + ivec2(i, j);
            ivec2 screen_coord = source_coord * 2;

            vec2 trace_result = texelFetch(TraceResult, source_coord, 0).xy;
            float trace_depth = get_depth_at(screen_coord);
            float trace_score = max(0.0, 1.0 - 10.0 * abs(trace_depth - mid_depth));

            // Weight the sample, so that depths closer to the viewer have a greater weight.
            // Also scale by the fade factor to make sure a sample is present
            trace_score = (1 - trace_depth) * (trace_result.x > 1e-5 ? 1.0 : 0.0);

            if (trace_score > best_score) {
                best_result = trace_result;
                best_score = trace_score;
            }
        }
    }

    // In case no sample was found, fall back to the current pixels coordinate
    if (length_squared(best_result) < 1e-4) {
        best_result = texcoord;
    }

    best_result = truncate_coordinate(best_result); // xxx

    // Find depth of intersection
    float intersection_depth = get_depth_at(best_result);

    // Reconstruct last frame texcoord
    vec2 film_offset_bias = MainSceneData.current_film_offset * vec2(1.0, 1.0 / ASPECT_RATIO);
    vec3 pos = calculate_surface_pos(intersection_depth, texcoord - film_offset_bias);
    vec4 last_proj = MainSceneData.last_view_proj_mat_no_jitter * vec4(pos, 1);
    vec2 last_coord = fma(last_proj.xy / last_proj.w, vec2(0.5), vec2(0.5));

    // Compute velocity
    velocity = last_coord - texcoord;
}
