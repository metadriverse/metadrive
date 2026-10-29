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

#define USE_GBUFFER_EXTENSIONS
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/noise.inc.glsl"

#pragma include "motion_blur.inc.glsl"

uniform sampler2D ShadedScene;
uniform sampler2D PackedSceneData;
uniform sampler2D NeighborMinMax;

out vec3 result;

const int num_samples = GET_SETTING(motion_blur, num_samples);


void main() {

    vec2 texcoord = get_texcoord();
    ivec2 coord = ivec2(gl_FragCoord.xy);
    ivec2 tile = coord / tile_size;
    vec3 center_color = texelFetch(ShadedScene, coord, 0).xyz;

    vec2 tile_velocity = texelFetch(NeighborMinMax, tile, 0).xy; // max_v
    const float weight_step = 1.0 / float(num_samples - 1);

    // Early out
    if (length(tile_velocity) < 0.5 / WINDOW_WIDTH) {
        result = max(vec3(0.0), center_color);
        return;
    }

    // Get current pixel offset
    vec2 vx = texelFetch(PackedSceneData, coord, 0).xy;
    vec2 blur_step = tile_velocity * weight_step * 2.0;
    vec2 start_tc = texcoord - tile_velocity;

    float min_len_xy = -length(tile_velocity.xy);
    float len_xy_step = (-min_len_xy) * weight_step * 2.0;

    float initial_weight = saturate(float(num_samples / 40.0));
    initial_weight *= 1.0 / (max(1.0, vx.x));

    float jitter = rand(vec2(coord));

    float weight_accum = initial_weight;
    vec3 accum = center_color * initial_weight;

    for (int i = 0; i < num_samples; ++i) {

        vec2 tc = start_tc + blur_step * (i + jitter);
        vec2 vy = textureLod(PackedSceneData, tc, 0).xy;
        float len_xy = abs(min_len_xy + len_xy_step * (i + jitter));

        vec2 cmp_softz = soft_depth_cmp(vec2(vx.y, vy.y), vec2(vy.y, vx.y));
        vec4 cmp_batch = batch_cmp(len_xy, max(vec2(1e-6), vec2(vy.x, vx.x)));
        float w = dot(cmp_softz, cmp_batch.xy) + (cmp_batch.z * cmp_batch.w) * 2.0;

        accum += saturate(textureLod(ShadedScene, tc, 0).xyz) * w;
        weight_accum += w;

    }

    result = max(vec3(0.0), accum / max(0.001, weight_accum));
}
