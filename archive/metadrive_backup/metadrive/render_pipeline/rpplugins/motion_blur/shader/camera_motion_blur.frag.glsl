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

#pragma optionNV (unroll all)

#if GET_SETTING(motion_blur, enable_object_blur)
uniform sampler2D SourceTex;
#define SOURCE SourceTex
#else
uniform sampler2D ShadedScene;
#define SOURCE ShadedScene
#endif


uniform sampler2D CombinedVelocity;
out vec3 result;

const int num_samples = GET_SETTING(motion_blur, num_camera_samples) * 2;
float max_velocity = 70.0 / WINDOW_WIDTH;
float min_velocity = 0.5 / WINDOW_WIDTH;

void main() {

    vec2 texcoord = get_texcoord();
    ivec2 coord = ivec2(gl_FragCoord.xy);

    #if DEBUG_MODE
        result = textureLod(SOURCE, texcoord, 0).xyz;
        return;
    #endif

    // Reconstruct last frame texcoord
    vec2 film_offset_bias = MainSceneData.current_film_offset * vec2(1.0, 1.0 / ASPECT_RATIO);
    vec3 pos = get_world_pos_at(texcoord - film_offset_bias);
    vec4 last_proj = MainSceneData.last_view_proj_mat_no_jitter * vec4(pos, 1);
    vec2 last_coord = fma(last_proj.xy / last_proj.w, vec2(0.5), vec2(0.5));

    // Compute velocity in screen space
    vec2 velocity = last_coord - texcoord;

    // Make sure that when we have low-fps, we reduce motion blur, and when we
    // have higher fps, we increase it - this way it perceptually always stays
    // the same (otherwise it feels really laggy at low FPS)
    const float target_fps = 60.0;
    velocity *= (1.0 / target_fps) / MainSceneData.frame_delta;
    velocity *= GET_SETTING(motion_blur, camera_blur_factor);

    float velocity_len = length(velocity);

    // We can abort early when no velocity is present
    if (velocity_len < min_velocity) {
        result = textureLod(SOURCE, texcoord, 0).xyz;
        return;
    }

    if (velocity_len > max_velocity) {
        float scale_factor = max_velocity / velocity_len;
        velocity *= scale_factor;
        velocity_len *= scale_factor;
    }

    // Weight the center sample by a small bit to make sure we always have a weight.
    // However, we don't weight it too much to make the blur not look weird.
    float weights = 1e-3;
    vec3 accum = textureLod(SOURCE, texcoord, 0).xyz * weights;
    float jitter = rand(texcoord);

    // Blur in both directions
    for (int i = -num_samples + 1; i < num_samples; ++i) {
        vec2 offs = (i + 0.5 * jitter) / float(num_samples) * velocity;

        // Prevent bleeding when rotating - that is, objects moving into different directions
        vec2 sample_velocity = textureLod(CombinedVelocity, texcoord + offs, 0).xy;
        float weight = saturate(dot(sample_velocity, velocity) * WINDOW_WIDTH * 3);
        accum += textureLod(SOURCE, texcoord + offs, 0).xyz * weight;
        weights += weight;
    }

    accum /= weights;
    result = max(vec3(0.0), accum);
}
