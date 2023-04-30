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

#define USE_TIME_OF_DAY 1
#define USE_GBUFFER_EXTENSIONS
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/shadows.inc.glsl"
#pragma include "includes/noise.inc.glsl"


#if GET_SETTING(pssm, use_pcf)
    uniform sampler2DShadow PSSMShadowAtlasPCF;
#else
    uniform sampler2D PSSMShadowAtlas;
#endif


#pragma include "/$$rp/rpplugins/pssm/shader/filter_pssm.inc.glsl"

uniform sampler2D ShadedScene;

uniform sampler2D PSSMShadowAtlas;
uniform mat4 pssm_mvps[GET_SETTING(pssm, split_count)];
uniform vec2 pssm_nearfar[GET_SETTING(pssm, split_count)];

out vec4 result;


void main() {

    #if DEBUG_MODE
        result = vec4(0);
        return;
    #endif

    vec2 texcoord = get_half_texcoord();

    vec3 start_pos = MainSceneData.camera_pos;
    vec3 end_pos = get_gbuffer_position(GBuffer, texcoord);

    // Looks weird
    // if (is_skybox(end_pos, MainSceneData.camera_pos))
    // {
    //     result = vec4(0);
    //     return;
    // }

    float max_distance = GET_SETTING(volumetrics, volumetric_max_distance);

    vec3 step_vector = end_pos - start_pos;
    if (length(step_vector) > max_distance) {
        step_vector = normalize(step_vector) * max_distance;
    }

    float total_distance = length(step_vector);

    end_pos = start_pos + step_vector;

    float jitter = rand(ivec2(gl_FragCoord.xy) % 2);
    jitter = 0;

    const int num_steps = GET_SETTING(volumetrics, volumetric_num_steps);
    vec3 step_offs = step_vector / num_steps;

    float volumetrics = 0.0;

    vec3 sun_vector = get_sun_vector();
    vec3 sun_color = get_sun_color() * get_sun_color_scale(sun_vector);

    const float distance_fade = GET_SETTING(volumetrics, volumetric_shadow_fadein_distance);

    int start_split = 0;
    const float fixed_bias = 0.0005;

    for (int i = 0; i < num_steps; ++i) {
        vec3 pos = start_pos + (i + jitter) * step_offs;
        vec3 proj = project(pssm_mvps[start_split], pos);

        // Check if out of split
        while (out_of_screen(proj.xy) && start_split < GET_SETTING(pssm, split_count) - 1) {
            ++start_split;
            proj = project(pssm_mvps[start_split], pos);
        }

        if (out_of_screen(proj.xy)) {
            // Out of pssm range
            break;
        }

        float sun_influence = get_shadow(get_split_coord(proj.xy, start_split), proj.z - get_fixed_bias(start_split));

        // Apply distance fade
        float d = float(i) / float(num_steps) * total_distance;
        volumetrics += 0.02 * sun_influence * smoothstep(0, 1, d / distance_fade) * (1 - volumetrics);
    }


    // volumetrics /= float(num_steps);
    volumetrics = saturate(volumetrics);
    volumetrics = pow(volumetrics, GET_SETTING(volumetrics, volumetric_shadow_pow));

    vec3 color = sun_color * 0.5 * GET_SETTING(volumetrics, volumetric_shadow_brightness);
    result = vec4(color, 0.01 * GET_SETTING(volumetrics, volumetric_shadow_intensity)) * volumetrics;
    result.w = saturate(result.w);



}
