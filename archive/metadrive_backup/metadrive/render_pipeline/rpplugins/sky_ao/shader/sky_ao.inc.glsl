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
#pragma once

#pragma include "includes/sampling_sequences.inc.glsl"
#pragma include "includes/noise.inc.glsl"

uniform sampler2D SkyAOHeight;
uniform vec3 SkyAOCapturePosition;

float compute_sky_ao(vec3 ws_position, vec3 ws_normal) {

    // Constants
    const int ao_map_resolution = GET_SETTING(sky_ao, resolution);
    const float radius = GET_SETTING(sky_ao, sample_radius);
    const float ao_multiplier = GET_SETTING(sky_ao, ao_multiplier);
    const float ao_bias = GET_SETTING(sky_ao, ao_bias);
    const float fade_scale = GET_SETTING(sky_ao, blend_factor);
    const float ao_map_scale = GET_SETTING(sky_ao, max_radius); // world space
    const float film_size = ao_map_scale * 0.5;
    

    vec2 local_coord = (ws_position.xy - SkyAOCapturePosition.xy) / film_size * 0.5 + 0.5;

    if (out_of_screen(local_coord)) {
        return 1.0;
    }

    float blend = 1.0;
    blend *= saturate(min(local_coord.x, local_coord.y) / fade_scale);
    blend *= saturate(min(1 - local_coord.x, 1 - local_coord.y) / fade_scale);


    // Factor to convert from tex-space to world-space
    const float tc_to_ws = ao_map_scale;


    float accum = 0.0;
    vec2 jitter = rand_rgb(ivec2(gl_FragCoord.xy) % 2).xy;

    // Capture samples

    // vec3 avg_unoccluded_normal = vec3(0);

    START_ITERATE_SEQUENCE(sky_ao, sample_sequence, vec2 offset)

        offset = (offset * 0.9 + 0.1 * jitter) / ao_map_resolution * radius;
        vec2 offcoord = local_coord + offset;
        
        float sample_z = textureLod(SkyAOHeight, offcoord, 0).x;
        

        accum += saturate(1.0 * (sample_z - ws_position.z));
        // if (sample_z > ws_position.z)
        //     accum += 1.0;
        // Project sample in normal direction
        // float dist = length(offset) * tc_to_ws;
        // float proj_z = ws_position.z + dist * ws_normal.z;


        // accum += saturate((sample_z - proj_z) * 2.0);
        // if (proj_z >= sample_z) {
            // accum += 1.0;
        // }

    END_ITERATE_SEQUENCE()

    NORMALIZE_SEQUENCE(sky_ao, sample_sequence, accum);

    accum = saturate(1 - accum + ao_bias);
    accum = pow(accum, 0.25 * ao_multiplier);
    accum = mix(1, accum, blend);

    return accum;
}
