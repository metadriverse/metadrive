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

#define USE_TIME_OF_DAY 1
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/sampling_sequences.inc.glsl"
#pragma include "includes/shadows.inc.glsl"
#pragma include "includes/noise.inc.glsl"

out float result;

uniform GBufferData GBuffer;

#if GET_SETTING(pssm, use_pcf)
    uniform sampler2DShadow PSSMShadowAtlasPCF;
#endif

uniform sampler2D PSSMShadowAtlas;

uniform mat4 pssm_mvps[GET_SETTING(pssm, split_count)];
uniform vec2 pssm_nearfar[GET_SETTING(pssm, split_count)];

#if GET_SETTING(pssm, use_distant_shadows)
    uniform mat4 PSSMDistSunShadowMapMVP;
    uniform sampler2D PSSMDistSunShadowMap;
#endif

#pragma include "filter_pssm.inc.glsl"


void main() {
    vec3 sun_vector = get_sun_vector();

    // Get current scene color
    vec2 texcoord = get_texcoord();
    ivec2 coord = ivec2(gl_FragCoord.xy);

    // Get the material data
    Material m = unpack_material(GBuffer);

    // Early out, different optimizations
    bool early_out = is_skybox(m) || sun_vector.z < SUN_VECTOR_HORIZON;
    early_out = early_out ||
        (m.shading_model != SHADING_MODEL_FOLIAGE && dot(m.normal, sun_vector) <= 1e-7);

    if (early_out) {
        result = 0.0;
        return;
    }

    // Variables to accumulate the shadows
    float shadow_factor = 1.0;

    // Compute distant shadows
    #if GET_SETTING(pssm, use_distant_shadows)
    // #if 0
    {
        // Compute the biased position based on the normal and slope scaled
        // bias.
        const float slope_bias = GET_SETTING(pssm, slope_bias) * 0.1 * 2;
        const float normal_bias = GET_SETTING(pssm, normal_bias) * 0.1;
        const float fixed_bias = 0.0005;
        vec3 biased_pos = get_biased_position(m.position,
            slope_bias, normal_bias, m.normal, sun_vector);
        vec3 proj = project(PSSMDistSunShadowMapMVP, biased_pos);
        proj.z -= fixed_bias;

        if (!out_of_unit_box(proj)) {
            const float esm_factor = 5.0;
            float depth_sample = textureLod(PSSMDistSunShadowMap, proj.xy, 0).x;
            shadow_factor = saturate(exp(-esm_factor * proj.z) * depth_sample);
            shadow_factor = pow(shadow_factor, 1e2);
        }
    }
    #endif

    // Find lowest split in range
    const int split_count = GET_SETTING(pssm, split_count);
    int split = 99;
    float border_bias = 0.5 - (0.5 / (1.0 + GET_SETTING(pssm, border_bias)));

    // Find the first matching split
    for (int i = 0; i < split_count; ++i) {
        vec3 coord = project(pssm_mvps[i], m.position);
        if (coord.x >= border_bias && coord.x <= 1-border_bias &&
            coord.y >= border_bias && coord.y <= 1-border_bias &&
            coord.z >= 0.0 && coord.z <= 1.0) {
            split = i;
            break;
        }
    }

    // Compute the shadowing factor
    if (split < GET_SETTING(pssm, split_count)) {

        // Get the MVP for the current split
        mat4 mvp = pssm_mvps[split];


        float rotation = interleaved_gradient_noise(
            gl_FragCoord.xy + (MainSceneData.frame_index % 4) / 3.0 );

        // XXX: Pretty noisy
        // mat2 rotation_mat = make_rotation_mat(rotation);
        mat2 rotation_mat = mat2(1, 0, 0, 1);

        const float filter_radius = GET_SETTING(pssm, filter_radius) /
            GET_SETTING(pssm, resolution);

        vec3 biased_pos = get_pssm_split_biased_position(m.position, m.normal, sun_vector, split);

        // Project the current pixel to the view of the light
        vec3 projected = project(mvp, biased_pos);
        vec2 projected_coord = get_split_coord(projected.xy, split);

        // Compute the fixed bias
        float ref_depth = projected.z - get_fixed_bias(split);

        // Find filter size
        vec2 filter_size = find_filter_size(mvp, sun_vector, filter_radius);

        // Increase filter size in the distance, to get better cache usage
        filter_size *= (1.0 + 10.5 * distance(m.position, MainSceneData.camera_pos) / 100.0);
        // vec2 filter_size = vec2(0.5 * filter_radius) * (1 / (1 + 0.7 * split));

        #if GET_SETTING(pssm, use_pcss)

            {
                /*

                    PCSS Kernel

                    Scan the region of the pixel for blockers, penumbra size is
                    amount of blockers compared to non-blockers.

                */

                float num_blockers = 0.0;
                float sum_blockers = 0.0;

                START_ITERATE_SEQUENCE(pssm, pcss_sequence, vec2 offset)

                    offset = rotation_mat * offset;

                    // Find depth at sample location
                    float sampled_depth = textureLod(PSSMShadowAtlas,
                        projected_coord + offset * filter_size * 4.0, 0).x;

                    // Compare the depth with the pixel depth, in case its smaller,
                    // we found a blocker
                    float factor = step(sampled_depth, ref_depth);
                    num_blockers += factor;
                    sum_blockers += sampled_depth * factor;
                
                END_ITERATE_SEQUENCE();

                // Examine ratio between blockers and non-blockers
                float avg_blocker_depth = sum_blockers / num_blockers;

                // Penumbra size also takes average blocker depth into account
                float penumbra_size = max(GET_SETTING(pssm, pcss_min_penumbra_size) * 0.03, ref_depth - avg_blocker_depth) /
                    ref_depth * GET_SETTING(pssm, pcss_penumbra_size);

                // Apply penumbra size
                filter_size *= penumbra_size;
            }

        #endif

        float local_shadow_factor = 0.0;


        // Do the actual shadow map filtering
        START_ITERATE_SEQUENCE(pssm, filter_sequence, vec2 offset)

            local_shadow_factor += get_shadow(
                projected_coord + (rotation_mat * offset) * filter_size, ref_depth);

        END_ITERATE_SEQUENCE();
        NORMALIZE_SEQUENCE(pssm, filter_sequence, local_shadow_factor);


        if (split >= GET_SETTING(pssm, split_count) - 1) {
            // Smoothly fade in distant shadows
            const float fade_size = 0.09;
            const float depth_fade = 0.01;
            float cascade_factor = 1;
            cascade_factor *= saturate(projected.x / fade_size);
            cascade_factor *= saturate(projected.y / fade_size);
            cascade_factor *= saturate((1 - projected.x) / fade_size);
            cascade_factor *= saturate((1 - projected.y) / fade_size);
            cascade_factor *= saturate(projected.z / depth_fade);
            cascade_factor *= saturate((1 - projected.z) / depth_fade);
            shadow_factor = mix(shadow_factor, local_shadow_factor, cascade_factor);
        } else {
            shadow_factor = local_shadow_factor;
        }

        // OPTIONAL: Transmittance - however this looks a bit buggy right now -
        // might look better on surfaces like ice and so on

        // #if HAVE_PLUGIN(skin_shading)
        //     // Get the current split near and far planes
        //     vec2 split_near_far = pssm_nearfar[split];

        //     // Bias to move the position "into" the object, prevents artifacts
        //     float skin_border_factor = 0.025;

        //     // Project the biased position to light space
        //     vec3 projected_skin = project(mvp, m.position - m.normal * skin_border_factor);
        //     vec2 projected_skin_coord = get_split_coord(projected_skin.xy, split);
        //     float skin_ref_depth = projected_skin.z;

        //     // Get the shadow sample
        //     float shadow_sample = textureLod(PSSMShadowAtlas, projected_skin_coord, 0).x;

        //     // Reconstruct intersection position
        //     // TODO: This is totally slow! Pass it as input
        //     mat4 inverse_mvp = inverse(mvp);
        //     vec3 intersection_pos = calculate_surface_pos_ortho(shadow_sample, projected_skin.xy, split_near_far.x, split_near_far.y, inverse_mvp);

        //     // Get the distance the light traveled through the medium
        //     float distance_through_medium = distance(m.position, intersection_pos.xyz);

        //     // TODO: Maybe we can remove this branch
        //     if (skin_ref_depth < shadow_sample) distance_through_medium = 0.0;

        //     // Fetch the skin transmittance
        //     // transmittance = skin_transmittance(distance_through_medium);
        //     transmittance = vec3(1);

        // #endif

    }

    // Raymarch clouds
    // CloudVoxels
    #if HAVE_PLUGIN(clouds)
        {
        // vec3 start_coord = vec3(m.position.xy / 800.0, 0);
        // vec3 end_coord = start_coord + vec3(0, 0, 1);
        // const int num_steps = 32;
        // vec3 step_dir = (end_coord - start_coord) / num_steps;
        // float cloud_factor = 0.0;
        // for (int i = 0; i < num_steps; ++i) {
        //     float cloud_sample = textureLod(CloudVoxels, start_coord, 0).w;
        //     cloud_factor += cloud_sample;
        //     start_coord += step_dir;
        // }
        // cloud_factor /= num_steps;
        // cloud_factor *= 10.0;
        // shadow_factor *= saturate(1.0 - cloud_factor);
        }
    #endif

    #if MODE_ACTIVE(PSSM_SPLITS)
        float factor = float(split) / GET_SETTING(pssm, split_count);
        shadow_factor = saturate(shadow_factor+0.5) * factor;
    #endif

    result = shadow_factor;
}
