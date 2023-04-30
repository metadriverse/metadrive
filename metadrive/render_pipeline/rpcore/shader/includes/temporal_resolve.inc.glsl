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

#define USE_GBUFFER_EXTENSIONS
#pragma include "includes/color_spaces.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"

// Maximum color distance
#ifndef RS_MAX_CLIP_DIST
    #define RS_MAX_CLIP_DIST 0.5
#endif

// Blending factor
#ifndef RS_DISTANCE_SCALE
    #define RS_DISTANCE_SCALE 10.0
#endif

// How long to keep good pixels
#ifndef RS_KEEP_GOOD_DURATION
    #define RS_KEEP_GOOD_DURATION 10.0
#endif

// How long to keep bad pixels
#ifndef RS_KEEP_BAD_DURATION
    #define RS_KEEP_BAD_DURATION 3.0
#endif

#ifndef RS_AABB_SIZE
    #define RS_AABB_SIZE 1.0
#endif

#ifndef RS_USE_SMOOTH_TECHNIQUE
    #define RS_USE_SMOOTH_TECHNIQUE 0
#endif

/*

Uses the reprojection suggested in:
http://www.crytek.com/download/Sousa_Graphics_Gems_CryENGINE3.pdf

Also based on:
https://github.com/playdeadgames/temporal

*/

#define USE_OPTIMIZATIONS 0
#define USE_CLIPPING 1

vec4 clip_aabb(vec3 aabb_min, vec3 aabb_max, vec4 p, vec4 q)
{
#if USE_OPTIMIZATIONS
    // note: only clips towards aabb center (but fast!)
    vec3 p_clip = 0.5 * (aabb_max + aabb_min);
    vec3 e_clip = 0.5 * (aabb_max - aabb_min);

    vec4 v_clip = q - vec4(p_clip, p.w);
    vec3 v_unit = v_clip.xyz / e_clip;
    vec3 a_unit = abs(v_unit);
    float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    if (ma_unit > 1.0)
        return vec4(p_clip, p.w) + v_clip / ma_unit;
    else
        return q; // point inside aabb
#else
    vec4 r = q - p;
    vec3 rmax = aabb_max - p.xyz;
    vec3 rmin = aabb_min - p.xyz;

    const float eps = 1e-8;

    if (r.x > rmax.x + eps)
        r *= (rmax.x / r.x);
    if (r.y > rmax.y + eps)
        r *= (rmax.y / r.y);
    if (r.z > rmax.z + eps)
        r *= (rmax.z / r.z);

    if (r.x < rmin.x - eps)
        r *= (rmin.x / r.x);
    if (r.y < rmin.y - eps)
        r *= (rmin.y / r.y);
    if (r.z < rmin.z - eps)
        r *= (rmin.z / r.z);

    return p + r;
#endif
}


#if RS_USE_POSITION_TECHNIQUE
    uniform sampler2D Previous_SceneDepth;
#endif


vec4 resolve_temporal(sampler2D current_tex, sampler2D last_tex, vec2 curr_coord, vec2 last_coord) {
    vec2 one_pixel = 1.0 / SCREEN_SIZE;
    vec4 curr_m = textureLod(current_tex, curr_coord, 0);

    // Out of screen, can early out
    if (out_of_screen(last_coord)) {
        return max(vec4(0.0), curr_m);
    }

    #if !RS_USE_SMOOTH_TECHNIQUE

        #if RS_USE_POSITION_TECHNIQUE

            float curr_z = get_depth_at(curr_coord);
            vec3 curr_pos = calculate_surface_pos(curr_z, curr_coord);

            float last_z = textureLod(Previous_SceneDepth, last_coord, 0).x;
            vec3 last_pos = calculate_surface_pos(
                last_z, last_coord, MainSceneData.last_inv_view_proj_mat_no_jitter);

            // Weight by distance
            float max_distance = RS_DISTANCE_SCALE;
            max_distance *= distance(curr_pos, MainSceneData.camera_pos) / 10.0;

            float weight = 1.0 - saturate(distance(curr_pos, last_pos) / max_distance);
            weight *= 1 - 1.0 / RS_KEEP_GOOD_DURATION;

            vec4 last_m = textureLod(last_tex, last_coord, 0);
            return mix(curr_m, last_m, weight);

        #else

            const float subpixel_threshold = 0.5;
            const float gather_base = 0.5;
            const float gather_subpixel_motion = 0.1666;

            float curr_depth = get_depth_at(curr_coord);
            float vs_dist = get_linear_z_from_z(curr_depth);

            vec2 velocity = last_coord - curr_coord;

            float texel_vel_mag = length(velocity) * vs_dist;
            float subpixel_motion = saturate(subpixel_threshold / (1e-8 + texel_vel_mag));
            float min_max_support = gather_base + gather_subpixel_motion * subpixel_motion;

            vec2 ss_offset01 = min_max_support * vec2(1, 0) / SCREEN_SIZE;
            vec2 ss_offset11 = min_max_support * vec2(0, 1) / SCREEN_SIZE;

            vec4 c00 = textureLod(current_tex, curr_coord - ss_offset11, 0);
            vec4 c10 = textureLod(current_tex, curr_coord - ss_offset01, 0);
            vec4 c01 = textureLod(current_tex, curr_coord + ss_offset01, 0);
            vec4 c11 = textureLod(current_tex, curr_coord + ss_offset11, 0);

            vec4 cmin = min4(c00, c10, c01, c11);
            vec4 cmax = max4(c00, c10, c01, c11);

            vec4 cavg = (c00 + c10 + c01 + c11) / 4.0;

            vec4 last_m = textureLod(last_tex, curr_coord + velocity, 0);

            #if USE_CLIPPING
                last_m = clip_aabb(cmin.xyz, cmax.xyz, clamp(cavg, cmin, cmax), last_m);
            #else
                last_m = clamp(last_m, cmin, cmax);
            #endif

            float lum0 = get_luminance(curr_m.rgb);
            float lum1 = get_luminance(last_m.rgb);

            const float feedback_min = 1 - 1.0 / RS_KEEP_BAD_DURATION;
            const float feedback_max = 1 - 1.0 / RS_KEEP_GOOD_DURATION;

            float unbiased_diff = abs(lum0 - lum1) / max(lum0, max(lum1, 0.2));
            float unbiased_weight = 1.0 - unbiased_diff;
            float unbiased_weight_sqr = unbiased_weight * unbiased_weight;
            float feedback = mix(feedback_min, feedback_max, unbiased_weight_sqr);

            // output
            return mix(curr_m, last_m, feedback);

        #endif

    #else

        // Bounding box size
        const float bbs = RS_AABB_SIZE;

        // Get current frame neighbor texels
        vec4 curr_tl = textureLod(current_tex, curr_coord + vec2(-bbs, -bbs) * one_pixel, 0);
        vec4 curr_tr = textureLod(current_tex, curr_coord + vec2(bbs, -bbs) * one_pixel, 0);
        vec4 curr_bl = textureLod(current_tex, curr_coord + vec2(-bbs, bbs) * one_pixel, 0);
        vec4 curr_br = textureLod(current_tex, curr_coord + vec2(bbs, bbs) * one_pixel, 0);

        // Get current frame neighbor AABB
        vec4 curr_min = min5(curr_m, curr_tl, curr_tr, curr_bl, curr_br);
        vec4 curr_max = max5(curr_m, curr_tl, curr_tr, curr_bl, curr_br);

        // Get last frame texels
        float blend_weight = 1.0;
        vec4 last_m = textureLod(last_tex, last_coord, 0);
        vec4 last_tl = textureLod(last_tex, last_coord + vec2(-bbs, -bbs) * one_pixel, 0);
        vec4 last_tr = textureLod(last_tex, last_coord + vec2(bbs, -bbs) * one_pixel, 0);
        vec4 last_bl = textureLod(last_tex, last_coord + vec2(-bbs, bbs) * one_pixel, 0);
        vec4 last_br = textureLod(last_tex, last_coord + vec2(bbs, bbs) * one_pixel, 0);

        float neighbor_diff = length(clamp(last_tl.xyz, curr_min.xyz, curr_max.xyz) - last_tl.xyz)
                            + length(clamp(last_tr.xyz, curr_min.xyz, curr_max.xyz) - last_tr.xyz)
                            + length(clamp(last_bl.xyz, curr_min.xyz, curr_max.xyz) - last_bl.xyz)
                            + length(clamp(last_br.xyz, curr_min.xyz, curr_max.xyz) - last_br.xyz);

        const float tolerance = 0.0;

        float max_difference = clamp(max(
            get_luminance(last_m.xyz),
            get_luminance(curr_m.xyz)), 0.0001, 15.0) * RS_MAX_CLIP_DIST;

        if (neighbor_diff >= max_difference)
            blend_weight = 0.0;

        float blend_amount = saturate(distance(last_m.xyz, curr_m.xyz) *
            RS_DISTANCE_SCALE);

        // Merge the sample with the current color, in case we can't pick it
        last_m = mix(curr_m, last_m, blend_weight);

        float weight = saturate(1.0 /
            mix(RS_KEEP_GOOD_DURATION, RS_KEEP_BAD_DURATION, blend_amount));

        return max(vec4(0.0), mix(last_m, curr_m, weight));
    #endif

}
