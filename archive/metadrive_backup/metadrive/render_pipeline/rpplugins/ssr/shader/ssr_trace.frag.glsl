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
#pragma include "includes/brdf.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/importance_sampling.inc.glsl"

uniform sampler2D DownscaledDepth;

out vec2 result;

#define USE_LINEAR_DEPTH 0
#define NUM_RAYDIR_RETRIES 3

const int num_steps = GET_SETTING(ssr, trace_steps);
const float hit_tolerance_ws = 0.00001 * GET_SETTING(ssr, hit_tolerance);
const float hit_tolerance_backface = 0.00001;

bool point_between_planes(float z, float z_a, float z_b, float trace_length, out bool hit_factor) {

    // This traces correct, but looks weird because gaps are not filled
    // return z + hit_tolerance_ws >= min(z_a, z_b) - 0.00015 && z - hit_tolerance_ws <= max(z_a, z_b);

    hit_factor = false;

    // This traces "incorrect", but looks better because gaps are getting filled then
    if (z - hit_tolerance_ws * trace_length <= max(z_a, z_b)) {

        #if GET_SETTING(ssr, abort_on_object_infront)
            hit_factor = (z + hit_tolerance_ws * trace_length) >=
                            min(z_a, z_b) - hit_tolerance_backface;
        #else
            hit_factor = true;
        #endif

        return true;
    }

    return false;
}


void main()
{
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec2 texcoord = get_half_texcoord();

    vec3 normal_vs = get_view_normal(texcoord);
    Material m = unpack_material(GBuffer, texcoord);
    normal_vs = world_normal_to_view(m.normal);

    vec3 ray_start_vs = get_view_pos_at(texcoord);
    float pixeldist = distance(m.position, MainSceneData.camera_pos);

    // Skip skybox and distant pixels
    if (pixeldist > 3000) {
        result = vec2(0);
        return;
    }

    // Important: For clearcoat we trace the outer layer (with low roughness)
    // instead of the high roughness layer
    float roughness = get_effective_roughness(m);

    // Skip pixels with too high roughness
    if (roughness > GET_SETTING(ssr, roughness_fade)) {
        result = vec2(0);
        return;
    }


    // Get ray start
    vec3 view_dir = normalize(ray_start_vs);
    vec3 ray_dir = normalize(reflect(view_dir, normal_vs));

    // Get tangent and binormal
    vec3 tangent, binormal;
    find_arbitrary_tangent(ray_dir, tangent, binormal);

    vec3 importance_ray_dir = vec3(0);


    // Generate ray directions until we find a value which is valid
    int retry = 0;
    for (;retry < NUM_RAYDIR_RETRIES; ++retry) {

        vec2 seed = texcoord + 1.3123 * retry + 0.176445 * (MainSceneData.frame_index % 32);

        // Get random sequence, should probably use halton or so
        vec2 xi = clamp(abs(rand_rgb(seed).xy), vec2(0.01), vec2(0.99));

        // Clamp brdf tail, see frostbite slides for details
        const float brdf_bias = 0.8;
        xi.y = mix(xi.y, 0.0, brdf_bias);

        // Get importance sampled directory
        vec3 rho = importance_sample_ggx(xi, clamp(sqrt(roughness), 0.0001, 1.0));
        importance_ray_dir = normalize(1e-5 + rho.x * tangent + rho.y * binormal + rho.z * ray_dir);

        // If the ray dir is fine, abort, otherwise continue
        if (dot(importance_ray_dir, normal_vs) > 0.005) {
            break;
        }
    }

    // Still didn't find a good ray dir
    if (retry >= NUM_RAYDIR_RETRIES - 1) {
        result = vec2(0);
        return;
    }

    ray_dir = importance_ray_dir;

    // Ray not in view
    if (dot(ray_dir, view_dir) < 1e-5) {
        result = vec2(0);
        return;
    }

    float max_ray_len = 1.0 * pixeldist;

    // Clip ray to near plane
    #if 1
        float ray_len = ((ray_start_vs.z + ray_dir.z * max_ray_len) > CAMERA_NEAR) ?
                            (CAMERA_NEAR - ray_start_vs.z) / ray_dir.z :
                            max_ray_len;
    #else
        float ray_len = max_ray_len;
    #endif

    // Convert start and end pos from view to screen space
    vec3 ray_start_screen = view_to_screen(ray_start_vs);
    vec3 ray_end_screen = view_to_screen(ray_start_vs + ray_len * ray_dir);

    vec3 ray_pos = ray_start_screen;
    vec3 ray_dir_screen = ray_end_screen - ray_start_screen;

    // Make sure the ray does not leave the screen
    float scale_max_x = min(1, 0.99 * (1.0 - ray_start_screen.x) / max(1e-5, ray_dir_screen.x));
    float scale_max_y = min(1, 0.99 * (1.0 - ray_start_screen.y) / max(1e-5, ray_dir_screen.y));
    float scale_min_x = min(1, 0.99 * ray_start_screen.x / max(1e-5, -ray_dir_screen.x));
    float scale_min_y = min(1, 0.99 * ray_start_screen.y / max(1e-5, -ray_dir_screen.y));
    ray_dir_screen *= min(scale_max_x, scale_max_y);
    ray_dir_screen *= min(scale_min_x, scale_min_y);

    #if USE_LINEAR_DEPTH
        // Linearize depth so we can interpolate it
        ray_start_screen.z = get_linear_z_from_z(ray_start_screen.z);
        ray_end_screen.z = get_linear_z_from_z(ray_end_screen.z);
    #endif

    vec3 ray_step = (ray_end_screen - ray_start_screen) / num_steps;

    float distance_scale = 1.0 + 0.00001 * pixeldist;

    // Initial ray bias to avoid self intersection
    // ray_pos += (15.0 + roughness * 0.0) * ray_step * float(num_steps) / 512.0 / clamp(dot(normal_vs, -view_dir), 0.1, 1.0) * GET_SETTING(ssr, intial_bias) * distance_scale;

    vec2 intersection = vec2(-1);

    // Jitter ray position to make sure we catch all details
    float jitter = abs(rand(ivec2(gl_FragCoord.xy) % 8 +
        (MainSceneData.frame_index % GET_SETTING(ssr, history_length)) * 0.1));
    
    // Rough sourfaces need more jitter
    jitter *= max(0.5, m.roughness);

    ray_pos += jitter * ray_step;

    int i = 0;
    float intersection_weight = 0.0;
    bool hit_factor = false;

    for (i = 1; i < num_steps; ++i) {
        ray_pos += ray_step;

        // Current coordinate is in the mid of two samples, not at the end, so
        // substract half of a step
        vec2 curr_coord = ray_pos.xy - 0.5 * ray_step.xy;


        // Increase ray bias as we advance the ray
        float trace_len = GET_SETTING(ssr, intial_bias) * 10.0 +
            100.0 * distance_squared(curr_coord, texcoord);
        trace_len *= 1.0 + 1.0 * roughness;

        // Check for intersection
        #if USE_LINEAR_DEPTH
            float depth_sample = textureLod(DownscaledDepth, curr_coord, 0).x;
        #else
            float depth_sample = textureLod(GBuffer.Depth, curr_coord, 0).x;
        #endif

        if (point_between_planes(depth_sample, ray_pos.z,
                                    ray_pos.z - ray_step.z, trace_len, hit_factor)) {
            intersection = curr_coord;
            break;
        }
    }

    // Make sure we hit exactly one pixel
    intersection = truncate_coordinate(intersection);

    // Check if we hit something
    if (min(intersection.x, intersection.y) <= 0.0 || out_of_screen(intersection)) {
        result = vec2(0);
        return;
    }

    if (!hit_factor) {
        result = vec2(0);
        return;
    }
    result = intersection;
}
