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

uniform sampler2D ShadedScene;

#if GET_SETTING(volumetrics, enable_volumetric_shadows)
    uniform sampler2D VolumetricsTex;
#endif

out vec3 result;

float compute_fog(vec3 ray_start, vec3 ray_end)
{
    float dist = length(ray_start - ray_end);
    vec3 ray_dir = (ray_start - ray_end) / dist;
    const float c = 1.0 * TimeOfDay.volumetrics.fog_intensity;
    const float b = 0.2 / TimeOfDay.volumetrics.fog_ramp_size;
    ray_dir.z = -ray_dir.z;
    if (abs(ray_dir.z) < 1e-2)
        ray_dir.z = 1e-2;
    return saturate(c * exp(-ray_start.z * b) * (1.0 - exp(-dist * ray_dir.z * b)) / ray_dir.z);
}

void main() {
    vec2 texcoord = get_texcoord();

    float depth = get_depth_at(texcoord);
    vec3 surface_pos = calculate_surface_pos(depth, texcoord);

    #if GET_SETTING(volumetrics, enable_volumetric_shadows)
        vec4 volumetrics = textureLod(VolumetricsTex, texcoord, 0);
    #else
        vec4 volumetrics = vec4(0);
    #endif

    vec3 scene_color = textureLod(ShadedScene, texcoord, 0).xyz;
    

    vec3 fog_color = TimeOfDay.volumetrics.fog_color * 4.5 * TimeOfDay.volumetrics.fog_brightness * TimeOfDay.scattering.sun_intensity;

    vec3 ray_dir = normalize(MainSceneData.camera_pos - surface_pos);

    float fog_weight = compute_fog(MainSceneData.camera_pos, surface_pos);

    vec3 merged_color = scene_color;
    #if !DEBUG_MODE
        merged_color = mix(scene_color, fog_color, fog_weight);
        merged_color = mix(merged_color, volumetrics.xyz, volumetrics.w);
    #endif


    result = merged_color;
}
