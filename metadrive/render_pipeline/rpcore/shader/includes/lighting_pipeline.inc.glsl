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

#pragma include "includes/material.inc.glsl"
#pragma include "includes/light_culling.inc.glsl"
#pragma include "includes/lights.inc.glsl"
#pragma include "includes/light_data.inc.glsl"
#pragma include "includes/shadows.inc.glsl"
#pragma include "includes/noise.inc.glsl"
#pragma include "includes/light_classification.inc.glsl"
#pragma include "includes/poisson_disk.inc.glsl"

uniform isampler2DArray CellIndices;
uniform usamplerBuffer PerCellLights;
uniform usamplerBuffer PerCellLightsCounts;
uniform samplerBuffer AllLightsData;
uniform samplerBuffer ShadowSourceData;

uniform sampler2D ShadowAtlas;

#if SUPPORT_PCF
uniform sampler2DShadow ShadowAtlasPCF;
#endif

int get_pointlight_source_offs(vec3 direction) {
    vec3 abs_dir = abs(direction);
    float max_comp = max3(abs_dir.x, abs_dir.y, abs_dir.z);
    if (abs_dir.x >= max_comp - 1e-5) return direction.x >= 0.0 ? 0 : 1;
    if (abs_dir.y >= max_comp - 1e-5) return direction.y >= 0.0 ? 2 : 3;
    return direction.z >= 0.0 ? 4 : 5;
}

// Processes a spot light
vec3 process_spotlight(Material m, LightData light_data, vec3 view_vector, float shadow_factor) {
    const vec3 transmittance = vec3(1); // <-- TODO

    // Get the lights data
    int ies_profile = get_ies_profile(light_data);
    vec3 position = get_light_position(light_data);
    float radius = get_spotlight_radius(light_data);
    float fov = get_spotlight_fov(light_data);
    vec3 direction = get_spotlight_direction(light_data);
    vec3 l = position - m.position;
    vec3 l_norm = normalize(l);

    float attenuation = get_spotlight_attenuation(
        l_norm, direction, fov, radius, dot(l, l), ies_profile);

    // Compute the lights influence
    return apply_light(
        m, view_vector, l_norm, get_light_color(light_data), attenuation,
        shadow_factor, transmittance, ONE_BY_PI, ONE_BY_PI, l_norm);
}

// Processes a point light
vec3 process_pointlight(Material m, LightData light_data, vec3 view_vector, float shadow_factor) {
    const vec3 transmittance = vec3(1); // <-- TODO

    // Get the lights data
    float radius = get_pointlight_radius(light_data);
    float inner_radius = get_pointlight_inner_radius(light_data);
    vec3 position = get_light_position(light_data);
    int ies_profile = get_ies_profile(light_data);
    vec3 l = position - m.position;
    float l_len_square = length_squared(l);

    float energy = 1.0 / FOUR_PI;
    float dist_sq = l_len_square;
    float clearcoat_energy = energy;
    vec3 l_diff = l;

    // Spherical area light
    if (inner_radius > 0.02) {
        l_diff = get_spherical_area_light_horizon(l, m.normal, inner_radius);
        l = get_spherical_area_light_vector(m.normal, l, view_vector, inner_radius);
        energy = get_spherical_area_light_energy(m.roughness, inner_radius, dist_sq);
        dist_sq = max(square(inner_radius) + 0.01, l_len_square - square(inner_radius));
        clearcoat_energy = get_spherical_area_light_energy(CLEARCOAT_ROUGHNESS, inner_radius, dist_sq);
    }

    // Get the point light attenuation
    float attenuation = attenuation_curve(dist_sq, radius) * get_ies_factor(-l, ies_profile);

    // Compute the lights influence
    return apply_light(m, view_vector, normalize(l), get_light_color(light_data),
        attenuation, shadow_factor, transmittance, energy, clearcoat_energy, l_diff);
}



// Filters a shadow map
float filter_shadowmap(Material m, SourceData source, vec3 l) {

    // TODO: Examine if this is faster
    // if (dot(m.normal, -l) < 0) return 0.0;

    mat4 mvp = get_source_mvp(source);
    vec4 uv = get_source_uv(source);

    float rotation = interleaved_gradient_noise(
        gl_FragCoord.xy + MainSceneData.frame_index % 32);
    mat2 rotation_mat = make_rotation_mat(rotation);

    // TODO: make this configurable
    // XXX: Scale by resolution (higher resolution needs smaller bias)
    const float slope_bias = 0.001;
    const float normal_bias = 0.0001;
    const float const_bias = 0.003;
    vec3 biased_pos = get_biased_position(m.position, slope_bias, normal_bias, m.normal, -l);

    vec3 projected = project(mvp, biased_pos);
    vec2 projected_coord = projected.xy * uv.zw + uv.xy;

    const int num_samples = 8;
    const float filter_size = 3.0 / SHADOW_ATLAS_SIZE;

    float accum = 0.0;

    for (int i = 0; i < num_samples; ++i) {
        vec2 offs = projected_coord.xy + (rotation_mat * shadow_sample_offsets_8[i]) * filter_size;
        #if SUPPORT_PCF
            accum += textureLod(ShadowAtlasPCF, vec3(offs, projected.z - const_bias), 0).x;
        #else
            accum += textureLod(ShadowAtlas, vec2(offs), 0).x >
                projected.z - const_bias ? 1.0 : 0.0;
        #endif
    }

    return accum / num_samples;
}



// Shades the material from the per cell light buffer
vec3 shade_material_from_tile_buffer(Material m, ivec3 tile) {

    #if DEBUG_MODE && !MODE_ACTIVE(LIGHT_COUNT) && !SPECIAL_MODE_ACTIVE(LIGHT_TILES)
        return vec3(0);
    #endif

    // Skip emissive materials
    if (m.shading_model == SHADING_MODEL_EMISSIVE)
        return vec3(0);

    vec3 shading_result = vec3(0);

    // Find per tile lights
    int cell_index = texelFetch(CellIndices, tile, 0).x;
    int count_offs = cell_index * (1 + LIGHT_CLS_COUNT); // 1 for total count, rest for light classes


    // Get the light counts
    uint num_total_lights  = texelFetch(PerCellLightsCounts, count_offs).x;

    // Early out when no lights are there
    #if !MODE_ACTIVE(LIGHT_COUNT) && !SPECIAL_MODE_ACTIVE(LIGHT_TILES)
        if (num_total_lights == 0) {
            return vec3(0);
        }
    #endif

    // Get the per-class counts
    uint num_spot_noshadow  = texelFetch(PerCellLightsCounts, count_offs + 1 + LIGHT_CLS_SPOT_NOSHADOW).x;
    uint num_spot_shadow    = texelFetch(PerCellLightsCounts, count_offs + 1 + LIGHT_CLS_SPOT_SHADOW).x;
    uint num_point_noshadow = texelFetch(PerCellLightsCounts, count_offs + 1 + LIGHT_CLS_POINT_NOSHADOW).x;
    uint num_point_shadow   = texelFetch(PerCellLightsCounts, count_offs + 1 + LIGHT_CLS_POINT_SHADOW).x;

    // Compute the index into the culled lights list
    int data_offs = cell_index * LC_MAX_LIGHTS_PER_CELL;
    int curr_offs = data_offs;

    #if MODE_ACTIVE(LIGHT_COUNT)
        float factor = num_total_lights / float(LC_MAX_LIGHTS_PER_CELL);
        return vec3(factor, 1 - factor, 0);
    #endif

    // Debug mode, show tile bounds
    #if SPECIAL_MODE_ACTIVE(LIGHT_TILES)
        // Show tiles
        #if IS_SCREEN_SPACE
            if (int(gl_FragCoord.x) % LC_TILE_SIZE_X == 0 ||
                int(gl_FragCoord.y) % LC_TILE_SIZE_Y == 0) {
                shading_result += 1.0;
            }
            float light_factor = num_total_lights / float(LC_MAX_LIGHTS_PER_CELL);
            shading_result += ((tile.z + 1) % 2) * 0.2;
            shading_result += light_factor;
        #endif
    #endif

    // Compute view vector
    vec3 v = normalize(MainSceneData.camera_pos - m.position);

    // Spotlights without shadow
    for (int i = 0; i < num_spot_noshadow; ++i) {
        int light_offs = int(texelFetch(PerCellLights, curr_offs++).x);
        LightData light_data = read_light_data(AllLightsData, light_offs);
        shading_result += process_spotlight(m, light_data, v, 1.0);
    }

    // Pointlights without shadow
    for (int i = 0; i < num_point_noshadow; ++i) {
        int light_offs = int(texelFetch(PerCellLights, curr_offs++).x);
        LightData light_data = read_light_data(AllLightsData, light_offs);
        shading_result += process_pointlight(m, light_data, v, 1.0);
    }

    // Spotlights with shadow
    for (int i = 0; i < num_spot_shadow; ++i) {
        int light_offs = int(texelFetch(PerCellLights, curr_offs++).x);
        LightData light_data = read_light_data(AllLightsData, light_offs);

        // Get shadow factor
        vec3 v2l = normalize(m.position - get_light_position(light_data));
        int source_index = get_shadow_source_index(light_data);
        SourceData source_data = read_source_data(ShadowSourceData, source_index * 5);
        float shadow_factor = filter_shadowmap(m, source_data, v2l);
        shading_result += process_spotlight(m, light_data, v, shadow_factor);
    }

    // Pointlights with shadow
    for (int i = 0; i < num_point_shadow; ++i) {
        int light_offs = int(texelFetch(PerCellLights, curr_offs++).x);
        LightData light_data = read_light_data(AllLightsData, light_offs);

        // Get shadow factor
        int source_index = get_shadow_source_index(light_data);
        vec3 v2l = normalize(m.position - get_light_position(light_data));
        source_index += get_pointlight_source_offs(v2l);

        SourceData source_data = read_source_data(ShadowSourceData, source_index * 5);
        float shadow_factor = filter_shadowmap(m, source_data, v2l);
        shading_result += process_pointlight(m, light_data, v, shadow_factor);
    }

    // Fade out lights as they reach the culling distance
    float curr_dist = distance(m.position, MainSceneData.camera_pos);
    float fade = saturate(curr_dist / LC_MAX_DISTANCE);
    fade = 1 - pow(fade, 10.0);

    return shading_result * fade;
}
