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
#pragma include "includes/brdf.inc.glsl"
#pragma include "includes/color_spaces.inc.glsl"
#pragma include "includes/light_culling.inc.glsl"

// Global probe data
uniform struct {
    int num_probes;
    samplerCubeArray cubemaps;
    samplerCubeArray diffuse_cubemaps;
    samplerBuffer dataset;
} EnvProbes;

// Per probe instance
struct Cubemap {
    mat4 transform;
    bool use_parallax;
    float border_smoothness;
    uint index;

    vec3 bounding_sphere_center;
    float bounding_sphere_radius;
};

// Per cell data
#ifdef APPLY_ENVPROBES_PASS
    uniform isampler2DArray CellIndices;
    uniform isamplerBuffer PerCellProbes;
#endif

Cubemap get_cubemap(int index) {
    Cubemap result;
    int offs = index * 5;
    vec4 data0 = texelFetch(EnvProbes.dataset, offs);
    vec4 data1 = texelFetch(EnvProbes.dataset, offs + 1);
    vec4 data2 = texelFetch(EnvProbes.dataset, offs + 2);
    vec4 data3 = texelFetch(EnvProbes.dataset, offs + 3);
    vec4 data4 = texelFetch(EnvProbes.dataset, offs + 4);

    // Unpack the packed matrix, we only store a 3x4 matrix to save space,
    // sine the last row is always (0, 0, 0, 1)
    result.transform = mat4(
            data0.x, data0.y, data0.z, 0,
            data0.w, data1.x, data1.y, 0,
            data1.z, data1.w, data2.x, 0,
            data2.y, data2.z, data2.w, 1
    );
    result.index = uint(data3.x + 0.5);
    result.use_parallax = data3.y > 0.5;
    result.border_smoothness = data3.z;
    result.bounding_sphere_center = data4.xyz;
    result.bounding_sphere_radius = data4.w;
    return result;
}

// https://seblagarde.wordpress.com/2012/09/29/image-based-lighting-approaches-and-parallax-corrected-cubemap/
float correct_parallax(Cubemap map, Material m, vec3 vector, out float factor) {
    // Intersection with OBB, convert to unit box space
    // Transform in local unit parallax cube space (scaled and rotated)
    vec3 ray_ls = (map.transform * vec4(vector, 0)).xyz;
    vec3 position_ls = (map.transform * vec4(m.position, 1)).xyz;

    // Get fading factor
    vec3 local_v = abs(position_ls);
    factor = max(local_v.x, max(local_v.y, local_v.z));

    if (!map.use_parallax) {
        return 1e10;
    }

    // Intersect with unit box
    vec3 first_plane = (1.0 - position_ls) / ray_ls;
    vec3 second_plane = (-1.0 - position_ls) / ray_ls;
    vec3 furthest_plane = max(first_plane, second_plane);
    return min(furthest_plane.x, min(furthest_plane.y, furthest_plane.z));
}

vec3 get_cubemap_vector(Cubemap map, Material m, vec3 vector, out float factor, out float dist) {
    dist = correct_parallax(map, m, vector, factor);

    // Use distance in world space directly to recover intersection.
    // Mix parallax corrected and original vector based on roughness
    vec3 intersection_pos = mix(
        m.position + vector * dist,
        map.bounding_sphere_center + vector,
        m.roughness);
    return (map.transform * vec4(intersection_pos, 1)).xyz;
}

vec3 get_reflection_vector(Cubemap map, Material m, out float factor, out float dist) {
    vec3 view_vector = normalize(m.position - MainSceneData.camera_pos);
    vec3 reflected = get_reflection_vector(m, view_vector);
    return get_cubemap_vector(map, m, reflected, factor, dist);
}

vec3 get_diffuse_vector(Cubemap map, Material m) {
    #if 0
        // very expensive, have to think of a better solution -
        // maybe we can precompute the matrix and store it.
        mat3 tpose_inverse = transpose(inverse(mat3(map.transform)));
        return tpose_inverse * (m.normal);
    #else
        // This is mathematically wrong, but it works fast and reasonable
        vec4 transformed = map.transform * vec4(fma(m.normal, vec3(1e5),
            map.bounding_sphere_center), 1);
        return transformed.xyz;
    #endif
}

void apply_cubemap(int id, Material m, out vec4 diffuse, out vec4 specular,
    inout float total_weight, inout float total_blend) {

    const float max_mip = 7.0;
    float roughness = get_effective_roughness(m);

    float factor = 0.0;
    float mipmap = m.linear_roughness * 10.0;
    float intersection_distance = 1.0;

    Cubemap map = get_cubemap(id);

    vec3 direction = get_reflection_vector(map, m, factor, intersection_distance);
    vec3 diffuse_direction = get_diffuse_vector(map, m);

    vec3 vector_to_source = normalize(map.bounding_sphere_center - m.position);

    // float normal_blend_factor = saturate(0.1 + 1 * dot(vector_to_source, m.normal));
    float normal_blend_factor = saturate(3.0 * dot(vector_to_source, m.normal));
    float blend = saturate((1 - factor) / max(1e-10, map.border_smoothness));
    blend *= normal_blend_factor;

    // Make sure the gradient looks right after tonemapping
    // blend = square(blend);

    float local_distance = intersection_distance / map.bounding_sphere_radius;

    #if 0
        if (map.use_parallax) {
            // mipmap *= intersection_distance * 0.012;
        } else {
            mipmap *= 0.1;
        }
    #else
        // mipmap *= 0.1;
    #endif


    specular = textureLod(EnvProbes.cubemaps,
        vec4(direction, map.index), clamp(mipmap, 0.0, max_mip));
    diffuse = textureLod(EnvProbes.diffuse_cubemaps,
        vec4(diffuse_direction, map.index), 0);

    // Optional: Correct specular based on diffuse scolor intensity
    // specular.xyz = mix(specular.xyz, specular.xyz * get_luminance(diffuse.xyz), diffuse.w);

    // Make sure small probes contribute much more than large ones
    float weight = exp(-0.05 * map.bounding_sphere_radius);
    weight *= factor >= 1.0 ? 0.0 : 1.0;

    // Apply clip factors
    specular *= weight * blend;
    diffuse *= weight * blend;

    total_weight += weight * blend;
    total_blend += blend;
}


#ifdef APPLY_ENVPROBES_PASS
    void apply_all_probes(Material m, out vec4 specular, out vec4 diffuse) {
        ivec3 tile = get_lc_cell_index(
            ivec2(gl_FragCoord.xy),
            distance(MainSceneData.camera_pos, m.position));

        // Don't shade pixels out of the shading range
        if (tile.z >= LC_TILE_SLICES) {
            specular = vec4(0);
            diffuse = vec4(0);
            return;
        }

        int cell_index = texelFetch(CellIndices, tile, 0).x;
        int data_offs = cell_index * MAX_PROBES_PER_CELL;

        vec4 total_diffuse = vec4(0);
        vec4 total_specular = vec4(0);
        float total_blend = 0;
        float total_weight = 0;

        int processed_probes = 0;
        for (int i = 0; i < MAX_PROBES_PER_CELL; ++i) {
            int cubemap_index = texelFetch(PerCellProbes, data_offs + i).x - 1;
            if (cubemap_index < 0) break;
            vec4 diff, spec;

            processed_probes += 1;
            apply_cubemap(cubemap_index, m, diff, spec, total_weight, total_blend);
            total_diffuse += diff;
            total_specular += spec;
        }

        float scale = 1.0 / max(1e-9, total_weight) * min(1.0, total_blend);

        vec4 result_spec = total_specular * scale;
        vec4 result_diff = total_diffuse * scale;

        // Fade out cubemaps as they reach the culling distance
        float curr_dist = distance(m.position, MainSceneData.camera_pos);
        float fade = saturate(curr_dist / LC_MAX_DISTANCE);
        fade = 1 - pow(fade, 5.0);

        result_spec *= fade;
        result_diff *= fade;

        // Visualize probe count
        #if MODE_ACTIVE(ENVPROBE_COUNT)
            float probe_factor = float(processed_probes) / MAX_PROBES_PER_CELL;
            result_spec = result_diff = vec4(probe_factor, 1 - probe_factor, 0, 1);
        #endif

        specular = result_spec;
        diffuse = result_diff;
    }

#endif
