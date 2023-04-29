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

// Required inputs
uniform sampler3D SceneVoxels;
uniform samplerCube ScatteringIBLSpecular;
uniform samplerCube ScatteringIBLDiffuse;

// Voxel grid parameters
uniform vec3 voxelGridPosition;
uniform float voxelGridSize;
uniform int voxelGridResolution;


vec3 worldspace_to_voxelspace(vec3 worldspace) {
    vec3 voxel_coord = (worldspace - voxelGridPosition) / GET_SETTING(vxgi, grid_ws_size);
    return fma(voxel_coord, vec3(0.5), vec3(0.5));
}

float get_mipmap_from_cone_radius(float cone_radius) {
    return log2(cone_radius * GET_SETTING(vxgi, grid_resolution) * 0.6) - 1;
}

vec4 trace_cone(vec3 start_pos, vec3 nrm, vec3 direction, int max_steps,
        bool is_specular, float cone_grow_factor, float seed) {

    // Find initial cone radius
    float cone_radius = (1.0 + 5.0 * cone_grow_factor) / GET_SETTING(vxgi, grid_resolution);

    // Offset start position to avoid self intersection
    start_pos += nrm * 1.5 / GET_SETTING(vxgi, grid_resolution);
    start_pos += direction * 5.5 / GET_SETTING(vxgi, grid_resolution);

    if (!is_specular) {
        // start_pos += nrm * 1.5 / GET_SETTING(vxgi, grid_resolution);
    }

    // Cone parameters
    vec3 current_pos = start_pos;
    vec4 accum = vec4(0);
    float mipmap = 0.0;

    current_pos += direction * cone_radius * seed;

    // Trace the cone over the voxel grid
    for (int i = 0; i < max_steps; ++i) {
        mipmap = get_mipmap_from_cone_radius(cone_radius);
        vec4 sampled = textureLod(SceneVoxels, current_pos, mipmap);
        sampled.w *= 2.0;
        sampled.w = saturate(sampled.w);
        accum += sampled * (1.0 - accum.w);
        current_pos += direction * cone_radius / 2.25;
        cone_radius *= 1.01 + cone_grow_factor;
    }

    // Unpack packed color, since we use 8 bit targets only
    accum.xyz = accum.xyz / (1 - accum.xyz);
    // accum.xyz *= 15.0;

    if (is_specular) {
        // accum.xyz += textureLod(ScatteringIBLSpecular, direction, mipmap).xyz * (1-accum.w);
    } else {
        // accum.xyz += textureLod(ScatteringIBLDiffuse, direction, 0).xyz * (1-accum.w) * 0.1;
    }

    return accum;
}
