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

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/envprobes.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/light_culling.inc.glsl"

out vec4 result;

uniform isamplerBuffer CellListBuffer;
uniform writeonly iimageBuffer RESTRICT PerCellProbes;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    // Find the index of the cell we are about to process
    int idx = 1 + coord.x + coord.y * LC_CULLING_SLICE_WIDTH;
    int num_total_cells = texelFetch(CellListBuffer, 0).x;
    ivec2 precompute_size = MainSceneData.lc_tile_count;

    // If we found no remaining cell, we are done, so just return and hope for
    // good coherency.
    if (idx > num_total_cells) {
        discard;
    }

    // Fetch the cell data, this contains the cells position
    int packed_cell_data = texelFetch(CellListBuffer, idx).x;
    int cell_x, cell_y, cell_slice;
    unpack_cell_data(packed_cell_data, cell_x, cell_y, cell_slice);

    float distance_bias = 0.01;
    float bsphere_bias = 0.05;
    // Find the tiles minimum and maximum distance
    float min_distance = get_distance_from_slice(cell_slice) - distance_bias;
    float max_distance = get_distance_from_slice(cell_slice + 1) + distance_bias;

    // Compute sample directions
    vec3 local_ray_dirs[num_raydirs] = get_raydirs(cell_x, cell_y, precompute_size);

    int storage_offset = idx * MAX_PROBES_PER_CELL;
    int probes_written = 0;

    // Iterate over all probes
    for(int i = 0; i < EnvProbes.num_probes && probes_written < MAX_PROBES_PER_CELL; ++i) {
        Cubemap map = get_cubemap(i);
        vec4 pos_view = MainSceneData.view_mat_z_up * vec4(map.bounding_sphere_center, 1);

        bool visible = false;

        Sphere sphere;
        sphere.pos = pos_view.xyz;
        sphere.radius = map.bounding_sphere_radius + bsphere_bias;

        // Check for visibility
        for (int k = 0; k < num_raydirs; ++k) {
            visible = visible || viewspace_ray_sphere_distance_intersection(
                sphere, local_ray_dirs[k], min_distance, max_distance);
        }

        if (visible) {
            imageStore(PerCellProbes, storage_offset + probes_written, ivec4(1 + i));
            ++probes_written;
        }
    }

    // Append zero byte
    // if (probes_written < MAX_PROBES_PER_CELL) {
    //     imageStore(PerCellProbes, storage_offset + probes_written, ivec4(0));
    // }

    for (int i = probes_written; i < MAX_PROBES_PER_CELL; ++i) {
        imageStore(PerCellProbes, storage_offset + i, ivec4(0));
    }

    result = vec4(cell_x % 2, 0.6, 1.0, 1.0);
}
