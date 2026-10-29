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

// This shader collects the list of all lights for a given cell, sorts them
// and then writes them out as a list.

layout(early_fragment_tests) in;

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/light_culling.inc.glsl"
#pragma include "includes/light_data.inc.glsl"
#pragma include "includes/light_classification.inc.glsl"

uniform isamplerBuffer CellListBuffer;
uniform usamplerBuffer PerCellLightsBuffer;
layout(r32i) uniform iimageBuffer PerCellLightCountsBuffer;
uniform writeonly uimageBuffer RESTRICT GroupedCellLightsBuffer;
uniform writeonly uimageBuffer RESTRICT GroupedPerCellLightsCountBuffer;

uniform samplerBuffer AllLightsData;
uniform int maxLightIndex;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    // Find the index of the cell we are about to process
    int idx = 1 + coord.x + coord.y * LC_CULLING_SLICE_WIDTH;
    int num_total_cells = texelFetch(CellListBuffer, 0).x;

    // If we found no remaining cell, we are done, so just return and hope for
    // good coherency.
    if (idx > num_total_cells) {
        return;
    }

    // Get the offset in the per-cell culled light list
    int data_offset = LC_MAX_LIGHTS_PER_CELL * idx; // Culled lights offset
    int dest_offset = LC_MAX_LIGHTS_PER_CELL * idx; // Destination offset

    // Offset into the per-cell culled light counts 
    int cell_lights_offset = idx * (1 + LIGHT_CLS_COUNT);

    // Get the amount of (unsorted) lights for this cell
    int num_culled_lights = imageLoad(PerCellLightCountsBuffer, idx).x; 
    imageStore(PerCellLightCountsBuffer, idx, ivec4(0));

    num_culled_lights = min(LC_MAX_LIGHTS_PER_CELL, num_culled_lights);
    int num_processed_lights = 0;

    // Fetch the data of all light classes
    for (int light_class = 0; light_class < LIGHT_CLS_COUNT; ++light_class) {
        int light_count = 0;

        // Iterate over the list of culled lights, and store all lights which
        // belong to this class
        for (int i = 0; i < num_culled_lights; ++i) {
            // XXX: Have to use int, because of stupid OpenGL specification,
            // indexes into buffers are ints and not unsigned ints.
            int light_index = int(texelFetch(PerCellLightsBuffer, data_offset + i).x);
            int light_type = read_light_type(AllLightsData, light_index);
            bool casts_shadows = read_casts_shadows(AllLightsData, light_index);

            // In case the culled light is of the same type as the type we are looking for
            if (classify_light(light_type, casts_shadows) == light_class) {
                imageStore(
                    GroupedCellLightsBuffer,
                    dest_offset + num_processed_lights, uvec4(light_index));
                ++light_count;
                ++num_processed_lights;
            }
        }

        // Store the light class count
        imageStore(GroupedPerCellLightsCountBuffer, cell_lights_offset + 1 + light_class, uvec4(light_count));
    }

    // Store the total light count to be able to early out
    imageStore(GroupedPerCellLightsCountBuffer, cell_lights_offset, uvec4(num_processed_lights));
}
