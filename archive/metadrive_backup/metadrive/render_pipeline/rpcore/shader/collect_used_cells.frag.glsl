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

// Shader which iterates over the camera-space voxel grid and collects
// all cells which have been flagged into a big atomic driven list.

#pragma optionNV (unroll all)

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/light_culling.inc.glsl"

uniform sampler2DArray FlaggedCells;
layout(r32i) uniform iimageBuffer CellListBuffer;
uniform writeonly iimage2DArray RESTRICT CellListIndices;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    // Iterate over all slices
    for (int i = 0; i < LC_TILE_SLICES; ++i) {

        // Check if the cell is flagged
        bool visible = texelFetch(FlaggedCells, ivec3(coord, i), 0).x > 0.5;
        if (visible) {
            // Append the cell and mark it
            // Notice: We add 1 since the first index stores the amount
            // of collected cells.
            int flag_index = imageAtomicAdd(CellListBuffer, 0, 1) + 1;
            int cell_data = coord.x | coord.y << 10 | i << 20;
            imageStore(CellListBuffer, flag_index, ivec4(cell_data));
            imageStore(CellListIndices, ivec3(coord, i), ivec4(flag_index));
        }
    }
}
