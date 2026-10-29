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

in vec4 p3d_Vertex;
uniform isamplerBuffer CellListBuffer;
uniform int threadCount;

uint get_slice_count(uint cell_count) {
    return uint(ceil(float(cell_count * threadCount) / LC_CULLING_SLICE_WIDTH));
}

void main() {

    // XXX: Get rid of all of this unnecessary uint divides. It does not matter
    // much for performance (since this is a vertex shader), but the logic
    // is still awful and could be much simpler.

    // I don't want to move the logic to the CPU though, as passing uniforms
    // is totally slow without UBO's.

    // Find the amount of cells to shade
    int num_total_cells = texelFetch(CellListBuffer, 0).x;

    // Compute the amount of pixel lines required to shade all cells
    uint num_used_slices = get_slice_count(num_total_cells);

    // Compute the percentage factor of the used slices
    uint max_tiles = MainSceneData.lc_tile_count.x *
                        MainSceneData.lc_tile_count.y *
                        LC_TILE_SLICES;
    uint max_slices = get_slice_count(max_tiles);

    float percentage_height = saturate(num_used_slices / float(max_slices));
    // percentage_height = 1;

    // Store the vertex position.
    gl_Position = vec4(
        p3d_Vertex.x,
        fma(fma(p3d_Vertex.z, 0.5, 0.5) * percentage_height, 2.0, -1.0) , 0, 1);
}
