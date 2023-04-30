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

flat in int instance_id;

uniform int sourceMip;
uniform sampler3D SourceTex;
uniform writeonly image3D RESTRICT DestTex;

void main() {
    ivec3 coord = ivec3(gl_FragCoord.xy, instance_id);

    ivec3 parent_coord = coord * 2;

    vec4 accum = vec4(0);
    accum += texelFetch(SourceTex, parent_coord + ivec3(0, 0, 0), sourceMip);
    accum += texelFetch(SourceTex, parent_coord + ivec3(1, 0, 0), sourceMip);
    accum += texelFetch(SourceTex, parent_coord + ivec3(0, 1, 0), sourceMip);
    accum += texelFetch(SourceTex, parent_coord + ivec3(1, 1, 0), sourceMip);

    accum += texelFetch(SourceTex, parent_coord + ivec3(0, 0, 1), sourceMip);
    accum += texelFetch(SourceTex, parent_coord + ivec3(1, 0, 1), sourceMip);
    accum += texelFetch(SourceTex, parent_coord + ivec3(0, 1, 1), sourceMip);
    accum += texelFetch(SourceTex, parent_coord + ivec3(1, 1, 1), sourceMip);
    accum /= 8.0;
    imageStore(DestTex, coord, accum);
}
