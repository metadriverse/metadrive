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

uniform int sourceMip;
uniform sampler2D SourceTex;
uniform writeonly image2D RESTRICT DestTex;

uniform sampler2D ShadedScene;
uniform sampler2D SumTex;

uniform bool FirstUpsamplePass;


void main() {
    vec2 source_size = vec2(textureSize(SourceTex, sourceMip).xy);
    vec2 texcoord = (ivec2(gl_FragCoord.xy) + 0.5) / (2.0 * source_size);
    vec3 summed = textureLod(SourceTex, texcoord, sourceMip + 1).xyz;

    if (FirstUpsamplePass) {
        summed = vec3(0);
    }

    vec3 mip_color = vec3(1);

    // Hardcoded mipmap colors for now, should make this adjustable at some point
    switch(sourceMip) {
        case 0: mip_color = vec3(0.214, 0.429, 0.497); break;
        case 1: mip_color = vec3(0.964, 0.947, 0.991); break;
        case 2: mip_color = vec3(0.982, 0.542, 0.542); break;
        case 3: mip_color = vec3(0.301, 0.493, 1.000); break;
        case 4: mip_color = vec3(0.456, 0.209, 0.167); break;
    }

    // upsample texture
    ivec2 coord = ivec2(gl_FragCoord.xy);

    vec2 flt_coord = vec2(coord) / (2.0 * source_size);
    vec2 texel_size = 1.0 / source_size;

    vec3 source_sample = textureLod(SourceTex, flt_coord, sourceMip).xyz * 4;

    source_sample += textureLod(
        SourceTex, flt_coord + vec2(0, 1) * texel_size, sourceMip).xyz * 2;
    source_sample += textureLod(
        SourceTex, flt_coord + vec2(0, -1) * texel_size, sourceMip).xyz * 2;
    source_sample += textureLod(
        SourceTex, flt_coord + vec2(1, 0) * texel_size, sourceMip).xyz * 2;
    source_sample += textureLod(
        SourceTex, flt_coord + vec2(-1, 0) * texel_size, sourceMip).xyz * 2;

    source_sample += textureLod(
        SourceTex, flt_coord + vec2(-1, -1) * texel_size, sourceMip).xyz;
    source_sample += textureLod(
        SourceTex, flt_coord + vec2(1, -1) * texel_size, sourceMip).xyz;
    source_sample += textureLod(
        SourceTex, flt_coord + vec2(-1, 1) * texel_size, sourceMip).xyz;
    source_sample += textureLod(
        SourceTex, flt_coord + vec2(1, 1) * texel_size, sourceMip).xyz;

    source_sample /= 16.0;
    source_sample *= mip_color;

    vec3 pass_result = summed + source_sample;
    pass_result *= 0.5;

    vec4 old_data = texelFetch(SourceTex, coord, sourceMip - 1);
    imageStore(DestTex, coord, old_data + vec4(pass_result, 0));
}
