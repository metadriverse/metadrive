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

void main() {
    ivec2 int_coord = ivec2(gl_FragCoord.xy);
    vec2 parent_tex_size = vec2(textureSize(SourceTex, sourceMip).xy);
    vec2 texel_size = 1.0 / parent_tex_size;

    // Compute the floating point coordinate pointing to the exact center of the
    // parent texel center
    vec2 flt_coord = vec2(int_coord + 0.5) / parent_tex_size * 2.0;

    // Filter the image, see:
    // http://fs5.directupload.net/images/151213/qfnexcls.png
    vec3 center_sample = textureLod(SourceTex, flt_coord, sourceMip).xyz;

    // inner samples (marked red)
    vec3 sample_r_tl = textureLod(
        SourceTex, flt_coord + vec2(-1, 1) * texel_size, sourceMip).xyz;
    vec3 sample_r_tr = textureLod(
        SourceTex, flt_coord + vec2(1, 1) * texel_size, sourceMip).xyz;
    vec3 sample_r_bl = textureLod(
        SourceTex, flt_coord + vec2(-1, -1) * texel_size, sourceMip).xyz;
    vec3 sample_r_br = textureLod(
        SourceTex, flt_coord + vec2(1, -1) * texel_size, sourceMip).xyz;

    // corner samples
    vec3 sample_t = textureLod(
        SourceTex, flt_coord + vec2(0, 2) * texel_size, sourceMip).xyz;
    vec3 sample_r = textureLod(
        SourceTex, flt_coord + vec2(2, 0) * texel_size, sourceMip).xyz;
    vec3 sample_b = textureLod(
        SourceTex, flt_coord + vec2(0, -2) * texel_size, sourceMip).xyz;
    vec3 sample_l = textureLod(
        SourceTex, flt_coord + vec2(-2, 0) * texel_size, sourceMip).xyz;

    // edge samples
    vec3 sample_tl = textureLod(
        SourceTex, flt_coord + vec2(-2, 2) * texel_size, sourceMip).xyz;
    vec3 sample_tr = textureLod(
        SourceTex, flt_coord + vec2(2, 2) * texel_size, sourceMip).xyz;
    vec3 sample_bl = textureLod(
        SourceTex, flt_coord + vec2(-2, -2) * texel_size, sourceMip).xyz;
    vec3 sample_br = textureLod(
        SourceTex, flt_coord + vec2(2, -2) * texel_size, sourceMip).xyz;

    vec3 kernel_sum_red = sample_r_tl + sample_r_tr + sample_r_bl + sample_r_br;
    vec3 kernel_sum_yellow = sample_tl + sample_t + sample_l + center_sample;
    vec3 kernel_sum_green = sample_tr + sample_t + sample_r + center_sample;
    vec3 kernel_sum_purple = sample_bl + sample_b + sample_l + center_sample;
    vec3 kernel_sum_blue = sample_br + sample_b + sample_r + center_sample;

    vec3 summed_kernel = kernel_sum_red * 0.5 + kernel_sum_yellow * 0.125 +
                            kernel_sum_green * 0.125 + kernel_sum_purple * 0.125 +
                            kernel_sum_blue * 0.125;

    // since every sub-kernel has 4 samples, normalize that
    summed_kernel /= 4.0;

    // Decay
    // summed_kernel *= 0.92;
    summed_kernel *= 1.3;

    imageStore(DestTex, ivec2(gl_FragCoord.xy), vec4(summed_kernel, 0));
}
