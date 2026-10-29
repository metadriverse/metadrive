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

layout(local_size_x = 10, local_size_y = 4, local_size_z = 1) in;

uniform writeonly image2D RESTRICT DestTex;
uniform samplerBuffer ExposureTex;

void main() {

    // TODO: Might make this an input
    const ivec2 widget_size = ivec2(140, 20);
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    // Store the current pixels color
    vec4 color = vec4(0, 0, 0, 0);
    vec4 border_color = vec4(0.9, 0.9, 0.9, 1.0);

    // Border
    const int border_size = 1;
    if (coord.x < border_size || coord.x >= widget_size.x - border_size ||
        coord.y < border_size || coord.y >= widget_size.y - border_size) {
        color += border_color * (1 - color.w);
    }

    // Fetch exposure settings
    float min_exp = GET_SETTING(color_correction, min_exposure_value);
    float max_exp = GET_SETTING(color_correction, max_exposure_value);

    // Fetch current exposure
    float curr_exposure = texelFetch(ExposureTex, 0).x;

    // Slider
    float slider_pos = saturate((curr_exposure - min_exp) / (max_exp - min_exp));

    // Make visualization logarithmic
    // slider_pos = make_logarithmic(slider_pos, factor);

    const int slider_w = 4;
    int slider_pos_int = int(slider_pos * float(widget_size.x - 2 * border_size)) + border_size;

    if (coord.x > slider_pos_int - slider_w && coord.x < slider_pos_int + slider_w) {
        // Don't draw the slider over the border
        color += vec4(50, 255, 50, 255.0) / 255.0 * (1 - color.w);
    }

    imageStore(DestTex, coord, color);
}
