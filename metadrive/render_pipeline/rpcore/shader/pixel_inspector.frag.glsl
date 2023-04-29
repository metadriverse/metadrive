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

in vec2 texcoord;
out vec4 color;
uniform sampler2D SceneTex;
uniform vec2 mousePos;

void main() {
    int border = 3;
    int zoom = 5;

    ivec2 int_coord = ivec2(texcoord * vec2(400, 300));
    if (int_coord.x < border || int_coord.y < border ||
        int_coord.x >= 400 - border || int_coord.y >= 300 - border) {
        color = vec4(0.05, 0.05, 0.05, 1);
        return;
    }

    int_coord = (int_coord) / zoom - (ivec2(200, 150)) / zoom + ivec2(mousePos);
    color = texelFetch(SceneTex, int_coord, 0);
    color.w = 1.0;
}
