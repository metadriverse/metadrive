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

/*
From:
http://www.iryoku.com/translucency/
*/

const float skin_travel_weight = 1.0;

vec3 skin_transmittance(float distance_through_medium) {
    float s = distance_through_medium * skin_travel_weight;
    float s_sq = -s * s;
    vec3 t = vec3(0.233, 0.455, 0.649) * exp(s_sq / 0.0064) +
                vec3(0.1, 0.336, 0.344) * exp(s_sq / 0.0484) +
                vec3(0.118, 0.198, 0.0) * exp(s_sq / 0.187) +
                vec3(0.113, 0.007, 0.007) * exp(s_sq / 0.567) +
                vec3(0.358, 0.004, 0.0) * exp(s_sq / 1.99) +
                vec3(0.078, 0.0, 0.0) * exp(s_sq / 7.41);
    return t;
}
