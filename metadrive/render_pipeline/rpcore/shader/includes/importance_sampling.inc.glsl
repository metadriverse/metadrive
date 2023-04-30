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

// From:
// http://www.trentreed.net/blog/physically-based-shading-and-image-based-lighting/
vec2 hammersley(uint i, uint N)
{
    return vec2(float(i) / float(N), float(bitfieldReverse(i)) * 2.3283064365386963e-10);
}

// From:
// http://www.gamedev.net/topic/655431-ibl-problem-with-consistency-using-ggx-anisotropy/
vec3 importance_sample_ggx(vec2 Xi, float alpha)
{
    // alpha is already squared roughness
    float r_square = alpha * alpha;
    float phi = TWO_PI * Xi.x;
    float cos_theta_sq = (1 - Xi.y) / max(1e-3, 1 + (r_square * r_square - 1) * Xi.y);
    float cos_theta = sqrt(cos_theta_sq);
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta_sq));
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

vec3 importance_sample_lambert(vec2 Xi)
{
    float phi = TWO_PI * Xi.x;
    float cos_theta = sqrt(Xi.y);
    float sin_theta = sqrt(1 - Xi.y);
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}
