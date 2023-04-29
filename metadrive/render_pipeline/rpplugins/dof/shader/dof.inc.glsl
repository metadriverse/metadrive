/**
 *
 * RenderPipeline
 *
 * Copyright (c) 2014-2016 tobspr <toref_val.springer1@gmail.com>
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

const int tile_size = 32;

#define DOF_DEPTH_SCALE_FOREGROUND 250.0
#define DOF_SINGLE_PIXEL_RADIUS 1.0
// #define DOF_SINGLE_PIXEL_RADIUS 1.0

vec2 compare_depth(float depth, float tile_max_depth)
{
    float d = DOF_DEPTH_SCALE_FOREGROUND * (tile_max_depth - depth);
    d -= 0.7;
    vec2 depth_cmp = vec2(0);
    depth_cmp.x = smoothstep(0.0, 1.0, saturate(d));
    // depth_cmp.x = step(0.01, d);
    depth_cmp.y = 1.0 - depth_cmp.x;
    return depth_cmp;
}

float sample_alpha(float sampleCoc)
{
    return min(1.0 / (M_PI * sampleCoc * sampleCoc), M_PI *
        DOF_SINGLE_PIXEL_RADIUS * DOF_SINGLE_PIXEL_RADIUS);
}
