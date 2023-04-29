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

const int tile_size = GET_SETTING(motion_blur, tile_size);
float blur_factor = GET_SETTING(motion_blur, blur_factor) * 0.5;
float max_velocity_len = GET_SETTING(motion_blur, max_blur_radius) *
                                tile_size / WINDOW_WIDTH * 0.2;
const vec2 soft_depth_factor = vec2(10.0);

vec2 adjust_velocity(vec2 velocity) {
    velocity *= blur_factor;

    // Make sure the velocity does not exceed the maximum length
    float vel_len = length(velocity);
    if (vel_len > max_velocity_len) {
        velocity *= max_velocity_len / vel_len;
    }
    return velocity;
}


vec2 soft_depth_cmp(vec2 z0, vec2 z1)
{
    return saturate(fma(z0, soft_depth_factor, vec2(1.0)) - z1 * soft_depth_factor);
}

vec4 batch_cmp(float len_xy_sq, vec2 velocities)
{
    return saturate(vec4((1 - len_xy_sq / (velocities.xyxy)) + vec4(0.0, 0.0, 0.95, 0.95)));
}
