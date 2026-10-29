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




const float sample_radius = GET_SETTING(ao, hbao_sample_radius);
const int num_angles = GET_SETTING(ao, hbao_ray_count);
const int num_ray_steps = GET_SETTING(ao, hbao_ray_steps);
const float tangent_bias = GET_SETTING(ao, hbao_tangent_bias);
const float max_sample_distance = GET_SETTING(ao, hbao_max_distance) * 0.3;

float accum = 0.0;

for (int i = 0; i < num_angles; ++i) {
    float angle = (i + 2 * noise_vec.x) / float(num_angles) * TWO_PI;

    vec2 sample_dir = vec2(cos(angle), sin(angle));

    // Find the tangent andle
    float tangent_angle = acos(dot(vec3(sample_dir, 0), pixel_view_normal)) - 0.5 * M_PI
                        + tangent_bias;

    // Assume the horizon angle is the same as the tangent angle at the beginning
    // of the ray
    float horizon_angle = tangent_angle;

    vec3 last_diff = vec3(0);

    // Raymarch in the sample direction
    for (int k = 0; k < num_ray_steps; ++k) {

        // Get new texture coordinate
        vec2 texc = texcoord +
            sample_dir * (k + 2.0 + 2 * noise_vec.y) /
                num_ray_steps * pixel_size * sample_radius * kernel_scale * 0.3;

        // Fetch view pos at that position and compare it
        vec3 view_pos = get_view_pos_at(texc);
        vec3 diff = view_pos - pixel_view_pos;

        if(length(diff) < max_sample_distance) {

            // Compute actual angle
            float sample_angle = atan(diff.z / length(diff.xy));

            // Correct horizon angle
            horizon_angle = max(horizon_angle, sample_angle);
            last_diff = diff;
        }
    }

    // Now that we know the average horizon angle, add it to the result
    // For that we simply take the angle-difference
    float occlusion = saturate(sin(horizon_angle) - sin(tangent_angle));
    occlusion *= 1.0 / (1 + length(last_diff));
    accum += occlusion;
}

// Normalize samples
accum /= num_angles;
result = 1 - accum;
