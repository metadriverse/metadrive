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


/*

CryEngine SSAO - Screen Space Ambient Occlusion

Samples the depth buffer in a hemisphere arround the current pixel to approximate
AO.

*/

const float sample_radius = GET_SETTING(ao, ssao_sample_radius);
const float bias = GET_SETTING(ao, ssao_bias) * 0.5 / kernel_scale;
float max_range = GET_SETTING(ao, ssao_max_distance);

float sample_offset = sample_radius * pixel_size.x * 30.0;
float range_accum = 0.0;
float accum = 0.0;

// Make sure we have nice sampling coherency by scaling the kernel
sample_offset /= 0.8 * kernel_scale;

START_ITERATE_SEQUENCE(ao, ssao_sequence, vec3 offset)

    offset = mix(offset, noise_vec, 0.5);
    offset *= 0.6;

    // Flip offset in case it faces away from the normal
    offset = face_forward(offset, pixel_view_normal);

    // Compute offset position in view space
    vec3 offset_pos = pixel_view_pos + offset * sample_offset;

    // Project offset position to screen space
    vec3 projected = view_to_screen(offset_pos);

    // Fetch the expected depth
    float sample_depth = get_linear_depth_at(projected.xy);

    // Linearize both depths
    float linz_a = get_linear_z_from_z(projected.z);
    float linz_b = sample_depth;

    // Compare both depths by distance to find the AO factor
    float modifier = step(distance(linz_a, linz_b), max_range * 0.2);
    range_accum += fma(modifier, 0.5, 0.5);
    modifier *= step(linz_b + bias, linz_a);
    accum += modifier;

END_ITERATE_SEQUENCE()


// normalize samples
accum /= max(0.1, range_accum);

// Renormalize to match with the other techniques
accum *= 0.4;
result = 1 - accum;
