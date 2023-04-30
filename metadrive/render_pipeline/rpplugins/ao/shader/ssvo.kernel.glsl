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

SSVO - Screen Space Volumetric Obscurance

This algorithm casts rays to a sphere arround the current point in view space,
and approximates the spheres volume by using line intetrals. The percentage
of the spheres volume is then used to compute AO.

*/



vec2 sphere_radius = GET_SETTING(ao, ssvo_sphere_radius) * pixel_size;
float max_depth_diff = GET_SETTING(ao, ssvo_max_distance) / kernel_scale;

float accum = 0.0;
float pixel_linz = get_linear_z_from_z(pixel_depth);

START_ITERATE_SEQUENCE(ao, ssvo_sequence, vec2 offset)

    offset = mix(offset, noise_vec.xy, 0.3);

    vec2 offc = offset * sphere_radius * 5.0 * kernel_scale;

    // Use paired samples, this enables us to hide depth buffer discontinuities
    vec2 offcoord_a = texcoord + offc;
    vec2 offcoord_b = texcoord - offc;

    // Compute the sphere height at the sample location
    float sphere_height = sqrt(1 - dot(offset, offset));

    // Get the depth at the sample locations, also
    // make the depth linear, this enables us to compare them better
    float depth_linz_a = get_linear_depth_at(offcoord_a);
    float depth_linz_b = get_linear_depth_at(offcoord_b);

    // Clamp both differences to the maximum depth difference
    float diff_a = (pixel_linz - depth_linz_a) / max_depth_diff;
    float diff_b = (pixel_linz - depth_linz_b) / max_depth_diff;

    // Compute the line integrals of boths, this is simply the height of the
    // line divided by the sphere height. However, we need to substract the
    // sphere height once, since we didn't start at the sphere top, but at
    // the sphere mid (since we took the depth of point p which is the center
    // of the sphere).
    float volume_a = (sphere_height - diff_a) / (2.0 * sphere_height);
    float volume_b = (sphere_height - diff_b) / (2.0 * sphere_height);

    // Check if the volumes are valid
    bool valid_a = diff_a <= sphere_height && diff_a >= -sphere_height;
    bool valid_b = diff_b <= sphere_height && diff_b >= -sphere_height;

    // In case either the first or second sample is valid, we can weight them
    if (valid_a || valid_b) {

        // Because we use paired samples, we can easily account for discontinuities:
        // If a is invalid, we can take the inverse of b as value for a, and vice-versa.
        // This works out quite well, even if not mathematically correct.
        accum += valid_a ? volume_a : 1 - volume_b;
        accum += valid_b ? volume_b : 1 - volume_a;

    // In case both samples are invalid, theres nothing we can do. Just increase
    // the integral in this case.
    } else {
        accum += 1.0;
    }

END_ITERATE_SEQUENCE();

NORMALIZE_SEQUENCE(ao, ssvo_sequence, accum);


result = accum;
