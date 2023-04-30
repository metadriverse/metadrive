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

uniform sampler3D IESDatasetTex;

// The ies profiles give us candela values, which we can directly convert to lumens
const float IES_SCALE = 1.0 /* lumens */;

// Computes the IES factor from a given light vector and profile.
// A profile < 0 means no light profile.
float get_ies_factor(vec3 light_vector, int profile) {
    if (profile < 0)
        return 1.0;
    light_vector = normalize(light_vector);
    float horiz_angle = acos(light_vector.z) / M_PI;
    float vert_angle = fma(atan(light_vector.x, light_vector.y), 1.0 / TWO_PI, 0.5);
    float profile_coord = (profile+0.5) / MAX_IES_PROFILES;
    float data = textureLod(IESDatasetTex, vec3(horiz_angle, vert_angle, profile_coord), 0).x;
    return data * IES_SCALE;
}

// Computes the IES factor from a given spherical coordinate and profile.
// theta and phi should range from 0 to 1.
float get_ies_factor(int profile, float theta, float phi) {
    if (profile < 0)
        return 1.0;
    float profile_coord = (profile+0.5) / MAX_IES_PROFILES;
    float data = textureLod(IESDatasetTex, vec3(theta * 0.5 + 0.5, phi, profile_coord), 0).x;
    return data * IES_SCALE;
}
