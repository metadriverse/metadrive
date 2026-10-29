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

#pragma include "includes/color_spaces.inc.glsl"

/*

Tonemapping Operators, some are from:
http://filmicgames.com/archives/75

A good sample using them can be found here:
https://www.shadertoy.com/view/lslGzl

*/


const float exposure_adjustment = 2.0;


// No tonemapping
vec3 tonemap_none(vec3 color) {
    color *= exposure_adjustment;
    return color;
}

// Different versions of the reinhard tonemap operators
vec3 tonemap_reinhard(vec3 color)
{
    color *= exposure_adjustment;

    #if ENUM_V_ACTIVE(color_correction, reinhard_version, rgb)

        // Simple reinhard operator in rgb space
        vec3 rgb = color / (1.0 + color);

    #elif ENUM_V_ACTIVE(color_correction, reinhard_version, luminance)

        // Simple reinhard operator based on luminance
        float luminance = get_luminance(color);
        vec3 rgb = color / (1.0 + luminance);

    #elif ENUM_V_ACTIVE(color_correction, reinhard_version, white_preserve)

        // Reinhard operator based on luminance, but white preserving
        float white = 5.0;
        float luminance = get_luminance(color);
        vec3 rgb = color * (1.0 + luminance / (white * white)) / (1.0 + luminance);

    #elif ENUM_V_ACTIVE(color_correction, reinhard_version, luminosity)

        // Reinhard operator in the CIE-Yxy color space
        vec3 xyY = rgb_to_xyY(color);

        // This saturates too much?
        xyY.z = xyY.z / (1.0 + xyY.z);

        // White preserving approach, doesn't work out well, too
        // xyY.z = xyY.z * (1.0 + xyY.z * (white*white)) / (1.0 + xyY.z);
        vec3 rgb = xyY_to_rgb(xyY);

    #else
        #error Unkown reinhard operator version!
    #endif

    return rgb;
}


// Optimized version of the Haarm-Peter Duiker’s curve
vec3 tonemap_optimized(vec3 color)
{
    color *= exposure_adjustment;
    vec3 x = max(vec3(0.0), color - 0.004);
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

// Exponential tonemapping
vec3 tonemap_exponential(vec3 color) {
    color *= exposure_adjustment;
    color = 1.0 - exp(-GET_SETTING(color_correction, exponential_factor) * color);
    return color;
}

// Alternative exponential tonemapping
vec3 tonemap_exponential_2(vec3 color) {
    color *= exposure_adjustment;
    color = exp(-1.0 / (2.72 * color + 0.15));
    return color;
}


// Uncharted 2 tonemapping formula
// See: http://de.slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting
vec3 uncharted_2_tonemap_formula(vec3 x)
{
    const float A = GET_SETTING(color_correction, uc2t_shoulder_strength);
    const float B = GET_SETTING(color_correction, uc2t_linear_strength);
    const float C = GET_SETTING(color_correction, uc2t_linear_angle);
    const float D = GET_SETTING(color_correction, uc2t_toe_strength);
    const float E = GET_SETTING(color_correction, uc2t_toe_numerator);
    const float F = GET_SETTING(color_correction, uc2t_toe_denumerator);
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// Uncharted 2 tonemap operator
vec3 tonemap_uncharted2(vec3 color)
{
    const float white_base = GET_SETTING(color_correction, uc2t_reference_white);
    color *= exposure_adjustment;
    const float exposure_bias = 2.0;
    vec3 curr = uncharted_2_tonemap_formula(exposure_bias * color);
    vec3 white = uncharted_2_tonemap_formula(vec3(white_base));
    return curr / white;
}

// Tonemapping selector
vec3 do_tonemapping(vec3 color) {

    // Clamp extreme values of the color
    color = clamp(color, 1e-5, 1000.0);

    // Select tonemapping operator
    #if ENUM_V_ACTIVE(color_correction, tonemap_operator, none)
        color = tonemap_none(color);
    #elif ENUM_V_ACTIVE(color_correction, tonemap_operator, optimized)
        color = tonemap_optimized(color);
        return color;
    #elif ENUM_V_ACTIVE(color_correction, tonemap_operator, reinhard)
        color = tonemap_reinhard(color);
    #elif ENUM_V_ACTIVE(color_correction, tonemap_operator, uncharted2)
        color = tonemap_uncharted2(color);
    #elif ENUM_V_ACTIVE(color_correction, tonemap_operator, exponential)
        color = tonemap_exponential(color);
    #elif ENUM_V_ACTIVE(color_correction, tonemap_operator, exponential2)
        color = tonemap_exponential_2(color);
    #else
        #error Unkown tonemapping operator!
    #endif

    return rgb_to_srgb(color);
}


// From:
// http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf

float computeEV100(float aperture, float shutter_time, float ISO) {
    // EV number is defined as:
    // 2^ EV_s = N^2 / t and EV_s = EV_100 + log2 (S /100)
    // This gives
    // EV_s = log2 (N^2 / t)
    // EV_100 + log2 (S /100) = log2 (N^2 / t)
    // EV_100 = log2 (N^2 / t) - log2 (S /100)
    // EV_100 = log2 (N^2 / t . 100 / S)
    return log2((aperture * aperture) / shutter_time * 100 / ISO);
}

float computeEV100FromAvgLuminance(float avg_luminance) {
    // We later use the middle gray at 12.7% in order to have
    // a middle gray at 18% with a sqrt (2) room for specular highlights
    // But here we deal with the spot meter measuring the middle gray
    // which is fixed at 12.5 for matching standard camera
    // constructor settings (i.e. calibration constant K = 12.5)
    return log2(avg_luminance * 100.0 / 12.5);
}

float convertEV100ToExposure(float EV100) {
    // Compute the maximum luminance possible with H_sbs sensitivity
    // maxLum = 78 / ( S * q ) * N^2 / t
    // = 78 / ( S * q ) * 2^ EV_100
    // = 78 / (100 * 0.65) * 2^ EV_100
    // = 1.2 * 2^ EV
    float max_luminance = 1.2 * pow(2.0, EV100);
    return 1.0 / max_luminance;
}
