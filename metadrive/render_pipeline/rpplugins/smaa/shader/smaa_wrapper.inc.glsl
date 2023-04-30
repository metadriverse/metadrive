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

// This file just sets a few defines and then includes the SMAA header

#define SMAA_GLSL_4
#define SMAA_RT_METRICS vec4(1.0 / WINDOW_WIDTH, 1.0 / WINDOW_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT)


// Get seleted SMAA quality
#if ENUM_V_ACTIVE(smaa, smaa_quality, low)
    #define SMAA_PRESET_LOW
#elif ENUM_V_ACTIVE(smaa, smaa_quality, medium)
    #define SMAA_PRESET_MEDIUM
#elif ENUM_V_ACTIVE(smaa, smaa_quality, high)
    #define SMAA_PRESET_HIGH
#elif ENUM_V_ACTIVE(smaa, smaa_quality, ultra)
    #define SMAA_PRESET_ULTRA
#else
    #error Unkown smaa quality value!
#endif


// Include both Pixel and Vertex shader, because we do the vertex shader logic
// in the pixel shader.
#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 1
#define SMAA_DECODE_VELOCITY(_sample) error, custom resolve pass

vec3 SMAA_GET_COLOR(vec3 _color) {
    return _color;
}

vec4 SMAA_GET_COLOR(vec4 _color) {
    return _color;
}

// Optionally enable smaa predication
// #define SMAA_PREDICATION 1
// #define SMAA_PREDICATION_SCALE 3.0
// #define SMAA_PREDICATION_THRESHOLD 0.00001

// SMAA defines its own saturate, make sure we don't run into conflicts
#ifdef saturate
    #undef saturate
#endif

// Include the actual smaa header
#pragma include "SMAA.inc.glsl"
