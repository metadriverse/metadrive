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

#pragma include "includes/light_data.inc.glsl"

#define LIGHT_CLS_INVALID -1
#define LIGHT_CLS_SPOT_NOSHADOW 0
#define LIGHT_CLS_POINT_NOSHADOW 1
#define LIGHT_CLS_SPOT_SHADOW 2
#define LIGHT_CLS_POINT_SHADOW 3

#define LIGHT_CLS_COUNT 4

#if LIGHT_CLS_COUNT != LC_LIGHT_CLASS_COUNT
    #error GLSL and Python lighting system class count do not match up!
#endif

int classify_light(int light_type, bool casts_shadows) {
    switch (light_type) {
        case LT_SPOT_LIGHT:
            return casts_shadows ? LIGHT_CLS_SPOT_SHADOW : LIGHT_CLS_SPOT_NOSHADOW;
        case LT_POINT_LIGHT:
            return casts_shadows ? LIGHT_CLS_POINT_SHADOW : LIGHT_CLS_POINT_NOSHADOW;

    };
    return LIGHT_CLS_INVALID;
}
