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

// Main configuration file, included by all shaders, provides generic defines
// and functions.

#pragma include "/$$rptemp/$$pipeline_shader_config.inc.glsl"

// Optionally unroll *all* loops, this might be faster, but might also be
// slower. Right now, every shader specifies on his own if he wants to unroll
// his loops or not.
// #pragma optionNV (unroll all)

// Nvidia specific optimizations
#ifdef IS_NVIDIA
#pragma optionNV (fastmath on)
#pragma optionNV (ifcvt none)
#pragma optionNV (inline all)
#pragma optionNV (strict on)
#endif

#pragma include "/$$rptemp/$$main_scene_data.inc.glsl"

#ifdef USE_TIME_OF_DAY
#pragma include "/$$rptemp/$$daytime_config.inc.glsl"
#endif

// Screen size macro
#define WINDOW_WIDTH MainSceneData.screen_size.x
#define WINDOW_HEIGHT MainSceneData.screen_size.y
#define NATIVE_WINDOW_WIDTH MainSceneData.native_screen_size.x
#define NATIVE_WINDOW_HEIGHT MainSceneData.native_screen_size.y
#define SCREEN_SIZE vec2(WINDOW_WIDTH, WINDOW_HEIGHT)
#define SCREEN_SIZE_INT ivec2(WINDOW_WIDTH, WINDOW_HEIGHT)
#define ASPECT_RATIO float(float(WINDOW_HEIGHT) / float(WINDOW_WIDTH))
#define NATIVE_SCREEN_SIZE vec2(NATIVE_WINDOW_WIDTH, NATIVE_WINDOW_HEIGHT)

// Plugin functions
#define HAVE_PLUGIN(PLUGIN_NAME) (HAVE_PLUGIN_ ## PLUGIN_NAME)
#define GET_SETTING(PLUGIN_NAME, SETTING_NAME) (PLUGIN_NAME ## _ ## SETTING_NAME)
#define GET_ENUM_VALUE(PLUGIN_NAME, SETTING_NAME, ENUM_KEY) (enum_ ##PLUGIN_NAME ## _ ## SETTING_NAME ## _ ## ENUM_KEY)
#define ENUM_V_ACTIVE(PLUGIN_NAME, SETTING_NAME, ENUM_KEY) (HAVE_PLUGIN(PLUGIN_NAME) && GET_SETTING(PLUGIN_NAME, SETTING_NAME) && GET_SETTING(PLUGIN_NAME, SETTING_NAME) == GET_ENUM_VALUE(PLUGIN_NAME, SETTING_NAME, ENUM_KEY))

// Render mode functions
#define DEBUG_MODE ANY_DEBUG_MODE
#define MODE_ACTIVE(MODE_ID) (DEBUG_MODE && (_RM_ ## MODE_ID))
#define SPECIAL_MODE_ACTIVE(MODE_ID) (_RM_ ## MODE_ID)

// Branch modes for translucency.
// This serves for the purpose to be enabled or disabled easily.
// Right now, it seems its faster not to branch. It heavily depends on
// the amount of translucent materials on the screen.
#if 0
#define BRANCH_TRANSLUCENCY(m) if (m.translucency > 0.01) {
#define END_BRANCH_TRANSLUCENCY() }
#else
#define BRANCH_TRANSLUCENCY(m)
#define END_BRANCH_TRANSLUCENCY()
#endif

// Disable translucency?
#if 1
#undef BRANCH_TRANSLUCENCY
#undef END_BRANCH_TRANSLUCENCY
#define BRANCH_TRANSLUCENCY(m) if (false) {
#define END_BRANCH_TRANSLUCENCY() }
#endif

// Restrict qualifier, only on AMD cards, Nvidia can't handle it. See:
// https://devtalk.nvidia.com/default/topic/546817/restrict-keyword-crashes-glsl-compiler/
// Also, intel seems to expect the keyword (correctly) *before* the image specifier,
// in contrast to AMD, so we disable it on intel gpus, too.
#if IS_AMD
    #define RESTRICT restrict
#else
    #define RESTRICT
#endif

// TODO:
#define SUPPORT_PCF 1

// Controls the roughness of the clearcoat layer
#define CLEARCOAT_ROUGHNESS 0.001
#define CLEARCOAT_SPECULAR 0.16
#define CLEARCOAT_IOR 1.51

// Controls the brightness of the fallback cubemap
#if REFERENCE_MODE
    #define DEFAULT_ENVMAP_BRIGHTNESS 1.0
#else
    #if HAVE_PLUGIN(color_correction)
        #define DEFAULT_ENVMAP_BRIGHTNESS 1.0
    #else
        #define DEFAULT_ENVMAP_BRIGHTNESS 1.0
    #endif
#endif

// Minimum roughness, avoids infinitely bright highlights
#define MINIMUM_ROUGHNESS 0.01

// Controls at which point the sun is below the horizon and does not have any
// influence anymore
#define SUN_VECTOR_HORIZON 0.0

// Whether to use a completely white environment, only used in reference mode
#define USE_WHITE_ENVIRONMENT 0


#pragma include "includes/common_functions.inc.glsl"
