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

#version 430

#pragma include "render_pipeline_base.inc.glsl"

#define FXAA_PC 1
#define FXAA_GLSL_130 1

// Define quality level
#if ENUM_V_ACTIVE(fxaa, quality, low)
    #define FXAA_QUALITY_PRESET 12
#elif ENUM_V_ACTIVE(fxaa, quality, medium)
    #define FXAA_QUALITY_PRESET 20
#elif ENUM_V_ACTIVE(fxaa, quality, high)
    #define FXAA_QUALITY_PRESET 29
#elif ENUM_V_ACTIVE(fxaa, quality, ultra)
    #define FXAA_QUALITY_PRESET 39
#else
    #error Unkown fxaa quality
#endif

#pragma include "FXAA.inc.glsl"

uniform sampler2D SourceTex;
out vec4 color;

void main() {

    vec2 texcoord = get_texcoord();
    vec2 pixel_size = vec2(1.0) / SCREEN_SIZE;

    vec2 upper_left = texcoord + vec2(-0.5, 0.5) * pixel_size;
    vec2 lower_right = texcoord + vec2(0.5, -0.5) * pixel_size;

    const float sharp = 0.5; // unused in the pc version
    const float edge_threshold = GET_SETTING(fxaa, edge_threshold);
    const float edge_min_threshold = 0.1 * GET_SETTING(fxaa, min_threshold);
    const float edge_sharpness = 8.0; // power of two, unused in the pc  version
    vec4 size_twice = vec4(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT);

    vec4 result = FxaaPixelShader(
        texcoord, // pos
        vec4(upper_left, lower_right), // fxaaConsolePosPos
        SourceTex, // tex
        SourceTex, // fxaaConsole360TexExpBiasNegOne
        SourceTex, // fxaaConsole360TexExpBiasNegTwo
        vec2(1.0) / SCREEN_SIZE, // fxaaQualityRcpFrame
        vec4(-sharp, -sharp, sharp, sharp) / size_twice, // fxaaConsoleRcpFrameOpt
        vec4(-2, -2, 2, 2) / size_twice, // fxaaConsoleRcpFrameOpt2
        vec4(8, 8, -4, -4) / size_twice, // fxaaConsole360RcpFrameOpt2
        GET_SETTING(fxaa, subpixel_quality), // fxaaQualitySubpix
        edge_threshold, // fxaaQualityEdgeThreshold
        edge_min_threshold, // fxaaQualityEdgeThresholdMin
        edge_sharpness, // fxaaConsoleEdgeSharpness
        edge_threshold, // fxaaConsoleEdgeThreshold
        edge_min_threshold, // fxaaConsoleEdgeThresholdMin
        vec4(1.0, -1.0, 0.25, -0.25)  // fxaaConsole360ConstDir
    );

    color = result;
}
