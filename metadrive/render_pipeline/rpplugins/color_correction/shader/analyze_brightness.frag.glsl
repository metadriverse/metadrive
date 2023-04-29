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

#define USE_TIME_OF_DAY 1
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/tonemapping.inc.glsl"

layout(r16f) uniform imageBuffer RESTRICT ExposureStorage;
uniform sampler2D DownscaledTex;

void main() {

    // Manually do the last downscale step
    ivec2 texsize = textureSize(DownscaledTex, 0).xy;
    float avg_luminance = 0.0;
    for (int x = 0; x < texsize.x; ++x) {
        for (int y = 0; y < texsize.y; ++y) {
            avg_luminance += texelFetch(DownscaledTex, ivec2(x, y), 0).x;
        }
    }

    avg_luminance /= float(texsize.x * texsize.y);
    avg_luminance = avg_luminance / (1 - avg_luminance);

    #if 0
        float exposure_val = computeEV100FromAvgLuminance(avg_luminance);
        float exposure = convertEV100ToExposure(exposure_val);
    #else
        // Same as a above - just factored out.
        float exposure = 0.1041666 * (1.0 / avg_luminance);
    #endif

    float min_ev = GET_SETTING(color_correction, min_exposure_value);
    float max_ev = GET_SETTING(color_correction, max_exposure_value);
    float exposure_scale = GET_SETTING(color_correction, exposure_scale);

    // XXX: Without a multiplier of two, the image gets way too dark - not sure
    // why, but most likely an issue in the exposure calculation. However,
    // this is not physically correct.
    exposure *= exposure_scale * 2.0;

    // Clamp to min and max exposure value
    exposure = clamp(exposure, min_ev, max_ev);

    // Transition between the last and current value smoothly
    float curr_exposure = imageLoad(ExposureStorage, 0).x;
    float adaption_rate = GET_SETTING(color_correction, brightness_adaption_rate);

    if (curr_exposure < exposure) {
        adaption_rate = GET_SETTING(color_correction, darkness_adaption_rate);
    }

    const float adaption_speed = 0.8;
    adaption_rate *= adaption_speed;

    float adjustment = saturate(MainSceneData.smooth_frame_delta * adaption_rate);
    float new_luminance = mix(curr_exposure, exposure, adjustment);
    new_luminance = clamp(new_luminance, 0.0, 1e4);
    imageStore(ExposureStorage, 0, vec4(new_luminance));
}
