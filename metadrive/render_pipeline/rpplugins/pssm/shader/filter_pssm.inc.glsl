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


vec2 get_split_coord(vec2 local_coord, int split_index) {
    local_coord.x = (local_coord.x + split_index) / float(GET_SETTING(pssm, split_count));
    return local_coord;
}

float get_shadow(vec2 coord, float refz) {
    #if GET_SETTING(pssm, use_pcf)
        return textureLod(PSSMShadowAtlasPCF, vec3(coord, refz), 0);
    #else
        float depth_sample = textureLod(PSSMShadowAtlas, coord, 0).x;
        return step(refz, depth_sample);
    #endif
}

float get_fixed_bias(int split) {
    return GET_SETTING(pssm, fixed_bias) * 0.001 * (1 + 1.5 * split);
}

vec3 get_pssm_split_biased_position(vec3 pos, vec3 normal, vec3 sun_vector, int split) {

    // Compute the biased position based on the normal and slope scaled
    // bias.
    float slope_bias = GET_SETTING(pssm, slope_bias) * 0.1 * (1 + 0.2 * split);
    const float normal_bias = GET_SETTING(pssm, normal_bias) * 0.1;
    
    // Compute the biased position based on the normal and slope scaled
    // bias.
    vec3 biased_pos = get_biased_position(pos, slope_bias, normal_bias, normal, sun_vector);

    return biased_pos;
}


