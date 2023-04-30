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

#pragma include "scattering_common.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform writeonly image3D RESTRICT destDeltaSR;
uniform writeonly image3D RESTRICT destDeltaSM;

void integrand(float r, float mu, float muS, float nu, float t, out vec3 ray, out vec3 mie) {
    ray = vec3(0.0);
    mie = vec3(0.0);
    float ri = sqrt(r * r + t * t + 2.0 * r * mu * t);
    float muSi = (nu * t + muS * r) / ri;
    ri = max(Rg, ri);
    if (muSi >= -sqrt(1.0 - Rg * Rg / (ri * ri))) {
        vec3 ti = transmittance(r, mu, t) * transmittance(ri, muSi);
        ray = exp(-(ri - Rg) / HR) * ti;
        mie = exp(-(ri - Rg) / HM) * ti;
    }
}


void inscatter(float r, float mu, float muS, float nu, out vec3 ray, out vec3 mie) {
    ray = vec3(0.0);
    mie = vec3(0.0);
    float dx = limit(r, mu) / float(INSCATTER_INTEGRAL_SAMPLES);
    float xi = 0.0;
    vec3 rayi;
    vec3 miei;
    integrand(r, mu, muS, nu, 0.0, rayi, miei);
    for (int i = 1; i <= INSCATTER_INTEGRAL_SAMPLES; ++i) {
        float xj = float(i) * dx;
        vec3 rayj;
        vec3 miej;
        integrand(r, mu, muS, nu, xj, rayj, miej);
        ray += (rayi + rayj) / 2.0 * dx;
        mie += (miei + miej) / 2.0 * dx;
        xi = xj;
        rayi = rayj;
        miei = miej;
    }
    ray *= betaR;
    mie *= betaMSca;
}


void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    int layer = int(gl_GlobalInvocationID.z);

    float r;
    vec4 dhdH;
    get_r_dhdh(layer, r, dhdH);

    vec3 ray;
    vec3 mie;
    float mu, muS, nu;
    getMuMuSNu(r, dhdH, mu, muS, nu);
    inscatter(r, mu, muS, nu, ray, mie);
    // store separately Rayleigh and Mie contributions, WITHOUT the phase function factor
    // (cf "Angular precision")

    imageStore(destDeltaSR, ivec3(coord, layer), vec4(ray, SCAT_DEBUG_ALPHA));
    imageStore(destDeltaSM, ivec3(coord, layer), vec4(mie, SCAT_DEBUG_ALPHA));

}
