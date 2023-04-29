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

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/material.inc.glsl"
#pragma include "includes/color_spaces.inc.glsl"


/*

 BRDFs from:
 http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf

 Also from:
 http://www.trentreed.net/blog/physically-based-shading-and-image-based-lighting/

 Also inspired by the implementation in Disneys BRDF Explorer:
 http://www.disneyanimation.com/technology/brdf.html

*/


// Lambert BRDF
float brdf_lambert() {
    return ONE_BY_PI;
}

// Schlicks approximation to fresnel
vec3 brdf_schlick_fresnel(vec3 specular, float f90, float VxH)
{
    // Fast pow, proposed in
    // http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
    return mix(specular, vec3(f90), pow(2, (-5.55473 * VxH - 6.98316) * VxH));

    // Regular pow
    // return mix(specular, vec3(f90), pow( 1.0 - VxH, 5.0));
}

vec3 brdf_schlick_fresnel(vec3 specular, float VxH)
{
    return brdf_schlick_fresnel(specular, 1.0, VxH);
}

// BRDF Proposed by Burley
float brdf_disney_diffuse(float NxV, float NxL, float LxH, float roughness) {

    // In case of squared roughness:
    float lin_roughness = roughness;
    float energy_bias = 0.5 * lin_roughness;
    float energy_factor = mix(1.0, 1.0 / 1.51, lin_roughness);
    float fd90 = energy_bias + 2.0 * LxH * LxH * lin_roughness;
    vec3 f0 = vec3(1);
    float light_scatter = brdf_schlick_fresnel(f0, fd90, NxL).x;
    float view_scatter = brdf_schlick_fresnel(f0, fd90, NxV).x;
    return light_scatter * view_scatter * energy_factor / M_PI;
}

/* Distribution functions */
float brdf_distribution_blinn_phong(float NxH, float roughness) {
    float r_sq = roughness * roughness;
    float inv_r = 1.0 / r_sq;
    return inv_r * ONE_BY_PI * pow(NxH, fma(inv_r, 2.0, -2.0));
}

float brdf_distribution_beckmann(float NxH, float roughness) {
    float r_sq = roughness * roughness;
    float nxh_sq = NxH * NxH;
    return exp((nxh_sq - 1.0) / (r_sq * nxh_sq)) / (M_PI * r_sq * nxh_sq * nxh_sq);
}

float brdf_distribution_ggx(float NxH, float alpha) {
    float r_square = alpha * alpha;
    float f = (NxH * r_square - NxH) * NxH + 1.0;
    return r_square / (f * f);
}

float brdf_distribution_exponential(float NxH, float roughness) {
    float r_sq = roughness * roughness;
    float d = exp(-acos(NxH) / roughness); // TODO: optimize
    return d * (1.0 + 4.0 * r_sq) / (2.0 * r_sq *
            (1.0 + exp(-(M_PI / (2.0 * roughness)))) * M_PI);
}

float brdf_distribution_gaussian(float NxH, float roughness) {
    float theta = acos(NxH); // TODO: optimize
    return exp(-theta * theta / (roughness * roughness));
}

float brdf_distribution_trowbridge_reitz(float NxH, float roughness) {
    float r_sq = roughness * roughness;
    float f = r_sq / (NxH * NxH * (r_sq - 1.0) + 1.0);
    return f * f * ONE_BY_PI / r_sq;
}


/* Visibility functions */
float brdf_visibility_implicit(float NxL, float NxV) {
    return NxL * NxV;
}

float brdf_visibility_neumann(float NxV, float NxL) {
    return NxL * NxV / max(1e-7, max(NxL, NxV));
}

float brdf_visibility_cook_torrance(float NxL, float NxV, float NxH, float VxH) {
    float nh_by_vh = 2.0 * NxH / VxH;
    float eq_nv = NxV * nh_by_vh;
    float eq_nl = NxL * nh_by_vh;
    return min(1.0, min(eq_nv, eq_nl));
}

float brdf_visibility_smith(float NxL, float NxV, float roughness) {
    float r_sq = roughness * roughness;
    float vis_v = NxL * (NxV * (1 - r_sq) + r_sq);
    float vis_l = NxV * (NxL * (1 - r_sq) + r_sq);
    return 0.5 / (vis_v + vis_l);
}

// Helper function for the schlick visibility
float _schlick_g(float NxL, float roughness_sq) {
    float k = sqrt(2.0 * roughness_sq * ONE_BY_PI);
    return NxL > 0.0 ? NxL / (NxL - k * NxL + k) : 0.0;
}

float brdf_visibility_schlick(float NxV, float NxL, float roughness) {
    float r_sq = roughness * roughness;
    return _schlick_g(NxL, r_sq) * _schlick_g(NxV, r_sq);
}


float brdf_visibility_smith_ggx(float NxL, float NxV, float roughness) {
    float alpha = roughness * roughness;
    float lambda_GGXV = NxL * sqrt((-NxV * alpha + NxV) * NxV + alpha);
    float lambda_GGXL = NxV * sqrt((-NxL * alpha + NxL) * NxL + alpha);
    return 0.5 / (lambda_GGXV + lambda_GGXL) * NxL;
}


/* Fresnel functions */

float ior_to_specular(float ior) {
    float f0 = (ior - AIR_IOR) / (ior + AIR_IOR);
    // Clamp between ior of 1 and 2.5
    return clamp(f0 * f0, 0.0, 0.18);
}

float brdf_fresnel_cook_torrance(float LxH, float roughness, float ior) {
    float g = sqrt(max(0, ior * ior + LxH * LxH - 1.0));
    float gpc = g + LxH;
    float gmc = g - LxH;
    return 0.5 * pow(gmc, 2) / pow(gpc, 2) * (1 + pow(LxH * gpc - 1, 2) /
        pow(LxH * gmc + 1, 2));
}

float brdf_fresnel_schlick_f0(float LxH, float roughness, float f0) {
    return f0 + (1 - f0) * pow(2, (-5.55473 * LxH - 6.98316) * LxH);
}

float brdf_fresnel_schlick(float LxH, float roughness, float ior) {
    float f0 = ior_to_specular(ior);
    return brdf_fresnel_schlick_f0(LxH, roughness, f0);
}

// Exact fresnel, can be slow.
float brdf_fresnel_exact(float cos_theta) {
    const float eta = 1.51;

    float scale = 1 / eta;
    float cos_theta_t_sqr = 1 - (1 - cos_theta * cos_theta) * (scale * scale);

    if (cos_theta_t_sqr <= 0.0)
        return 1.0;

    float cos_theta_i = abs(cos_theta);
    float cos_theta_t = sqrt(cos_theta_t_sqr);

    float Rs = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    float Rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

    return 0.5 * (Rs * Rs + Rp * Rp);
}

// Exact conductor fresnel, can be slow, also only single channel
float brdf_fresnel_conductor_exact(float cos_theta_i, float eta, float k) {
    /* Modified from "Optics" by K.D. Moeller, University Science Books, 1988 */
    float cosThetaI2 = cos_theta_i * cos_theta_i;
    float sinThetaI2 = 1 - cosThetaI2;
    float sinThetaI4 = sinThetaI2 * sinThetaI2;

    float temp1 = eta * eta - k * k - sinThetaI2;
    float a2pb2 = sqrt(temp1 * temp1 + 4 * k * k * eta * eta);
    float a = sqrt(0.5f * (a2pb2 + temp1));

    float term1 = a2pb2 + cosThetaI2;
    float term2 = 2 * a * cos_theta_i;

    float Rs2 = (term1 - term2) / (term1 + term2);

    float term3 = a2pb2 * cosThetaI2 + sinThetaI4;
    float term4 = term2 * sinThetaI2;

    float Rp2 = Rs2 * (term3 - term4) / (term3 + term4);
    return 0.5f * (Rp2 + Rs2);
}


// Approximation proposed in
// http://sirkan.iit.bme.hu/~szirmay/fresnel.pdf
vec3 brdf_fresnel_conductor_approx(float cos_theta, vec3 n, vec3 k) {
    vec3 k_sq = k * k;
    vec3 term0 = square(n - 1.0) + 4 * n * pow(1 - cos_theta, 5.0) + k_sq;
    vec3 term1 = square(n + 1.0) + k_sq;
    return term0 / term1;
}

// Diffuse BRDF
float brdf_diffuse(float NxV, float NxL, float LxH, float VxH, float roughness) {

    // Choose one:
    return brdf_lambert();
    // return brdf_disney_diffuse(NxV, NxL, LxH, roughness);
}


// Distribution
float brdf_distribution(float NxH, float roughness)
{
    NxH = max(1e-5, NxH);
    roughness = max(0.0019, roughness);

    // Choose one:
    // return brdf_distribution_blinn_phong(NxH, roughness);
    // return brdf_distribution_beckmann(NxH, roughness);
    // return brdf_distribution_exponential(NxH, roughness);
    // return brdf_distribution_gaussian(NxH, roughness);
    return brdf_distribution_trowbridge_reitz(NxH, roughness);
    // return brdf_distribution_ggx(NxH, roughness);
}

// Geometric Visibility
float brdf_visibility(float NxL, float NxV, float NxH, float VxH, float roughness) {

    // Choose one:
    // float vis = brdf_visibility_neumann(NxV, NxL);
    float vis = brdf_visibility_implicit(NxV, NxL);
    // float vis = brdf_visibility_smith_ggx(NxV, NxL, roughness);
    // float vis = brdf_visibility_schlick(NxV, NxL, roughness);
    // float vis = brdf_visibility_cook_torrance(NxL, NxV, NxH, VxH);
    // float vis = brdf_visibility_smith(NxL, NxV, roughness);


    return vis;
}

// Fresnel
float brdf_fresnel(float LxH, float roughness) {
    // Default index of refraction
    // TODO: Maybe make this configurable?
    const float ior = 1.2;

    return brdf_fresnel_schlick(LxH, roughness, ior);
    // return brdf_fresnel_cook_torrance(LxH, roughness, ior);
}


/* Material Functions */
vec3 get_material_f0(Material m) {
    // Material specular is already in the 0 .. 0.08 range
    return mix(vec3(m.specular), m.basecolor, m.metallic);
}


// Returns a reflection vector, bent into the normal direction
vec3 get_reflection_vector(Material m, vec3 view_vector) {
    float roughness = m.shading_model == SHADING_MODEL_CLEARCOAT ?
        CLEARCOAT_ROUGHNESS : m.roughness;
    vec3 reflected_dir = reflect(view_vector, m.normal);
    // XXX: Evaluate whats more physically correct
    // return normalize(mix(m.normal, reflected_dir, (1 - roughness) * (roughness + sqrt(1 - roughness))));
    return reflected_dir;
}


// Returns an approximated mipmap level based on the materials roughness
// level to approximate importance sampled references
float get_specular_mipmap(float roughness) {

    // Approximation to match importance sampled reference, tuned for a
    // resolution of 128.
    return (roughness * 12.0 - pow(roughness, 6.0) * 1.5);
}

float get_specular_mipmap(Material m) {
    return get_specular_mipmap(m.roughness);
}

vec3 get_brdf_from_lut(sampler3D lut_texture, float NxV, float roughness, float ior) {
    float lookup_slice = (ior - 1.01) / 1.5 + 0.5 / 15.0;
    vec3 data = textureLod(lut_texture, vec3(NxV, roughness, lookup_slice), 0).xyz;

    // Unpack packed data
    data *= data;

    return data;
}

vec3 get_brdf_from_lut(sampler2D lut_texture, float NxV, float roughness) {
    vec3 data = textureLod(lut_texture, vec2(NxV, roughness), 0).xyz;
    // Unpack packed data
    data *= data;
    return data;
}

float get_effective_roughness(Material m) {
    return m.shading_model == SHADING_MODEL_CLEARCOAT ? CLEARCOAT_ROUGHNESS : m.roughness;
}

float get_mipmap_for_roughness(samplerCube map, float roughness, float NxV) {
    return sqrt(roughness) * 7.0;
}


vec3 get_metallic_fresnel_approx(Material m, float NxV) {
    vec3 metallic_energy_f0 = vec3(1.0 - 0.7 * m.roughness) * m.basecolor;
    vec3 metallic_energy_f90 = mix(vec3(1), 0.5 * m.basecolor, m.linear_roughness);
    vec3 metallic_fresnel = mix(metallic_energy_f0, metallic_energy_f90,
        pow(1 - NxV, 3.6 - 2.6 * m.linear_roughness));
    return metallic_fresnel;
}
