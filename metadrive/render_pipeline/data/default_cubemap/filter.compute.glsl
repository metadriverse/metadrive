#version 430

// Shader to pre-filter the cubemap using importance sampling

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;


#define M_PI 3.1415926535897932384626433

vec3 get_transformed_coord(vec2 coord, uint face) {
    float f = 1.0;
    switch (face) {
        case 1: return vec3(-f, coord);
        case 2: return vec3(coord, -f);
        case 0: return vec3(f, -coord.x, coord.y);
        case 3: return vec3(coord.xy * vec2(1, -1), f);
        case 4: return vec3(coord.x, f, coord.y);
        case 5: return vec3(-coord.x, -f, coord.y);
    }
    return vec3(0);
}

// From:
// http://www.trentreed.net/blog/physically-based-shading-and-image-based-lighting/
vec2 hammersley(uint i, uint N)
{
    return vec2(float(i) / float(N), float(bitfieldReverse(i)) * 2.3283064365386963e-10);
}

// From:
// http://www.gamedev.net/topic/655431-ibl-problem-with-consistency-using-ggx-anisotropy/
vec3 importance_sample_ggx(vec2 xi, float roughness)
{
    float r_square = roughness * roughness;
    float phi = 2 * M_PI * xi.x;
    float cos_theta = sqrt((1 - xi.y) / (1 + (r_square * r_square - 1) * xi.y));
    float sin_theta = sqrt(1 - cos_theta * cos_theta);

    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Converts a normalized spherical coordinate (r = 1) to cartesian coordinates
vec3 spherical_to_vector(float theta, float phi) {
    float sin_theta = sin(theta);
    return normalize(vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos(theta)
    ));
}

float brdf_distribution_ggx(float NxH, float roughness) {
    float nxh_sq = NxH * NxH;
    float tan_sq = (1 - nxh_sq) / nxh_sq;
    float f = roughness / max(1e-10, nxh_sq * (roughness * roughness + tan_sq));
    return f * f / M_PI;
}

// Finds a tangent and bitangent vector based on a given normal
void find_arbitrary_tangent(vec3 normal, out vec3 tangent, out vec3 bitangent) {
    vec3 v0 = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(0, 1, 0);
    tangent = normalize(cross(v0, normal));
    bitangent = normalize(cross(tangent, normal));
}

vec3 transform_cubemap_coordinates(vec3 coord) {
    return normalize(coord.xyz * vec3(1, -1, 1));
}

uniform samplerCube SourceTex;
uniform int currentSize;
uniform int currentMip;
uniform int currentFace;
layout(rgba16f) uniform imageCube DestTex;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    vec2 texcoord = vec2(coord + 0.5) / float(currentSize);
    texcoord = texcoord * 2.0 - 1.0;

    vec3 n = get_transformed_coord(texcoord, currentFace);
    n = normalize(n);
    n = transform_cubemap_coordinates(n);
    float roughness = clamp(float(currentMip) / 7.0, 0.001, 1.0);
    // roughness *= roughness;

    vec3 tangent, binormal;
    find_arbitrary_tangent(n, tangent, binormal);

    vec4 accum = vec4(0);
    const uint num_samples = 512;

    // Ultra high quality, might cause TDR on low-end systems
    // const uint num_samples = 4096;
    for (uint i = 0; i < num_samples; ++i) {
        vec2 xi = hammersley(i, num_samples);
        vec3 r = importance_sample_ggx(xi, roughness);
        vec3 h = normalize(r.x * tangent + r.y * binormal + r.z * n);
        vec3 l = 2.0 * dot(n, h) * h - n;

        float NxL = clamp(dot(n, l), 0.0, 1.0);
        float NxH = clamp(dot(n, h), 0.0, 1.0);

        vec3 sampled = textureLod(SourceTex, l, 0).rgb;

        float weight = 1;
        weight = clamp(weight, 0.0, 1.0);
        accum += vec4(sampled, 1) * weight;
    }

    accum /= max(0.1, accum.w);
    accum.xyz = pow(accum.xyz, vec3(2.2));

    imageStore(DestTex, ivec3(coord, currentFace), vec4(accum.xyz, 1.0));
}
