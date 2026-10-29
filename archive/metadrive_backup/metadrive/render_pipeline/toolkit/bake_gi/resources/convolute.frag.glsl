#version 430

uniform samplerCube SourceTex;
layout(rgba16f) uniform image2D DestTex;
uniform ivec2 storeCoord;
// out vec3 color;

#define TWO_PI 6.2831853071795864769252867

vec2 hammersley(uint i, uint N) {
    return vec2(float(i) / float(N), float(bitfieldReverse(i)) * 2.3283064365386963e-10);
}

vec3 importance_sample_lambert(vec2 xi) {
    float phi = TWO_PI * xi.x;
    float cos_theta = sqrt(xi.y);
    float sin_theta = sqrt(1 - cos_theta * cos_theta);
    vec3 H;
    H.x = sin_theta * cos(phi);
    H.y = sin_theta * sin(phi);
    H.z = cos_theta;
    return vec3(H);
}

vec3 get_sky_color(vec3 v) {
    return vec3(51, 137, 233) / 255.0 * max(0, v.z) * 0.5;
}

void main() {
    int offs = int(gl_FragCoord.x);
    vec3 direction = vec3(0);
    switch(offs) {
        case 0: direction = vec3(1, 0, 0); break;
        case 1: direction = vec3(-1, 0, 0); break;
        case 2: direction = vec3(0, 1, 0); break;
        case 3: direction = vec3(0, -1, 0); break;
        case 4: direction = vec3(0, 0, 1); break;
        case 5: direction = vec3(0, 0, -1); break;
    }

    const int num_samples = 512;

    // Find tangent / binormal
    vec3 v0 = abs(direction.z) < 0.999 ? vec3(0, 0, 1) : vec3(0, 1, 0);
    vec3 tangent = normalize(cross(v0, direction));
    vec3 bitangent = normalize(cross(tangent, direction));

    vec3 accum = vec3(0);

    for (uint i = 0; i < num_samples; ++i) {
        vec2 xi = hammersley(i, num_samples);
        vec3 h = importance_sample_lambert(xi);
        h = normalize(h.x * tangent + h.y * bitangent + h.z * direction);

        // vec3 l = -reflect(direction, h);
        vec4 sampled_color = textureLod(SourceTex, h, 0);
        accum += sampled_color.xyz * sampled_color.w + get_sky_color(h) * (1 - sampled_color.w);
    }

    accum /= num_samples;

    // color = vec3(accum);

    imageStore(DestTex, ivec2(storeCoord.x * 6 + gl_FragCoord.x, storeCoord.y), vec4(accum, 1));
}
