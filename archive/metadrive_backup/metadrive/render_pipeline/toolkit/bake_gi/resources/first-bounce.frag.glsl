#version 430

in vec3 ws_position;
in vec3 ws_normal;
in vec2 ws_uv;

uniform sampler2D p3d_Texture0;
uniform sampler2D ShadowMap;
uniform mat4 shadowMVP;
uniform vec3 sunVector;

out vec4 color;

const vec3 sun_color = vec3(1.4, 1.1, 1.0) * 1;

void main() {
    vec4 projected = shadowMVP * vec4(ws_position, 1);
    vec3 shadow_space_pos = projected.xyz / projected.w * 0.5 + 0.5;

    float actual_depth = textureLod(ShadowMap, shadow_space_pos.xy, 0).x;

    const float bias = 0.001;
    float shadow = actual_depth >= shadow_space_pos.z - bias ? 1.0 : 0.0;

    float NxL = max(0.0, dot(ws_normal, sunVector));
    vec3 basecolor = texture(p3d_Texture0, ws_uv).xyz;
    // basecolor = vec3(0.8);

    vec3 ambient = vec3(0.0001) * basecolor;

    color = vec4(basecolor * NxL * shadow * sun_color + ambient, 1);
}
