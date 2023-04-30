#version 430

#pragma include "../_bake_params.glsl"

in vec3 ws_normal;
in vec3 ws_position;
in vec2 ms_uv;

uniform sampler2D p3d_Texture0;
uniform sampler2D GIDataTexture;

uniform sampler2D ShadowMap;
uniform mat4 shadowMVP;
uniform vec3 sunVector;

out vec3 color;

vec3 get_probe_at(ivec3 coord, vec3 nrm) {
        int offs = coord.x + coord.y * bake_mesh_probecount.x +
                    coord.z * bake_mesh_probecount.y * bake_mesh_probecount.x;
        if (offs > bake_mesh_probecount.x * bake_mesh_probecount.y * bake_mesh_probecount.z ||
            offs < 0) {
            return vec3(1, 0, 0);
        }

        ivec2 base_coord = ivec2((offs % bake_divisor) * 6, offs / bake_divisor);
        vec3 baked = vec3(0);
        baked += texelFetch(
            GIDataTexture, ivec2(0, 0) + base_coord, 0).xyz * max(0, dot(nrm, vec3(1, 0, 0)));
        baked += texelFetch(
            GIDataTexture, ivec2(1, 0) + base_coord, 0).xyz * max(0, dot(nrm, vec3(-1, 0, 0)));
        baked += texelFetch(
            GIDataTexture, ivec2(2, 0) + base_coord, 0).xyz * max(0, dot(nrm, vec3(0, 1, 0)));
        baked += texelFetch(
            GIDataTexture, ivec2(3, 0) + base_coord, 0).xyz * max(0, dot(nrm, vec3(0, -1, 0)));
        baked += texelFetch(
            GIDataTexture, ivec2(4, 0) + base_coord, 0).xyz * max(0, dot(nrm, vec3(0, 0, 1)));
        baked += texelFetch(
            GIDataTexture, ivec2(5, 0) + base_coord, 0).xyz * max(0, dot(nrm, vec3(0, 0, -1)));

        return baked;
}

void main() {

    vec3 biased_position = ws_position + 0.4 * ws_normal;
    vec3 local_coord = (biased_position - bake_mesh_start) / (bake_mesh_end - bake_mesh_start);

    ivec3 local_index = ivec3(local_coord * bake_mesh_probecount);
    vec3 fract_coord = (local_coord * bake_mesh_probecount - local_index);

    // 3D Lerp
    vec3 probe_000 = get_probe_at(local_index + ivec3(0, 0, 0), ws_normal);
    vec3 probe_001 = get_probe_at(local_index + ivec3(0, 0, 1), ws_normal);
    vec3 probe_010 = get_probe_at(local_index + ivec3(0, 1, 0), ws_normal);
    vec3 probe_011 = get_probe_at(local_index + ivec3(0, 1, 1), ws_normal);
    vec3 probe_100 = get_probe_at(local_index + ivec3(1, 0, 0), ws_normal);
    vec3 probe_101 = get_probe_at(local_index + ivec3(1, 0, 1), ws_normal);
    vec3 probe_110 = get_probe_at(local_index + ivec3(1, 1, 0), ws_normal);
    vec3 probe_111 = get_probe_at(local_index + ivec3(1, 1, 1), ws_normal);

    vec3 lerp_bottom = mix(
        mix(probe_000, probe_100, fract_coord.x),
        mix(probe_010, probe_110, fract_coord.x),
        fract_coord.y
    );

    vec3 lerp_top = mix(
        mix(probe_001, probe_101, fract_coord.x),
        mix(probe_011, probe_111, fract_coord.x),
        fract_coord.y
    );

    vec3 precomputed_gi = mix(lerp_bottom, lerp_top, fract_coord.z);

    vec3 basecolor = textureLod(p3d_Texture0, ms_uv, 0).xyz;
    // basecolor = vec3(0.8);


    // Shadows
    vec4 projected = shadowMVP * vec4(ws_position, 1);
    vec3 shadow_space_pos = projected.xyz / projected.w * 0.5 + 0.5;

    float actual_depth = textureLod(ShadowMap, shadow_space_pos.xy, 0).x;

    const float bias = 0.001;
    float shadow = actual_depth >= shadow_space_pos.z - bias ? 1.0 : 0.0;
    float NxL = max(0, dot(sunVector, ws_normal));
    const vec3 sun_color = vec3(1.4, 1.2, 1.0);

    color = vec3(basecolor * precomputed_gi + basecolor * shadow * sun_color * NxL);
    color = pow(color, vec3(1.0 / 2.2));
}
