#define GLOBAL_WATER_SCALE 0.6
#define WATER_DISPLACE (vec3(1.0, 1.0, 0.9) * 2.8 * GLOBAL_WATER_SCALE)
#define WATER_COORD_FACTOR (20.0 * GLOBAL_WATER_SCALE)
#define WATER_LOWRES_FACTOR 0.4
#define WATER_DISPLACE_DIST 25000.0

uniform mat4 p3d_ViewProjectionMatrixInverse;

vec3 reprojectCoord(vec2 coord, float depth) {
    vec4 proj = p3d_ViewProjectionMatrixInverse * vec4(coord, depth*2.0-1.0, 1.0);
    proj.xyz /= proj.w;
    return proj.xyz;
}