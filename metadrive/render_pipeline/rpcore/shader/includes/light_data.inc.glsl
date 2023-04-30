#pragma once


#pragma include "includes/light_data.struct.glsl"
#pragma include "includes/source_data.struct.glsl"

/*

Description of the packing and the layout can be found here:
https://github.com/tobspr/RenderPipeline/wiki/LightStorage

*/

// Reads the light data from a given buffer and offset
LightData read_light_data(samplerBuffer LightDataBuffer, int offset) {
    LightData data;
    data.Data0 = texelFetch(LightDataBuffer, offset * 4 + 0);
    data.Data1 = texelFetch(LightDataBuffer, offset * 4 + 1);
    data.Data2 = texelFetch(LightDataBuffer, offset * 4 + 2);
    data.Data3 = texelFetch(LightDataBuffer, offset * 4 + 3);
    return data;
}

// Only reads the light type, in case nothing else is required
int read_light_type(samplerBuffer LightDataBuffer, int offset) {
    float data0x = texelFetch(LightDataBuffer, offset * 4 + 0).x;
    return gpu_cq_unpack_int_from_float(data0x);
}

bool read_casts_shadows(samplerBuffer LightDataBuffer, int offset) {
    float data0z = texelFetch(LightDataBuffer, offset * 4 + 0).z;
    return gpu_cq_unpack_int_from_float(data0z) >= 0;
}

// Extracts the type of a light
int get_light_type(LightData data) {
    return gpu_cq_unpack_int_from_float(data.Data0.x);
}

// Extracts the ies profile index of a light
int get_ies_profile(LightData data) {
    return gpu_cq_unpack_int_from_float(data.Data0.y);
}

int get_shadow_source_index(LightData data) {
    return gpu_cq_unpack_int_from_float(data.Data0.z);
}

bool get_casts_shadows(LightData data) {
    return get_shadow_source_index(data) >= 0;
}

// Extracts the light world space position
vec3 get_light_position(LightData data) {
    return vec3(data.Data0.w, data.Data1.xy);
}

// Extracts the light color
vec3 get_light_color(LightData data) {
    return vec3(data.Data1.zw, data.Data2.x) * 100.0;
}

/*

Point Light Dataset

*/

// Extracts the radius of a point light
float get_pointlight_radius(LightData data) {
    return data.Data2.y;
}

// Extracts the inner radius of a point light
float get_pointlight_inner_radius(LightData data) {
    return data.Data2.z;
}

/*

Spot Light Dataset

*/

// Extracts the radius of a spot light
float get_spotlight_radius(LightData data) {
    return data.Data2.y;
}

// Extracts the fov of a spot light
float get_spotlight_fov(LightData data) {
    return data.Data2.z;
}

// Extracts the direction of a spot light
vec3 get_spotlight_direction(LightData data) {
    return vec3(data.Data2.w, data.Data3.xy);
}



/*

Shadow sources

*/

// Reads the shadow source data from a given buffer and offset
SourceData read_source_data(samplerBuffer SourceDataBuffer, int offset) {
    SourceData data;
    data.Data0 = texelFetch(SourceDataBuffer, offset + 0);
    data.Data1 = texelFetch(SourceDataBuffer, offset + 1);
    data.Data2 = texelFetch(SourceDataBuffer, offset + 2);
    data.Data3 = texelFetch(SourceDataBuffer, offset + 3);
    data.Data4 = texelFetch(SourceDataBuffer, offset + 4);
    return data;
}


mat4 get_source_mvp(SourceData data) {
    return mat4(data.Data0, data.Data1, data.Data2, data.Data3);
}

vec4 get_source_uv(SourceData data) {
    return data.Data4;
}
