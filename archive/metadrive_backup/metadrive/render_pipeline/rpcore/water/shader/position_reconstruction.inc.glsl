# pragma once

uniform mat4 trans_clip_of_mainCam_to_mainRender;
uniform mat4 currentProjMatInv;

const float ndcNear = CAMERA_NEAR;
const float ndcFar = CAMERA_FAR;
const float ndcA = ndcNear + ndcFar;
const float ndcB = ndcNear - ndcFar;
const float ndcC = 2.0 * ndcNear * ndcFar;
const float ndcD = ndcFar - ndcNear;

float getZFromNdc(vec3 ndcPos) {
  return ndcC / (ndcA+(ndcPos.z * ndcB));
}

float getZFromNdc(vec3 ndcPos, float near, float far) {
  float d = ndcPos.z * (near-far);
  return ( (2.0 * near * far) / ((near + far) + d));
}

float getZFromLinearZ(float z) {
  z /= ndcC;
  z = 1.0 / z;
  z -= ndcA;
  z /= ndcB;
  z /= 2.0;
  z += 0.5;
  return z;
}

float getLinearZFromZ(float z) {
    float z_n = z * 2.0 - 1.0;
    float z_e = ndcC / (ndcA - z_n * ndcD);
    return z_e;
}

float getCustomLinearZFromZ(float z, float near, float far) {
    float z_n = z * 2.0 - 1.0;
    float z_e = 2.0 * near * far / (far + near - z_n * (far - near));
    return z_e;
}

float normalizeZ(float z, float near, float far) {
  return getCustomLinearZFromZ(z, near, far) / far;
}

vec3 calculateSurfacePos(float z, vec2 tcoord, mat4 clipToRender) {
  vec3 ndcPos = vec3(tcoord.xy, z) * 2.0 - 1.0;    
  vec4 clipPos = vec4(getZFromNdc(ndcPos));
  clipPos.xyz = ndcPos * clipPos.w;
  return (clipToRender * clipPos).xyz;
}

vec3 calculateSurfacePos(float z, vec2 tcoord) {
  return calculateSurfacePos(z, tcoord, trans_clip_of_mainCam_to_mainRender);
}

vec3 calculateSurfacePos(float z, vec2 tcoord, float near, float far, mat4 clipToRender) {
  vec3 ndcPos = vec3(tcoord.xy, z) * 2.0 - 1.0;    
  vec4 clipPos = vec4(getZFromNdc(ndcPos, near, far));
  clipPos.xyz = ndcPos * clipPos.w;
  return (clipToRender * clipPos).xyz;
}


vec3 calculateViewPos(float z, vec2 tcoord) {
  vec3 ndc = vec3(tcoord, z) * 2.0 - 1.0;
  vec4 sampleViewPos = currentProjMatInv * vec4(ndc, 1.0);
  return (sampleViewPos.xzy * vec3(1,1,-1)) / sampleViewPos.w ;
}