#version 120

// Zhenghao notes: following replacement suggestion in https://stackoverflow.com/questions/24737705/opengl-shader-builder-errors-on-compiling

// Uniform inputs
uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform vec4 p3d_Color;

// Vertex inputs
attribute vec4 p3d_Vertex;

// Vertex outputs
varying float distanceToCamera;

void main() {
  vec4 cs_position = p3d_ModelViewMatrix * p3d_Vertex;
  distanceToCamera = length(cs_position.xyz);
  gl_Position = p3d_ProjectionMatrix * cs_position;
}