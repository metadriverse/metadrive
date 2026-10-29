#version 120

// Zhenghao notes: following replacement suggestion in https://stackoverflow.com/questions/24737705/opengl-shader-builder-errors-on-compiling

varying float distanceToCamera;
varying vec4 color;

// out vec4 fragColor;

void main() {
  float base=8;
  float b = 16;
  float c = log(distanceToCamera/base)/log(b);

  gl_FragColor = vec4(c, c, c, 0);
}