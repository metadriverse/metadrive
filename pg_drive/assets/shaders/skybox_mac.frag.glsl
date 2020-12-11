#version 120

// Zhenghao notes: following replacement suggestion in https://stackoverflow.com/questions/24737705/opengl-shader-builder-errors-on-compiling
// and change the textureLod to texture2D
// Now it works!

varying vec3 skybox_pos;
// out vec4 color;

uniform sampler2D p3d_Texture0;

void main() {

  vec3 view_dir = normalize(skybox_pos);
  vec2 skybox_uv;

  // Convert spherical coordinates
  const float pi = 3.14159265359;
  skybox_uv.x = (atan(view_dir.y, view_dir.x) + (0.5 * pi)) / (2 * pi);
  skybox_uv.y = clamp(view_dir.z * 0.72 + 0.35, 0.0, 1.0);

  // textureLod should be changed to texture2D
  vec3 skybox_color = texture2D(p3d_Texture0, skybox_uv, 0).xyz;

  gl_FragColor = vec4(skybox_color, 1);
}
