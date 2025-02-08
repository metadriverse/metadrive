#version 330

// Number of splits in the PSSM, it must be in line with what is configured in the PSSMCameraRig
const int split_count=2;
uniform  vec3 light_direction;

uniform mat3 p3d_NormalMatrix;

uniform struct {
  sampler2D data_texture;
  sampler2D heightfield;
  int view_index;
  int terrain_size;
  int chunk_size;
} ShaderTerrainMesh;

uniform struct {
  vec4 position;
  vec3 color;
  vec3 attenuation;
  vec3 spotDirection;
  float spotCosCutoff;
  float spotExponent;
  sampler2DShadow shadowMap;
  mat4 shadowViewMatrix;
} p3d_LightSource[1];

uniform struct {
  vec4 ambient;
} p3d_LightModel;

uniform vec3 wspos_camera;

// asset
uniform sampler2D yellow_tex;
uniform sampler2D white_tex;
uniform sampler2D road_tex;
uniform sampler2D road_normal;
uniform sampler2D road_rough;
uniform sampler2D crosswalk_tex;

uniform sampler2D grass_tex;
uniform sampler2D grass_normal;
uniform sampler2D grass_rough;
uniform float grass_tex_ratio;

uniform sampler2D rock_tex;
uniform sampler2D rock_normal;
uniform sampler2D rock_rough;

uniform sampler2D attribute_tex;

// just learned that uniform means the variable won't change in each stage, while in/out is able to do that : )
uniform float elevation_texture_ratio;
uniform float height_scale;

uniform sampler2D PSSMShadowAtlas;

uniform mat4 pssm_mvps[split_count];
uniform vec2 pssm_nearfar[split_count];
uniform float border_bias;
const float fixed_bias=0;
uniform bool use_pssm;
uniform bool fog;

in vec2 terrain_uv;
in vec3 vtx_pos;
in vec4 projecteds[1];

out vec4 color;

void main() {
  color = vec4(0.0, 0.0, 0.0, 0.0);
}