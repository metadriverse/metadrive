#version 330

// Number of splits in the PSSM, it must be in line with what is configured in the PSSMCameraRig
const int split_count=2;
uniform  vec3 light_direction;
#define saturate(v) clamp(v, 0, 1)
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
uniform sampler2D road_tex;
uniform sampler2D road_normal;
// uniform sampler2D road_rough;
uniform float road_tex_ratio;
uniform sampler2D crosswalk_tex;

uniform sampler2D grass_tex;
uniform sampler2D grass_normal;
// uniform sampler2D grass_rough;
uniform float grass_tex_ratio;

uniform sampler2D rock_tex;
uniform sampler2D rock_normal;
// uniform sampler2D rock_rough;
uniform float rock_tex_ratio;

uniform sampler2D rock_tex_2;
uniform sampler2D rock_normal_2;
// uniform sampler2D rock_rough_2;
uniform float rock_tex_ratio_2;

uniform sampler2D attribute_tex;

// just learned that uniform means the variable won't change in each stage, while in/out is able to do that : )
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

// Projects a point using the given mvp
vec3 project(mat4 mvp, vec3 p) {
    vec4 projected = mvp * vec4(p, 1);
    return (projected.xyz / projected.w) * vec3(0.5) + vec3(0.5);
}


vec3 get_normal(vec3 diffuse, sampler2D normal_tex, float tex_ratio, mat3 tbn){
      vec3 normal = texture(normal_tex, terrain_uv * tex_ratio).rgb*2.0-1.0;
      normal = normalize(tbn * normal);
      return normal;
}

void main() {
  vec4 attri = texture(attribute_tex, terrain_uv);

  // terrain normal
  vec3 pixel_size = vec3(1.0, -1.0, 0) / textureSize(ShaderTerrainMesh.heightfield, 0).xxx;
  float h_u0 = texture(ShaderTerrainMesh.heightfield, terrain_uv + pixel_size.yz).x * height_scale;
  float h_u1 = texture(ShaderTerrainMesh.heightfield, terrain_uv + pixel_size.xz).x * height_scale;
  float h_v0 = texture(ShaderTerrainMesh.heightfield, terrain_uv + pixel_size.zy).x * height_scale;
  float h_v1 = texture(ShaderTerrainMesh.heightfield, terrain_uv + pixel_size.zx).x * height_scale;
  vec3 tangent = normalize(vec3(1, 0, h_u1 - h_u0));
  vec3 binormal = normalize(vec3(0, 1, h_v1 - h_v0));
  vec3 terrain_normal = normalize(cross(tangent, binormal));
  vec3 normal = normalize(p3d_NormalMatrix * terrain_normal);
  vec3 viewDir = normalize(wspos_camera - vtx_pos);
  float height = (h_u0 + h_u1 + h_v0 + h_v1) / (4.0 * height_scale); // xxx
  float slope = 1.0 - terrain_normal.z;
  // normal.x *= -1;

  mat3 tbn = mat3(tangent, binormal, terrain_normal);
  vec3 shading = vec3(0.0);

  // get the color and terrain normal in world space
  vec3 diffuse = vec3(0.0, 0.0, 0.0);
  vec3 tex_normal_world;
  // float roughnessValue;
  float value = attri.r * 255; // Assuming it's a red channel texture
  if (value > 5){
    if (value < 16) {
        // white
        diffuse = vec3(1.0, 1.0, 1.0);
    } else if (value < 26) {
        // road
        diffuse = texture(road_tex, terrain_uv * road_tex_ratio).rgb;
    } else if (value < 34) {
        // yellow
        diffuse=vec3(1.0, 0.78, 0.0);
    }  else if (value > 39 ||  value < 222) {
        // crosswalk
        float theta=(value-40) * 2/180.0*3.1415926535;
        vec2 new_terrain_uv = vec2(cos(theta)*terrain_uv.x - sin(theta)*terrain_uv.y, sin(theta)*terrain_uv.x+cos(theta)*terrain_uv.y);
        diffuse = texture(crosswalk_tex, new_terrain_uv * road_tex_ratio).rgb;
    }
    tex_normal_world = get_normal(diffuse, road_normal, road_tex_ratio, tbn);
    // roughnessValue = texture(road_rough, terrain_uv * road_tex_ratio).r;
  }
  else{
      // texture splatting
      float grass = 0.0;
      float rock = 0.0;
      float rock_2 = 0.0;

      { // rock_2
        rock_2 = saturate(0.8 * (height-0.07));
        rock_2 *= saturate(pow(saturate(1.0 - slope), 2.0)) * 2.0;

        rock_2 = saturate(rock_2);
        }

        { // Rock
            rock = saturate((pow(slope, 1.2) * 15));
        }

        { // Grass
            grass = saturate(1.0 - saturate(rock + rock_2));
        }

      diffuse = diffuse + texture(grass_tex, terrain_uv * grass_tex_ratio).rgb * grass;
      diffuse = diffuse + texture(rock_tex, terrain_uv * rock_tex_ratio).rgb * rock;
      diffuse = diffuse + texture(rock_tex_2, terrain_uv * rock_tex_ratio_2).rgb * rock_2;

      tex_normal_world = tex_normal_world + (texture(grass_normal, terrain_uv * grass_tex_ratio).rgb*2.0-1.0) * grass;
      tex_normal_world = tex_normal_world + (texture(rock_normal, terrain_uv * rock_tex_ratio).rgb*2.0-1.0) * rock;
      tex_normal_world = tex_normal_world + (texture(rock_normal_2, terrain_uv * rock_tex_ratio_2).rgb*2.0-1.0) * rock_2;
      tex_normal_world = normalize(tbn * tex_normal_world);

      //roughnessValue = roughnessValue + texture(grass_rough, terrain_uv * grass_tex_ratio).r * grass;
      //roughnessValue = roughnessValue + texture(rock_rough, terrain_uv * rock_tex_ratio).r * rock;
      //roughnessValue = roughnessValue + texture(rock_rough_2, terrain_uv * rock_tex_ratio_2).r * rock_2;
      //roughnessValue = saturate(roughnessValue);
    }

//   vec3 terrain_normal_view =  normalize(tex_normal_world);

  // Calculate the shading of each light in the scene
  for (int i = 0; i < p3d_LightSource.length(); ++i) {
    vec3 diff = p3d_LightSource[i].position.xyz - vtx_pos * p3d_LightSource[i].position.w;
    vec3 light_vector = normalize(diff);
    vec3 light_shading = clamp(dot(normal, light_vector), 0.0, 1.0) * p3d_LightSource[i].color;

      // Specular (Blinn-Phong example)
    // vec3 halfDir   = normalize(light_vector + viewDir);
    // float NdotH    = max(dot(tex_normal_world, halfDir), 0.0);
    // float exponent = 2.0 + (1.0 - roughnessValue) * 256.0;
    // float spec     = pow(NdotH, exponent);
    // float specStrength = 0.4;
    // vec3 specColor = p3d_LightSource[i].color * spec * specStrength;
    // light_shading += specColor;


    shading += light_shading;
  }

  // static shadow
  vec3 light_dir = normalize(light_direction);
  shading *= max(0.0, dot(tex_normal_world, light_dir));
  shading += vec3(0.07, 0.07, 0.1);

//   dynamic shadow
  int split = 99;
  if (use_pssm) {
    // Find in which split the current point is present.

    float border_bias = 0.5 - (0.5 / (1.0 + border_bias));

    // Find the first matching split
    for (int i = 0; i < split_count; ++i) {
        vec3 coord = project(pssm_mvps[i], vtx_pos);
        if (coord.x >= border_bias && coord.x <= 1 - border_bias &&
            coord.y >= border_bias && coord.y <= 1 - border_bias &&
            coord.z >= 0.0 && coord.z <= 1.0) {
            split = i;
            break;
        }
    }

    // Compute the shadowing factor
    if (split < split_count) {

        // Get the MVP for the current split
        mat4 mvp = pssm_mvps[split];

        // Project the current pixel to the view of the light
        vec3 projected = project(mvp, vtx_pos);
        vec2 projected_coord = vec2((projected.x + split) / float(split_count), projected.y);
        // Apply a fixed bias based on the current split to diminish the shadow acne
        float ref_depth = projected.z - fixed_bias * 0.001 * (1 + 1.5 * split);

        // Check if the pixel is shadowed or not
        float shadow_factor=0.0;
        float samples = 9.0; // Number of samples
        float radius = 0.001; // Sample radius
        for(int x = -1; x <= 1; x++) {
            for(int y = -1; y <= 1; y++) {
                float depth_sample = texture(PSSMShadowAtlas, projected_coord.xy + vec2(x, y) * radius).r;
                shadow_factor += step(ref_depth, depth_sample);
        }
        }
        shadow_factor /= samples;
        shading *= shadow_factor;
    }
    }

  shading += p3d_LightModel.ambient.xyz;

  shading *= diffuse.xyz;

  if (fog) {
    // Fake fog
    float dist = distance(vtx_pos, wspos_camera);
    float fog_factor = smoothstep(0, 1, dist / 8000.0);
    shading = mix(shading, vec3(0.7, 0.7, 0.8), fog_factor);
  }
//   if (split==0){
//     shading = vec3(1, 0, 0);
//   }
//   else if (split==1) {
//     shading = vec3(0, 1, 0);
//   }
//   else if (split==2) {
//     shading = vec3(0, 0, 1);
//   }
//   else{
//     shading = vec3(0, 1, 1);
//   }
  color = vec4(shading, 1.0);
}