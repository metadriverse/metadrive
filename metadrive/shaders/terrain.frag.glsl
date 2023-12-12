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

// Projects a point using the given mvp
vec3 project(mat4 mvp, vec3 p) {
    vec4 projected = mvp * vec4(p, 1);
    return (projected.xyz / projected.w) * vec3(0.5) + vec3(0.5);
}


vec3 get_normal(vec3 diffuse, sampler2D normal_tex, sampler2D rough_tex, float tex_ratio, mat3 tbn){
      vec3 normal = normalize(texture(normal_tex, terrain_uv * tex_ratio).rgb*2.0-1.0);
      normal = normalize(tbn * normal);
      return normal;
}

void main() {
  float road_tex_ratio = 128;
  float grass_tex_ratio = grass_tex_ratio * 4;
  float r_min = (1-1/elevation_texture_ratio)/2;
  float r_max = (1-1/elevation_texture_ratio)/2+1/elevation_texture_ratio;
  vec4 attri;
  if (abs(elevation_texture_ratio - 1) < 0.001) {
    attri = texture(attribute_tex, terrain_uv);
  }
  else {
    attri = texture(attribute_tex, terrain_uv*elevation_texture_ratio+0.5);
  }

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
  // normal.x *= -1;

  mat3 tbn = mat3(tangent, binormal, terrain_normal);
  vec3 shading = vec3(0.0);

  // Calculate the shading of each light in the scene
  for (int i = 0; i < p3d_LightSource.length(); ++i) {
    vec3 diff = p3d_LightSource[i].position.xyz - vtx_pos * p3d_LightSource[i].position.w;
    vec3 light_vector = normalize(diff);
    vec3 light_shading = clamp(dot(normal, light_vector), 0.0, 1.0) * p3d_LightSource[i].color;
    // If PSSM is not used, use the shadowmap from the light
    // This is deeply ineficient, it's only to be able to compare the rendered shadows
    if (!use_pssm) {
      vec4 projected = projecteds[i];
      // Apply a bias to remove some of the self-shadow acne
      projected.z -= fixed_bias * 0.01 * projected.w;
      light_shading *= textureProj(p3d_LightSource[i].shadowMap, projected);
    }
    shading += light_shading;
  }

  // get the color and terrain normal in world space
  vec3 diffuse;
  vec3 tex_normal_world;
  if ((attri.r > 0.01) && (terrain_uv.x>=r_min) && (terrain_uv.y >= r_min) && (terrain_uv.x<=r_max) && (terrain_uv.y<=r_max)){
    float value = attri.r; // Assuming it's a red channel texture
    if (value < 0.11) {
        // yellow
        diffuse=texture(yellow_tex, terrain_uv * road_tex_ratio).rgb;
    } else if (value < 0.21) {
        // road
        diffuse = texture(road_tex, terrain_uv * road_tex_ratio).rgb;
    } else if (value < 0.31) {
        // white
        diffuse = texture(white_tex, terrain_uv * road_tex_ratio).rgb;
    }  else if (value > 0.3999 ||  value < 0.760001) {
        // crosswalk
        float theta=(value-0.39999) * 1000/180 * 3.1415926535;
        vec2 new_terrain_uv = vec2(cos(theta)*terrain_uv.x - sin(theta)*terrain_uv.y, sin(theta)*terrain_uv.x+cos(theta)*terrain_uv.y);
        diffuse = texture(crosswalk_tex, new_terrain_uv * road_tex_ratio).rgb;
    } else{
        // Semantics for value 4
        diffuse = texture(white_tex, terrain_uv * road_tex_ratio).rgb;
    }
    tex_normal_world = get_normal(diffuse, road_normal,  road_rough, road_tex_ratio, tbn);
  }
  else{

      // texture splatting, mixing ratio can be determined via rgba, no grass here
      diffuse = texture(grass_tex, terrain_uv * grass_tex_ratio).rgb;
      tex_normal_world = get_normal(diffuse, grass_normal, grass_rough, grass_tex_ratio, tbn);
    }

//   vec3 terrain_normal_view =  normalize(tex_normal_world);

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
                float depth_sample = texture2D(PSSMShadowAtlas, projected_coord.xy + vec2(x, y) * radius).r;
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