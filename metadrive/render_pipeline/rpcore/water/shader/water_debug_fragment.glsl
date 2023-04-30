#version 400
out vec3 color;
in vec2 texcoord;
uniform sampler2D displacement;
uniform sampler2D normal;


void main() {

    vec3 sun = normalize(vec3(1));

    vec3 displace = texture(displacement, texcoord).xyz;
    vec3 rawSample = texture(normal, texcoord).xyz;
    vec3 normalSample = normalize(vec3(rawSample.xy, 32.0 / 256.0));
    float fresnel = max(0.2, dot(normalSample, sun));
    vec3 diffuse = vec3(0.3,0.6,0.9) * fresnel;

    float specular = pow(max(0.0, dot(normalSample, sun)), 24.0) * 512.0;

    diffuse += vec3(rawSample.z) * 1.5;
    // diffuse += specular * vec3(1.2,1.2,1.0);

    color = vec3(diffuse.xyz);
}