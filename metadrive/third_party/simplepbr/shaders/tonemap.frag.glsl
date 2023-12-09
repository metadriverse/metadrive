#version 120

uniform sampler2D tex;
uniform float exposure;

varying vec2 v_texcoord;

#ifdef USE_330
out vec4 o_color;
#endif

void main() {
    vec3 color = texture2D(tex, v_texcoord).rgb;

    color *= exposure;
    color = max(vec3(0.0), color - vec3(0.004));
    color = (color * (vec3(6.2) * color + vec3(0.5))) / (color * (vec3(6.2) * color + vec3(1.7)) + vec3(0.06));

#ifdef USE_330
    o_color = vec4(color, 1.0);
#else
    gl_FragColor = vec4(color, 1.0);
#endif
}
