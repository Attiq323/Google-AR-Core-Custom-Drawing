#version 300 es
precision mediump float;

uniform vec4 u_Color;

in vec3 v_Normal;
out vec4 o_FragColor;

void main() {
  vec3 normal = normalize(v_Normal);
  vec3 lightDirection = normalize(vec3(0.35, 0.85, 0.45));
  float diffuse = max(dot(normal, lightDirection), 0.80);
  o_FragColor = vec4(u_Color.rgb * diffuse, u_Color.a);
}
