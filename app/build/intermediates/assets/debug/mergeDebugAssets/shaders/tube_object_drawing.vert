#version 300 es

uniform mat4 u_Model;
uniform mat4 u_ModelViewProjection;

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;

out vec3 v_Normal;

void main() {
  v_Normal = normalize(mat3(u_Model) * a_Normal);
  gl_Position = u_ModelViewProjection * vec4(a_Position, 1.0);
}
