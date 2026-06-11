#version 300 es
/*
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

uniform mat4 u_Model;
uniform mat4 u_ModelViewProjection;
uniform mat2 u_PlaneUvMatrix;
uniform vec3 u_Normal;
uniform int u_PlaneType; // 0 = horizontal, 1 = vertical

layout(location = 0) in vec3 a_XZPositionAlpha; // (x, z, alpha)

out vec3 v_TexCoordAlpha;
out vec3 v_WorldPos;

void main() {
   vec4 local_pos = vec4(a_XZPositionAlpha.x, 0.0, a_XZPositionAlpha.y, 1.0);
   vec4 world_pos = u_Model * local_pos;

// Build a stable UV basis so the grid "looks right":
// - Horizontal planes: U aligns to world +X, V aligns to world +Z
// - Vertical planes:   V aligns to world +Y (up), U is derived from normal x V
vec3 worldX = vec3(1.0, 0.0, 0.0);
vec3 worldY = vec3(0.0, 1.0, 0.0);

vec3 vec_u;
vec3 vec_v;

if (u_PlaneType == 1) {
  // Vertical plane: make V point "up" along world Y, projected onto the plane.
  vec_v = worldY - u_Normal * dot(worldY, u_Normal);
  float lenV = length(vec_v);
  if (lenV < 1e-4) {
    // Fallback if projection degenerates.
    vec_v = normalize(cross(u_Normal, vec3(1.0, 0.0, 0.0)));
  } else {
    vec_v /= lenV;
  }
  vec_u = normalize(cross(u_Normal, vec_v));
} else {
  // Horizontal plane: make U point along world X, projected onto the plane.
  vec_u = worldX - u_Normal * dot(worldX, u_Normal);
  float lenU = length(vec_u);
  if (lenU < 1e-4) {
    vec_u = normalize(cross(u_Normal, vec3(0.0, 0.0, 1.0)));
  } else {
    vec_u /= lenU;
  }
  vec_v = normalize(cross(u_Normal, vec_u));
}

   // Project vertices in world frame onto vec_u and vec_v.
   vec2 uv = vec2(dot(world_pos.xyz, vec_u), dot(world_pos.xyz, vec_v));
   v_TexCoordAlpha = vec3(u_PlaneUvMatrix * uv, a_XZPositionAlpha.z);

   v_WorldPos = world_pos.xyz;

   gl_Position = u_ModelViewProjection * local_pos;
}