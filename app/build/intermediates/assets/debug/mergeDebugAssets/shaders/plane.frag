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

precision highp float;

uniform sampler2D u_Texture;
uniform vec4 u_GridControl;  // dotThreshold, lineThreshold, lineFadeShrink, occlusionShrink

// Spotlight (Filament-like) focus area for plane visualization.
uniform bool u_SpotlightEnabled;
uniform vec3 u_SpotlightFocusPoint; // world space
uniform float u_SpotlightRadius;    // meters

// Differentiate horizontal vs vertical planes.
uniform float u_PlaneAlphaScale;    // e.g. 1.0 for horizontal, 0.75 for vertical

in vec3 v_TexCoordAlpha;
in vec3 v_WorldPos;

layout(location = 0) out vec4 o_FragColor;

void main() {
  vec4 control = texture(u_Texture, v_TexCoordAlpha.xy);
  float dotScale = v_TexCoordAlpha.z;

  float lineFade =
      max(0.0, u_GridControl.z * v_TexCoordAlpha.z - (u_GridControl.z - 1.0));

  float alphaBase =
      (control.r * dotScale > u_GridControl.x) ? 1.0
    : (control.g > u_GridControl.y)            ? lineFade
                                               : (0.1 * lineFade);

  // Spotlight falloff (1 near focus, fades outward).
  float spot = 1.0;
  if (u_SpotlightEnabled) {
    float d = distance(v_WorldPos, u_SpotlightFocusPoint);
    spot = 1.0 - smoothstep(u_SpotlightRadius, u_SpotlightRadius * 2.0, d);
  }

  float alpha = alphaBase * dotScale * spot * u_PlaneAlphaScale;

  if (alpha <= 0.001) {
    discard;
  }

  // Keep it subtle and neutral (Filament-like).
  o_FragColor = vec4(alpha);
}
