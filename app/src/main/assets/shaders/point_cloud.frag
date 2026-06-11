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
precision mediump float;

uniform vec4 u_Color;
uniform float u_MinConfidence;

in float v_Confidence;
out vec4 o_FragColor;

void main() {
  // Remove noisy/unstable points.
  if (v_Confidence < u_MinConfidence) {
    discard;
  }

  // Make points circular (instead of square).
  vec2 p = gl_PointCoord - vec2(0.5);
  float r = length(p);

  // Soft edge: inside -> 1, outside -> 0
  float alphaCircle = smoothstep(0.5, 0.35, r);

  // Fade in based on confidence.
  float alphaConf = smoothstep(u_MinConfidence, 1.0, v_Confidence);

  float alpha = u_Color.a * alphaCircle * alphaConf;
  if (alpha <= 0.001) {
    discard;
  }

  o_FragColor = vec4(u_Color.rgb, alpha);
}
