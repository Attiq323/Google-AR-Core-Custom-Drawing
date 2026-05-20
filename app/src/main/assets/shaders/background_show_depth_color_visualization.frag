#version 300 es
/*
 * Shows ARCore raw depth confidence instead of the camera image.
 *
 * NOTE: The app UI option still says "Show depth map". For debugging freehand
 * drawing stability, this shader now visualizes the raw depth confidence image.
 * Black/dark = weak or unavailable raw depth. Bright/white = strong confidence.
 */
precision mediump float;

uniform sampler2D u_CameraDepthConfidenceTexture;
uniform sampler2D u_ColorMap;

in vec2 v_CameraTexCoord;

layout(location = 0) out vec4 o_FragColor;

vec3 Confidence_GetColorVisualization(float confidence) {
  // Turbo palette: low confidence = dark/purple/blue, high confidence = yellow/red/white-ish.
  return texture(u_ColorMap, vec2(clamp(confidence, 0.0, 1.0), 0.5)).rgb;
}

void main() {
  float confidence = texture(u_CameraDepthConfidenceTexture, v_CameraTexCoord).r;

  // Use grayscale if you want the simplest view:
  // o_FragColor = vec4(vec3(confidence), 1.0);

  // Use color palette so weak/strong confidence areas are easier to see.
  vec3 confidenceColor = Confidence_GetColorVisualization(confidence);

  // Invalid/no-confidence pixels remain black.
  confidenceColor *= step(0.003, confidence);
  o_FragColor = vec4(confidenceColor, 1.0);
}
