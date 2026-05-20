/*
 * Renders free drawing as a real 3D triangle-mesh tube instead of GL_LINE.
 * The stroke is rendered into the same virtual scene framebuffer as normal AR objects.
 */
package com.google.ar.core.examples.java.helloar;

import android.opengl.Matrix;

import com.google.ar.core.examples.java.common.samplerender.Framebuffer;
import com.google.ar.core.examples.java.common.samplerender.IndexBuffer;
import com.google.ar.core.examples.java.common.samplerender.Mesh;
import com.google.ar.core.examples.java.common.samplerender.SampleRender;
import com.google.ar.core.examples.java.common.samplerender.Shader;
import com.google.ar.core.examples.java.common.samplerender.VertexBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;

public class TubeObjectDrawingRenderer {

  private static final int DEFAULT_SIDES = 12;
  private static final float DEFAULT_RADIUS_M = 0.006f; // 6 mm

  private final int sides;
  private float radiusMeters;

  private VertexBuffer positionVertexBuffer;
  private VertexBuffer normalVertexBuffer;
  private IndexBuffer indexBuffer;
  private Mesh mesh;
  private Shader shader;

  private final float[] modelMatrix = new float[16];
  private final float[] modelViewMatrix = new float[16];
  private final float[] modelViewProjectionMatrix = new float[16];

  public TubeObjectDrawingRenderer() {
    this(DEFAULT_SIDES, DEFAULT_RADIUS_M);
  }

  public TubeObjectDrawingRenderer(int sides, float radiusMeters) {
    this.sides = Math.max(6, sides);
    this.radiusMeters = radiusMeters;
    Matrix.setIdentityM(modelMatrix, 0);
  }

  public void createOnGlThread(SampleRender render) throws IOException {
    positionVertexBuffer = new VertexBuffer(render, 3, null); // layout(location = 0)
    normalVertexBuffer = new VertexBuffer(render, 3, null);   // layout(location = 1)
    indexBuffer = new IndexBuffer(render, null);

    mesh = new Mesh(
            render,
            Mesh.PrimitiveMode.TRIANGLES,
            indexBuffer,
            new VertexBuffer[]{positionVertexBuffer, normalVertexBuffer}
    );

    shader = Shader.createFromAssets(
                    render,
                    "shaders/tube_object_drawing.vert",
                    "shaders/tube_object_drawing.frag",
                    null)
            .setVec4("u_Color", new float[]{1.0f, 0.12f, 0.08f, 1.0f});
  }

  public void setRadiusMeters(float radiusMeters) {
    this.radiusMeters = radiusMeters;
  }

  public void setColor(float r, float g, float b, float a) {
    if (shader != null) {
      shader.setVec4("u_Color", new float[]{r, g, b, a});
    }
  }

  public void drawStroke(
          SampleRender render,
          List<float[]> localPoints,
          float[] strokeModelMatrix,
          float[] viewMatrix,
          float[] projectionMatrix,
          Framebuffer framebuffer) {

    if (mesh == null || shader == null || localPoints == null || localPoints.size() < 2) return;

    // Build the tube in STROKE-LOCAL coordinates.
    // The ARCore anchor pose is applied through strokeModelMatrix, same as a normal AR object.
    buildTubeMesh(localPoints);

    Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, strokeModelMatrix, 0);
    Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0);

    shader.setMat4("u_Model", strokeModelMatrix);
    shader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);

    render.draw(mesh, shader, framebuffer);
  }

  private void buildTubeMesh(List<float[]> points) {
    int ringCount = points.size();
    int vertexCount = ringCount * sides;
    int indexCount = (ringCount - 1) * sides * 6;

    float[] positions = new float[vertexCount * 3];
    float[] normals = new float[vertexCount * 3];
    int[] indices = new int[indexCount];

    float[] previousNormal = null;

    for (int i = 0; i < ringCount; i++) {
      float[] center = points.get(i);
      float[] tangent = getTangent(points, i);
      normalize(tangent);

      float[] up = new float[]{0f, 1f, 0f};
      if (Math.abs(dot(tangent, up)) > 0.90f) {
        up = new float[]{1f, 0f, 0f};
      }

      float[] ringNormal = cross(tangent, up);
      normalize(ringNormal);

      // Keep ring orientation stable and reduce tube twisting/flickering.
      if (previousNormal != null && dot(ringNormal, previousNormal) < 0f) {
        ringNormal[0] *= -1f;
        ringNormal[1] *= -1f;
        ringNormal[2] *= -1f;
      }
      previousNormal = ringNormal;

      float[] ringBinormal = cross(tangent, ringNormal);
      normalize(ringBinormal);

      for (int j = 0; j < sides; j++) {
        double angle = (2.0 * Math.PI * j) / sides;
        float cos = (float) Math.cos(angle);
        float sin = (float) Math.sin(angle);

        float nx = ringNormal[0] * cos + ringBinormal[0] * sin;
        float ny = ringNormal[1] * cos + ringBinormal[1] * sin;
        float nz = ringNormal[2] * cos + ringBinormal[2] * sin;

        int out = (i * sides + j) * 3;
        positions[out] = center[0] + nx * radiusMeters;
        positions[out + 1] = center[1] + ny * radiusMeters;
        positions[out + 2] = center[2] + nz * radiusMeters;

        normals[out] = nx;
        normals[out + 1] = ny;
        normals[out + 2] = nz;
      }
    }

    int k = 0;
    for (int i = 0; i < ringCount - 1; i++) {
      for (int j = 0; j < sides; j++) {
        int a = i * sides + j;
        int b = i * sides + ((j + 1) % sides);
        int c = (i + 1) * sides + j;
        int d = (i + 1) * sides + ((j + 1) % sides);

        indices[k++] = a;
        indices[k++] = c;
        indices[k++] = b;

        indices[k++] = b;
        indices[k++] = c;
        indices[k++] = d;
      }
    }

    positionVertexBuffer.set(createFloatBuffer(positions));
    normalVertexBuffer.set(createFloatBuffer(normals));
    indexBuffer.set(createIntBuffer(indices));
  }

  private static float[] getTangent(List<float[]> points, int index) {
    int last = points.size() - 1;
    if (index == 0) return subtract(points.get(1), points.get(0));
    if (index == last) return subtract(points.get(last), points.get(last - 1));

    float[] prev = subtract(points.get(index), points.get(index - 1));
    float[] next = subtract(points.get(index + 1), points.get(index));
    return add(prev, next);
  }

  private static FloatBuffer createFloatBuffer(float[] data) {
    ByteBuffer bb = ByteBuffer.allocateDirect(data.length * 4).order(ByteOrder.nativeOrder());
    FloatBuffer fb = bb.asFloatBuffer();
    fb.put(data);
    fb.flip();
    return fb;
  }

  private static IntBuffer createIntBuffer(int[] data) {
    ByteBuffer bb = ByteBuffer.allocateDirect(data.length * 4).order(ByteOrder.nativeOrder());
    IntBuffer ib = bb.asIntBuffer();
    ib.put(data);
    ib.flip();
    return ib;
  }

  private static float[] subtract(float[] a, float[] b) {
    return new float[]{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  }

  private static float[] add(float[] a, float[] b) {
    return new float[]{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
  }

  private static float dot(float[] a, float[] b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  private static float[] cross(float[] a, float[] b) {
    return new float[]{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
    };
  }

  private static void normalize(float[] v) {
    float len = (float) Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < 1e-6f) return;
    v[0] /= len;
    v[1] /= len;
    v[2] /= len;
  }
}
