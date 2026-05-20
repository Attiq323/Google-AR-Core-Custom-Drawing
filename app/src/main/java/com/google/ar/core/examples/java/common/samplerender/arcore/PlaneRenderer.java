/*
 * Copyright 2020 Google LLC
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
package com.google.ar.core.examples.java.common.samplerender.arcore;

import android.opengl.Matrix;

import com.google.ar.core.Camera;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.TrackingState;
import com.google.ar.core.examples.java.common.samplerender.IndexBuffer;
import com.google.ar.core.examples.java.common.samplerender.Mesh;
import com.google.ar.core.examples.java.common.samplerender.SampleRender;
import com.google.ar.core.examples.java.common.samplerender.Shader;
import com.google.ar.core.examples.java.common.samplerender.Shader.BlendFactor;
import com.google.ar.core.examples.java.common.samplerender.Texture;
import com.google.ar.core.examples.java.common.samplerender.VertexBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Renders the detected AR planes.
 */
public class PlaneRenderer {
    private static final String TAG = PlaneRenderer.class.getSimpleName();

    // Shader names.
    private static final String VERTEX_SHADER_NAME = "shaders/plane.vert";
    private static final String FRAGMENT_SHADER_NAME = "shaders/plane.frag";
    private static final String TEXTURE_NAME = "models/trigrid.png";

    private static final int BYTES_PER_FLOAT = Float.SIZE / 8;
    private static final int BYTES_PER_INT = Integer.SIZE / 8;
    private static final int COORDS_PER_VERTEX = 3; // x, z, alpha

    private static final int VERTS_PER_BOUNDARY_VERT = 2;
    private static final int INDICES_PER_BOUNDARY_VERT = 3;
    private static final int INITIAL_BUFFER_BOUNDARY_VERTS = 64;

    private static final int INITIAL_VERTEX_BUFFER_SIZE_BYTES =
            BYTES_PER_FLOAT * COORDS_PER_VERTEX * VERTS_PER_BOUNDARY_VERT * INITIAL_BUFFER_BOUNDARY_VERTS;

    private static final int INITIAL_INDEX_BUFFER_SIZE_BYTES =
            BYTES_PER_INT
                    * INDICES_PER_BOUNDARY_VERT
                    * INDICES_PER_BOUNDARY_VERT
                    * INITIAL_BUFFER_BOUNDARY_VERTS;

    private static final float FADE_RADIUS_M = 0.25f;
    private static final float DOTS_PER_METER = 10.0f;
    // Filament-like differentiation: vertical planes slightly denser and dimmer.
    private static final float DOTS_PER_METER_HORIZONTAL = 10.0f;
    private static final float DOTS_PER_METER_VERTICAL = 14.0f;
    // Spotlight radius for plane visualization (meters).
    private static final float SPOTLIGHT_RADIUS_M = 0.6f;
    private static final float EQUILATERAL_TRIANGLE_SCALE = (float) (1 / Math.sqrt(3));

    // Using the "signed distance field" approach to render sharp lines and circles.
    // {dotThreshold, lineThreshold, lineFadeSpeed, occlusionScale}
    // dotThreshold/lineThreshold: red/green intensity above which dots/lines are present
    // lineFadeShrink:  lines will fade in between alpha = 1-(1/lineFadeShrink) and 1.0
    // occlusionShrink: occluded planes will fade out between alpha = 0 and 1/occlusionShrink
    private static final float[] GRID_CONTROL = {0.2f, 0.4f, 2.0f, 1.5f};

    private final Mesh mesh;
    private final IndexBuffer indexBufferObject;
    private final VertexBuffer vertexBufferObject;
    private final Shader shader;

    private FloatBuffer vertexBuffer =
            ByteBuffer.allocateDirect(INITIAL_VERTEX_BUFFER_SIZE_BYTES)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
    private IntBuffer indexBuffer =
            ByteBuffer.allocateDirect(INITIAL_INDEX_BUFFER_SIZE_BYTES)
                    .order(ByteOrder.nativeOrder())
                    .asIntBuffer();

    // Temporary lists/matrices allocated here to reduce number of allocations for each frame.
    private final float[] viewMatrix = new float[16];
    private final float[] modelMatrix = new float[16];
    private final float[] modelViewMatrix = new float[16];
    private final float[] modelViewProjectionMatrix = new float[16];
    private final float[] planeAngleUvMatrix =
            new float[4]; // 2x2 rotation matrix applied to uv coords.
    private final float[] normalVector = new float[3];

    // Spotlight focus point (world space) used to make plane visualization more realistic.
    private final float[] spotlightFocusPoint = new float[]{0f, 0f, 0f};
    private boolean spotlightEnabled = false;

    // Focused plane type: -1 = none, 0 = horizontal, 1 = vertical
    private int focusPlaneType = -1;
    private static final float FOCUSED_PLANE_ALPHA = 1.0f;
    private static final float UNFOCUSED_PLANE_ALPHA = 0.12f;
    private float spotlightRadiusM = SPOTLIGHT_RADIUS_M;

    private final Map<Plane, Integer> planeIndexMap = new HashMap<>();

    /**
     * Allocates and initializes OpenGL resources needed by the plane renderer. Must be called during
     * a {@link SampleRender.Renderer} callback, typically in {@link
     * SampleRender.Renderer#onSurfaceCreated}.
     */
    public PlaneRenderer(SampleRender render) throws IOException {
        Texture texture =
                Texture.createFromAsset(
                        render, TEXTURE_NAME, Texture.WrapMode.REPEAT, Texture.ColorFormat.LINEAR);
        shader =
                Shader.createFromAssets(render, VERTEX_SHADER_NAME, FRAGMENT_SHADER_NAME, /*defines=*/ null)
                        .setTexture("u_Texture", texture)
                        .setVec4("u_GridControl", GRID_CONTROL)
                        .setBlend(
                                BlendFactor.DST_ALPHA, // RGB (src)
                                BlendFactor.ONE, // RGB (dest)
                                BlendFactor.ZERO, // ALPHA (src)
                                BlendFactor.ONE_MINUS_SRC_ALPHA) // ALPHA (dest)
                        .setDepthWrite(false);

        indexBufferObject = new IndexBuffer(render, /*entries=*/ null);
        vertexBufferObject = new VertexBuffer(render, COORDS_PER_VERTEX, /*entries=*/ null);
        VertexBuffer[] vertexBuffers = {vertexBufferObject};
        mesh = new Mesh(render, Mesh.PrimitiveMode.TRIANGLE_STRIP, indexBufferObject, vertexBuffers);


    }
    /** Enables/disables a Filament-like spotlight around the user's focus point. */
    /**
     * Sets which plane type should be emphasized by the plane visualizer.
     */
    public void setFocusPlaneType(@androidx.annotation.Nullable Plane.Type type) {
        if (type == null) {
            this.focusPlaneType = -1;
        } else if (type == Plane.Type.VERTICAL) {
            this.focusPlaneType = 1;
        } else {
            // Treat all horizontal types (UPWARD/DOWNWARD) as horizontal visualization.
            this.focusPlaneType = 0;
        }
    }

    public void setSpotlight(boolean enabled, float[] focusPointWorld, float radiusMeters) {
        this.spotlightEnabled = enabled;
        if (focusPointWorld != null && focusPointWorld.length >= 3) {
            this.spotlightFocusPoint[0] = focusPointWorld[0];
            this.spotlightFocusPoint[1] = focusPointWorld[1];
            this.spotlightFocusPoint[2] = focusPointWorld[2];
        }
        if (radiusMeters > 0f) {
            this.spotlightRadiusM = radiusMeters;
        }
    }

    /**
     * Updates the plane model transform matrix and extents.
     */
    private void updatePlaneParameters(
            float[] planeMatrix, float extentX, float extentZ, FloatBuffer boundary) {
        System.arraycopy(planeMatrix, 0, modelMatrix, 0, 16);
        if (boundary == null) {
            vertexBuffer.limit(0);
            indexBuffer.limit(0);
            return;
        }

        // Generate a new set of vertices and a corresponding triangle strip index set so that
        // the plane boundary polygon has a fading edge. This is done by making a copy of the
        // boundary polygon vertices and scaling it down around center to push it inwards. Then
        // the index buffer is setup accordingly.
        boundary.rewind();
        int boundaryVertices = boundary.limit() / 2;
        int numVertices;
        int numIndices;

        numVertices = boundaryVertices * VERTS_PER_BOUNDARY_VERT;
        // drawn as GL_TRIANGLE_STRIP with 3n-2 triangles (n-2 for fill, 2n for perimeter).
        numIndices = boundaryVertices * INDICES_PER_BOUNDARY_VERT;

        if (vertexBuffer.capacity() < numVertices * COORDS_PER_VERTEX) {
            int size = vertexBuffer.capacity();
            while (size < numVertices * COORDS_PER_VERTEX) {
                size *= 2;
            }
            vertexBuffer =
                    ByteBuffer.allocateDirect(BYTES_PER_FLOAT * size)
                            .order(ByteOrder.nativeOrder())
                            .asFloatBuffer();
        }
        vertexBuffer.rewind();
        vertexBuffer.limit(numVertices * COORDS_PER_VERTEX);

        if (indexBuffer.capacity() < numIndices) {
            int size = indexBuffer.capacity();
            while (size < numIndices) {
                size *= 2;
            }
            indexBuffer =
                    ByteBuffer.allocateDirect(BYTES_PER_INT * size)
                            .order(ByteOrder.nativeOrder())
                            .asIntBuffer();
        }
        indexBuffer.rewind();
        indexBuffer.limit(numIndices);

        // Note: when either dimension of the bounding box is smaller than 2*FADE_RADIUS_M we
        // generate a bunch of 0-area triangles.  These don't get rendered though so it works
        // out ok.
        float xScale = Math.max((extentX - 2 * FADE_RADIUS_M) / extentX, 0.0f);
        float zScale = Math.max((extentZ - 2 * FADE_RADIUS_M) / extentZ, 0.0f);

        while (boundary.hasRemaining()) {
            float x = boundary.get();
            float z = boundary.get();
            vertexBuffer.put(x);
            vertexBuffer.put(z);
            vertexBuffer.put(0.0f);
            vertexBuffer.put(x * xScale);
            vertexBuffer.put(z * zScale);
            vertexBuffer.put(1.0f);
        }

        // step 1, perimeter
        indexBuffer.put((short) ((boundaryVertices - 1) * 2));
        for (int i = 0; i < boundaryVertices; ++i) {
            indexBuffer.put((short) (i * 2));
            indexBuffer.put((short) (i * 2 + 1));
        }
        indexBuffer.put((short) 1);
        // This leaves us on the interior edge of the perimeter between the inset vertices
        // for boundary verts n-1 and 0.

        // step 2, interior:
        for (int i = 1; i < boundaryVertices / 2; ++i) {
            indexBuffer.put((short) ((boundaryVertices - 1 - i) * 2 + 1));
            indexBuffer.put((short) (i * 2 + 1));
        }
        if (boundaryVertices % 2 != 0) {
            indexBuffer.put((short) ((boundaryVertices / 2) * 2 + 1));
        }
    }

    /**
     * Draws the collection of tracked planes, with closer planes hiding more distant ones.
     *
     * @param allPlanes        The collection of planes to draw.
     * @param cameraPose       The pose of the camera, as returned by {@link Camera#getPose()}
     * @param cameraProjection The projection matrix, as returned by {@link
     *                         Camera#getProjectionMatrix(float[], int, float, float)}
     */
    public void drawPlanes(
            SampleRender render, Collection<Plane> allPlanes, Pose cameraPose, float[] cameraProjection) {
        // Planes must be sorted by distance from camera so that we draw closer planes first, and
        // they occlude the farther planes.
        List<SortablePlane> sortedPlanes = new ArrayList<>();

        for (Plane plane : allPlanes) {
            if (plane.getTrackingState() != TrackingState.TRACKING || plane.getSubsumedBy() != null) {
                continue;
            }

            float distance = calculateDistanceToPlane(plane.getCenterPose(), cameraPose);
            // Plane back-facing test. Use a small tolerance to avoid flicker/culling
            // issues on some devices/orientations (especially for vertical planes).
            if (distance < -0.05f) {
                continue;
            }
            sortedPlanes.add(new SortablePlane(distance, plane));
        }
        Collections.sort(
                sortedPlanes,
                new Comparator<SortablePlane>() {
                    @Override
                    public int compare(SortablePlane a, SortablePlane b) {
                        return Float.compare(b.distance, a.distance);
                    }
                });

        cameraPose.inverse().toMatrix(viewMatrix, 0);

        // Spotlight uniforms are shared across planes.
        shader.setBool("u_SpotlightEnabled", spotlightEnabled);
        shader.setVec3("u_SpotlightFocusPoint", spotlightFocusPoint);
        shader.setFloat("u_SpotlightRadius", spotlightRadiusM);

        for (SortablePlane sortedPlane : sortedPlanes) {
            Plane plane = sortedPlane.plane;
            float[] planeMatrix = new float[16];
            plane.getCenterPose().toMatrix(planeMatrix, 0);

            // Get transformed Y axis of plane's coordinate system.
            plane.getCenterPose().getTransformedAxis(1, 1.0f, normalVector, 0);

            // Offset the plane visualization downward along its normal to sit on the surface
            // instead of floating above it. Use a slightly larger offset.
//            float offsetDown = -0.002f; // -2mm downward along normal
//            planeMatrix[12] += normalVector[0] * offsetDown; // translate X
//            planeMatrix[13] += normalVector[1] * offsetDown; // translate Y
//            planeMatrix[14] += normalVector[2] * offsetDown; // translate Z

            updatePlaneParameters(
                    planeMatrix, plane.getExtentX(), plane.getExtentZ(), plane.getPolygon());

            // Get plane index. Keep a map to assign same indices to same planes.
            Integer planeIndex = planeIndexMap.get(plane);
            if (planeIndex == null) {
                planeIndex = planeIndexMap.size();
                planeIndexMap.put(plane, planeIndex);
            }

            // Each plane will have its own angle offset from others, to make them easier to
            // distinguish. Compute a 2x2 rotation matrix from the angle.

            final boolean isVertical = (plane.getType() == Plane.Type.VERTICAL);
            float angleRadians = isVertical ? 0.0f : (planeIndex * 0.144f);
            float dotsPerMeter = isVertical ? DOTS_PER_METER_VERTICAL : DOTS_PER_METER_HORIZONTAL;
            float uScale = dotsPerMeter;
            float vScale = dotsPerMeter * EQUILATERAL_TRIANGLE_SCALE;
            planeAngleUvMatrix[0] = +(float) Math.cos(angleRadians) * uScale;
            planeAngleUvMatrix[1] = -(float) Math.sin(angleRadians) * vScale;
            planeAngleUvMatrix[2] = +(float) Math.sin(angleRadians) * uScale;
            planeAngleUvMatrix[3] = +(float) Math.cos(angleRadians) * vScale;

            // Build the ModelView and ModelViewProjection matrices
            // for calculating cube position and light.
            Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0);
            Matrix.multiplyMM(modelViewProjectionMatrix, 0, cameraProjection, 0, modelViewMatrix, 0);

            // Populate the shader uniforms for this frame.
            shader.setMat4("u_Model", modelMatrix);
            shader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);
            shader.setMat2("u_PlaneUvMatrix", planeAngleUvMatrix);
            shader.setVec3("u_Normal", normalVector);
            shader.setInt("u_PlaneType", isVertical ? 1 : 0);

// Base intensity by orientation.
            float baseAlpha = isVertical ? 0.75f : 1.0f;

// Emphasize only the focused plane type; keep others faint to avoid "mixing".
            float focusMul;
            if (focusPlaneType == -1) {
                focusMul = 0.35f; // no focus yet: show everything softly
            } else if ((focusPlaneType == 1 && isVertical) || (focusPlaneType == 0 && !isVertical)) {
                focusMul = FOCUSED_PLANE_ALPHA;
            } else {
                focusMul = UNFOCUSED_PLANE_ALPHA;
            }

            shader.setFloat("u_PlaneAlphaScale", baseAlpha * focusMul);

            // Set the position of the plane
            vertexBufferObject.set(vertexBuffer);
            indexBufferObject.set(indexBuffer);

            render.draw(mesh, shader);
        }
    }

    private static class SortablePlane {
        final float distance;
        final Plane plane;

        SortablePlane(float distance, Plane plane) {
            this.distance = distance;
            this.plane = plane;
        }
    }

    // Calculate the normal distance to plane from cameraPose, the given planePose should have y axis
    // parallel to plane's normal, for example plane's center pose or hit test pose.
    public static float calculateDistanceToPlane(Pose planePose, Pose cameraPose) {
        float[] normal = new float[3];
        float cameraX = cameraPose.tx();
        float cameraY = cameraPose.ty();
        float cameraZ = cameraPose.tz();
        // Get transformed Y axis of plane's coordinate system.
        planePose.getTransformedAxis(1, 1.0f, normal, 0);
        // Compute dot product of plane's normal with vector from camera to plane center.
        return (cameraX - planePose.tx()) * normal[0]
                + (cameraY - planePose.ty()) * normal[1]
                + (cameraZ - planePose.tz()) * normal[2];
    }
}
