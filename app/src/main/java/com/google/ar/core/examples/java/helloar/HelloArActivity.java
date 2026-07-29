/*
 * Copyright 2017 Google LLC
 * Licensed under the Apache License, Version 2.0
 */
package com.google.ar.core.examples.java.helloar;

import android.content.DialogInterface;
import android.content.res.Resources;
import android.media.Image;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.PopupMenu;
import android.widget.Toast;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.google.ar.core.Anchor;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.ArCoreApk.Availability;
import com.google.ar.core.Camera;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Config;
import com.google.ar.core.Config.InstantPlacementMode;
import com.google.ar.core.DepthPoint; // ✅ needed
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.InstantPlacementPoint;
import com.google.ar.core.LightEstimate;
import com.google.ar.core.Plane;
import com.google.ar.core.Point;
import com.google.ar.core.PointCloud;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingFailureReason;
import com.google.ar.core.TrackingState;
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper;
import com.google.ar.core.examples.java.common.helpers.DepthSettings;
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper;
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper;
import com.google.ar.core.examples.java.common.helpers.InstantPlacementSettings;
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper;
import com.google.ar.core.examples.java.common.helpers.TapHelper;
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper;
import com.google.ar.core.examples.java.common.samplerender.Framebuffer;
import com.google.ar.core.examples.java.common.samplerender.GLError;
import com.google.ar.core.examples.java.common.samplerender.Mesh;
import com.google.ar.core.examples.java.common.samplerender.SampleRender;
import com.google.ar.core.examples.java.common.samplerender.Shader;
import com.google.ar.core.examples.java.common.samplerender.Shader.BlendFactor;
import com.google.ar.core.examples.java.common.samplerender.Texture;
import com.google.ar.core.examples.java.common.samplerender.VertexBuffer;
import com.google.ar.core.examples.java.common.samplerender.arcore.BackgroundRenderer;
import com.google.ar.core.examples.java.common.samplerender.arcore.PlaneRenderer;
import com.google.ar.core.examples.java.common.samplerender.arcore.SpecularCubemapFilter;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;

public class HelloArActivity extends AppCompatActivity implements SampleRender.Renderer {

    private static final String TAG = HelloArActivity.class.getSimpleName();

    private static final String SEARCHING_PLANE_MESSAGE = "Searching for surfaces...";
    private static final String WAITING_FOR_TAP_MESSAGE = "Tap on a surface to place an object.";

    private static final float[] sphericalHarmonicFactors = {
            0.282095f, -0.325735f, 0.325735f, -0.325735f, 0.273137f, -0.273137f, 0.078848f, -0.273137f, 0.136569f,
    };

    private static final float Z_NEAR = 0.1f;
    private static final float Z_FAR = 100f;

    private static final int CUBEMAP_RESOLUTION = 16;
    private static final int CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES = 32;

    private GLSurfaceView surfaceView;
    private boolean installRequested;

    private Session session;
    private final SnackbarHelper messageSnackbarHelper = new SnackbarHelper();
    private DisplayRotationHelper displayRotationHelper;
    private final TrackingStateHelper trackingStateHelper = new TrackingStateHelper(this);
    private TapHelper tapHelper;
    private SampleRender render;

    // Plane visualization focus
    private Plane.Type focusPlaneTypeStable = null;
    private Plane.Type focusPlaneTypeCandidate = null;
    private int focusTypeStableFrames = 0;
    private static final int FOCUS_TYPE_STABILIZE_FRAMES = 8;

    private final float[] focusPointWorld = new float[3];
    private boolean hasFocusPoint = false;
    private float lastFocusHitDistance = 3.0f; // meters
    private static final float PLANE_SPOTLIGHT_RADIUS_M = 1.0f;

    private PlaneRenderer planeRenderer;
    private final float[] planeSpotlightFocusPoint = new float[]{0f, 0f, 0f};
    private float lastPlaneHitDistanceM = 3.0f;

    private BackgroundRenderer backgroundRenderer;
    private Framebuffer virtualSceneFramebuffer;
    private boolean hasSetTextureNames = false;

    private final DepthSettings depthSettings = new DepthSettings();
    private boolean[] depthSettingsMenuDialogCheckboxes = new boolean[2];

    private final InstantPlacementSettings instantPlacementSettings = new InstantPlacementSettings();
    private boolean[] instantPlacementSettingsMenuDialogCheckboxes = new boolean[1];

    private static final float APPROXIMATE_DISTANCE_METERS = 2.0f;

    // Point Cloud
    private VertexBuffer pointCloudVertexBuffer;
    private Mesh pointCloudMesh;
    private Shader pointCloudShader;
    private long lastPointCloudTimestamp = 0;

    // Virtual object (Arrow)
    private Mesh virtualObjectMesh;
    private Shader virtualObjectShader;
    private Texture virtualObjectAlbedoTexture;
    private Texture virtualObjectAlbedoInstantPlacementTexture;

    private final List<WrappedAnchor> wrappedAnchors = new ArrayList<>();

    // Environmental HDR
    private Texture dfgTexture;
    private SpecularCubemapFilter cubemapFilter;

    // Interaction
    private enum InteractionMode {PLACE_PAWN, DRAW_LINE}
    public enum UserRole {SUPERVISOR, TECHNICIAN}

    private InteractionMode currentMode = InteractionMode.PLACE_PAWN;
    private UserRole currentUserRole = UserRole.SUPERVISOR;
    private Button roleButton;

    private static final float[] SUPERVISOR_COLOR = new float[]{0.05f, 0.22f, 0.78f, 1.0f};
    private static final float[] TECHNICIAN_COLOR = new float[]{0.10f, 0.72f, 0.24f, 1.0f};

    // Drawing lines
    private VertexBuffer drawLineVertexBuffer;
    private Mesh drawLineMesh;
    private Shader drawLineShader;

    // New object-style drawing renderer.
    // This renders drawing as a real 3D tube mesh object instead of GL_LINE.
    private TubeObjectDrawingRenderer tubeObjectDrawingRenderer;

    private static class StrokeSegment {
        // Each segment has its own anchor. Keeping points close to their segment anchor
        // reduces drift on curved/large surfaces compared with one anchor for a whole stroke.
        Anchor anchor;
        final List<float[]> localPoints = new ArrayList<>();
    }

    private static class Stroke {
        UserRole owner;
        final List<StrokeSegment> segments = new ArrayList<>();
        StrokeSegment activeSegment;

        // Live stroke can be rough while finger is moving.
        // When user releases finger, we rebuild this stroke as one stable smoothed tube.
        boolean finalizedSmoothTube = false;
    }

    private final List<Stroke> drawStrokes = new ArrayList<>();
    private Stroke currentStroke = null;

    // Debug overlay for checking whether depth drawing is selecting the front object surface
    // or jumping to a farther/background depth cluster. This is intentionally UI-only and
    // does not change drawing placement.
    private static final int DEPTH_DEBUG_MAX_ROWS = 8;
    private final ArrayList<String> depthDebugRows = new ArrayList<>();
    private int depthDebugSampleIndex = 0;
    private Float lastDebugFinalDistanceM = null;
    private RawDepthSample lastRawDepthSample = null;

    private static class RawDepthSample {
        float depthMeters;
        float minDepthMeters;
        float maxDepthMeters;
        float avgConfidence;
        int validCount;

        float depthRangeMeters() {
            return maxDepthMeters - minDepthMeters;
        }
    }

    /**
     * Backend-only mini surface patch used when ARCore has no Plane.
     *
     * It is not an ARCore Plane trackable. It is a small local plane estimated once from the
     * dense/full depth neighborhood around the touch point. We immediately create an Anchor from
     * this pose and then render drawing/arrow from that anchor, so old content is not recalculated
     * from noisy depth pixels every frame.
     */
    private static class MiniSurfacePatch {
        Pose pose;
        float[] centerWorld;
        float[] normalWorld;
        int validDepthCount;
        float minDepthMeters;
        float maxDepthMeters;
        boolean rawConfidenceGood;
    }

    private static final float DRAW_MIN_ANCHOR_DISTANCE_M = 0.010f; // 1 cm between accepted drawing samples
    // Fade-fix test: was 0.0015f. Larger offset keeps plane drawings ahead of depth map after revisit.
    private static final float DRAW_SURFACE_OFFSET_M = 0.012f; // 12 mm
    private static final float DRAW_MAX_BAD_DEPTH_JUMP_M = 0.12f; // reject sudden raw-depth/background jumps
    private static final float DRAW_MAX_SEGMENT_DISTANCE_M = 0.07f; // create a new local anchor every ~7 cm for curved/non-plane surfaces
    private static final int DRAW_INTERP_SEGMENTS = 1;

    // Final-stroke smoothing. This matches the reference video behavior:
    // during draw = rough preview, after ACTION_UP = clean stable tube.
    private static final int FINAL_STROKE_SMOOTHING_ITERATIONS = 2;
    private static final float FINAL_STROKE_SIMPLIFY_TOLERANCE_M = 0.006f;
    private static final float FINAL_STROKE_MIN_POINT_DISTANCE_M = 0.006f;
    private static final float FINAL_STROKE_MAX_KEEP_JUMP_M = 0.22f;

    private static final float HIT_MAX_DISTANCE_M = 8.0f;
    private static final float HIT_MIN_DISTANCE_M = 0.10f;

    // Raw-depth drawing point selection. This follows the Raw Depth codelab idea:
    // use raw depth + confidence, ignore invalid/low-confidence pixels, and use a local patch.
    private static final int RAW_DEPTH_PATCH_RADIUS_PX = 3; // 7x7 patch
    private static final int RAW_DEPTH_MIN_VALID_SAMPLES = 5;
    private static final float RAW_DEPTH_MIN_CONFIDENCE = 0.35f;

    // Backend-only custom surface patches for non-plane/curved surfaces.
    // These are small hidden local planes estimated once from the dense/full depth image.
    private static final int MINI_PATCH_RADIUS_PX = 4;
    private static final int MINI_PATCH_MIN_VALID_SAMPLES = 8;
    private static final float MINI_PATCH_MAX_DEPTH_RANGE_M = 0.18f;
    private static final float MINI_PATCH_MAX_CENTER_TO_MEDIAN_M = 0.08f;

    // Draw as a thin object/decal. Big tube radius and large surface offset look like floating geometry.
    private static final float DRAW_BASE_RADIUS_M = 0.0035f;
    private static final float DRAW_MIN_RADIUS_M = 0.0038f;
    // Fade-fix test: was 0.0090f. Slightly thicker at distance so occlusion blur eats less of the stroke.
    private static final float DRAW_MAX_RADIUS_M = 0.018f;

    // Matrices
    private final float[] modelMatrix = new float[16];
    private final float[] viewMatrix = new float[16];
    private final float[] projectionMatrix = new float[16];
    private final float[] modelViewMatrix = new float[16];
    private final float[] modelViewProjectionMatrix = new float[16];
    private final float[] sphericalHarmonicsCoefficients = new float[9 * 3];
    private final float[] viewInverseMatrix = new float[16];
    private final float[] worldLightDirection = {0.0f, 0.0f, 0.0f, 0.0f};
    private final float[] viewLightDirection = new float[4];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        surfaceView = findViewById(R.id.surfaceview);

        displayRotationHelper = new DisplayRotationHelper(this);

        tapHelper = new TapHelper(this);
        surfaceView.setOnTouchListener(tapHelper);

        render = new SampleRender(surfaceView, this, getAssets());
        installRequested = false;

        depthSettings.onCreate(this);
        instantPlacementSettings.onCreate(this);

        ImageButton settingsButton = findViewById(R.id.settings_button);
        settingsButton.setOnClickListener(v -> {
            PopupMenu popup = new PopupMenu(HelloArActivity.this, v);
            popup.setOnMenuItemClickListener(HelloArActivity.this::settingsMenuClick);
            popup.inflate(R.menu.settings_menu);
            popup.show();
        });

        Button modeButton = findViewById(R.id.mode_button);
        updateModeButtonUi(modeButton);
        modeButton.setOnClickListener(v -> {
            currentMode = (currentMode == InteractionMode.PLACE_PAWN)
                    ? InteractionMode.DRAW_LINE
                    : InteractionMode.PLACE_PAWN;
            updateModeButtonUi(modeButton);
        });

        roleButton = findViewById(R.id.role_button);
        updateRoleButtonUi(roleButton);
        roleButton.setOnClickListener(v -> {
            currentUserRole = (currentUserRole == UserRole.SUPERVISOR)
                    ? UserRole.TECHNICIAN
                    : UserRole.SUPERVISOR;
            updateRoleButtonUi(roleButton);
        });
    }

    protected boolean settingsMenuClick(MenuItem item) {
        if (item.getItemId() == R.id.depth_settings) {
            launchDepthSettingsMenuDialog();
            return true;
        } else if (item.getItemId() == R.id.instant_placement_settings) {
            launchInstantPlacementSettingsMenuDialog();
            return true;
        }
        return false;
    }

    @Override
    protected void onDestroy() {
        for (Stroke stroke : drawStrokes) {
            for (StrokeSegment segment : stroke.segments) {
                if (segment.anchor != null) {
                    segment.anchor.detach();
                }
            }
        }
        drawStrokes.clear();

        if (session != null) {
            session.close();
            session = null;
        }
        super.onDestroy();
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (session == null) {
            Exception exception = null;
            String message = null;
            try {
                Availability availability = ArCoreApk.getInstance().checkAvailability(this);
                if (availability != Availability.SUPPORTED_INSTALLED) {
                    switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                        case INSTALL_REQUESTED:
                            installRequested = true;
                            return;
                        case INSTALLED:
                            break;
                    }
                }

                if (!CameraPermissionHelper.hasCameraPermission(this)) {
                    CameraPermissionHelper.requestCameraPermission(this);
                    return;
                }

                session = new Session(this);

            } catch (UnavailableArcoreNotInstalledException |
                     UnavailableUserDeclinedInstallationException e) {
                message = "Please install ARCore";
                exception = e;
            } catch (UnavailableApkTooOldException e) {
                message = "Please update ARCore";
                exception = e;
            } catch (UnavailableSdkTooOldException e) {
                message = "Please update this app";
                exception = e;
            } catch (UnavailableDeviceNotCompatibleException e) {
                message = "This device does not support AR";
                exception = e;
            } catch (Exception e) {
                message = "Failed to create AR session";
                exception = e;
            }

            if (message != null) {
                messageSnackbarHelper.showError(this, message);
                Log.e(TAG, "Exception creating session", exception);
                return;
            }
        }

        try {
            configureSession();
            session.resume();
        } catch (CameraNotAvailableException e) {
            messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
            session = null;
            return;
        }

        surfaceView.onResume();
        displayRotationHelper.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (session != null) {
            displayRotationHelper.onPause();
            surfaceView.onPause();
            session.pause();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
            Toast.makeText(this, "Camera permission is needed to run this application", Toast.LENGTH_LONG).show();
            if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
                CameraPermissionHelper.launchPermissionSettings(this);
            }
            finish();
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus);
    }

    @Override
    public void onSurfaceCreated(SampleRender render) {
        try {
            planeRenderer = new PlaneRenderer(render);
            backgroundRenderer = new BackgroundRenderer(render);
            virtualSceneFramebuffer = new Framebuffer(render, 1, 1);

            cubemapFilter = new SpecularCubemapFilter(render, CUBEMAP_RESOLUTION, CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES);

            dfgTexture = new Texture(render, Texture.Target.TEXTURE_2D, Texture.WrapMode.CLAMP_TO_EDGE, false);
            final int dfgResolution = 64;
            final int dfgChannels = 2;
            final int halfFloatSize = 2;

            ByteBuffer buffer = ByteBuffer.allocateDirect(dfgResolution * dfgResolution * dfgChannels * halfFloatSize);
            try (InputStream is = getAssets().open("models/dfg.raw")) {
                is.read(buffer.array());
            }

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, dfgTexture.getTextureId());
            GLError.maybeThrowGLException("Failed to bind DFG texture", "glBindTexture");
            GLES30.glTexImage2D(
                    GLES30.GL_TEXTURE_2D,
                    0,
                    GLES30.GL_RG16F,
                    dfgResolution,
                    dfgResolution,
                    0,
                    GLES30.GL_RG,
                    GLES30.GL_HALF_FLOAT,
                    buffer
            );
            GLError.maybeThrowGLException("Failed to populate DFG texture", "glTexImage2D");

            // point cloud
            pointCloudShader =
                    Shader.createFromAssets(render, "shaders/point_cloud.vert", "shaders/point_cloud.frag", null)
                            .setVec4("u_Color", new float[]{1.0f, 1.0f, 1.0f, 1.0f})
                            .setFloat("u_PointSize", 8.0f)
                            .setFloat("u_MinConfidence", 0.30f)
                            .setBlend(
                                    BlendFactor.SRC_ALPHA, BlendFactor.ONE_MINUS_SRC_ALPHA,
                                    BlendFactor.ONE, BlendFactor.ONE_MINUS_SRC_ALPHA)
                            .setDepthWrite(false);

            pointCloudVertexBuffer = new VertexBuffer(render, 4, null);
            pointCloudMesh = new Mesh(render, Mesh.PrimitiveMode.POINTS, null, new VertexBuffer[]{pointCloudVertexBuffer});

            // OLD GL_LINE drawing renderer.
            // Commented because the new drawing should behave like a 3D object, not a GL_LINE.
//      drawLineVertexBuffer = new VertexBuffer(render, 3, null);
//      drawLineMesh = new Mesh(render, Mesh.PrimitiveMode.LINE_STRIP, null, new VertexBuffer[]{drawLineVertexBuffer});
//      drawLineShader =
//              Shader.createFromAssets(render, "shaders/line.vert", "shaders/line.frag", null)
//                      .setVec4("u_Color", new float[]{1.0f, 0.2f, 0.2f, 1.0f});

            // NEW object-style drawing renderer.
            // It creates a triangle tube mesh from drawing anchors and renders it into
            // virtualSceneFramebuffer, same as normal virtual objects.
            tubeObjectDrawingRenderer = new TubeObjectDrawingRenderer();
            tubeObjectDrawingRenderer.createOnGlThread(render);
            tubeObjectDrawingRenderer.setRadiusMeters(DRAW_MIN_RADIUS_M);
            tubeObjectDrawingRenderer.setColor(SUPERVISOR_COLOR[0], SUPERVISOR_COLOR[1], SUPERVISOR_COLOR[2], SUPERVISOR_COLOR[3]);

            // arrow assets
            virtualObjectAlbedoTexture =
                    Texture.createFromAsset(render, "models/arrow_albedo.png", Texture.WrapMode.CLAMP_TO_EDGE, Texture.ColorFormat.SRGB);
            virtualObjectAlbedoInstantPlacementTexture =
                    Texture.createFromAsset(render, "models/arrow_albedo.png", Texture.WrapMode.CLAMP_TO_EDGE, Texture.ColorFormat.SRGB);
            Texture virtualObjectPbrTexture =
                    Texture.createFromAsset(render, "models/arrow_albedo.png", Texture.WrapMode.CLAMP_TO_EDGE, Texture.ColorFormat.LINEAR);

            virtualObjectMesh = Mesh.createFromAsset(render, "models/arrow.obj");
            virtualObjectShader =
                    Shader.createFromAssets(
                                    render,
                                    "shaders/environmental_hdr.vert",
                                    "shaders/environmental_hdr.frag",
                                    new HashMap<String, String>() {{
                                        put("NUMBER_OF_MIPMAP_LEVELS", Integer.toString(cubemapFilter.getNumberOfMipmapLevels()));
                                    }})
                            .setTexture("u_AlbedoTexture", virtualObjectAlbedoTexture)
                            .setTexture("u_RoughnessMetallicAmbientOcclusionTexture", virtualObjectPbrTexture)
                            .setTexture("u_Cubemap", cubemapFilter.getFilteredCubemapTexture())
                            .setTexture("u_DfgTexture", dfgTexture)
                            .setVec4("u_TintColor", new float[]{1.0f, 1.0f, 1.0f, 1.0f});

        } catch (IOException e) {
            Log.e(TAG, "Failed to read a required asset file", e);
            messageSnackbarHelper.showError(this, "Failed to read a required asset file: " + e);
        }
    }

    @Override
    public void onSurfaceChanged(SampleRender render, int width, int height) {
        displayRotationHelper.onSurfaceChanged(width, height);
        virtualSceneFramebuffer.resize(width, height);
    }

    @Override
    public void onDrawFrame(SampleRender render) {
        if (session == null) return;

        if (!hasSetTextureNames) {
            session.setCameraTextureNames(new int[]{backgroundRenderer.getCameraColorTexture().getTextureId()});
            hasSetTextureNames = true;
        }

        displayRotationHelper.updateSessionIfNeeded(session);

        Frame frame;
        try {
            frame = session.update();
        } catch (CameraNotAvailableException e) {
            Log.e(TAG, "Camera not available during onDrawFrame", e);
            messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
            return;
        }

        Camera camera = frame.getCamera();

        // ✅ IMPORTANT: keep occlusion tied to the setting (you had 'false' here, which disables it)
        try {
            backgroundRenderer.setUseDepthVisualization(render, depthSettings.depthColorVisualizationEnabled());
            backgroundRenderer.setUseOcclusion(render, depthSettings.useDepthForOcclusion());
        } catch (IOException e) {
            Log.e(TAG, "Failed to read a required asset file", e);
            messageSnackbarHelper.showError(this, "Failed to read a required asset file: " + e);
            return;
        }

        backgroundRenderer.updateDisplayGeometry(frame);

        // Update depth resources.
        // Occlusion still needs the normal packed depth image.
        if (camera.getTrackingState() == TrackingState.TRACKING
                && depthSettings.useDepthForOcclusion()) {
            try (Image depthImage = frame.acquireDepthImage16Bits()) {
                backgroundRenderer.updateCameraDepthTexture(depthImage);
            } catch (NotYetAvailableException e) {
                // Depth not ready yet - ok.
            }
        }

        // The existing UI option still says "Show depth map", but for debugging drawing stability
        // we now visualize the raw depth confidence image instead of the depth color map.
        // This does not replace the depth texture used for occlusion.
        if (camera.getTrackingState() == TrackingState.TRACKING
                && depthSettings.depthColorVisualizationEnabled()) {
            try (Image confidenceImage = frame.acquireRawDepthConfidenceImage()) {
                backgroundRenderer.updateCameraDepthConfidenceTexture(confidenceImage);
            } catch (NotYetAvailableException e) {
                // Raw depth confidence not ready yet - ok.
            }
        }

        handleTap(frame, camera);

        trackingStateHelper.updateKeepScreenOnFlag(camera.getTrackingState());

        String message = null;
        if (camera.getTrackingState() == TrackingState.PAUSED) {
            if (camera.getTrackingFailureReason() == TrackingFailureReason.NONE) {
                message = SEARCHING_PLANE_MESSAGE;
            } else {
                message = TrackingStateHelper.getTrackingFailureReasonString(camera);
            }
        } else if (hasTrackingPlane()) {
            if (wrappedAnchors.isEmpty()) message = WAITING_FOR_TAP_MESSAGE;
        } else {
            message = SEARCHING_PLANE_MESSAGE;
        }

        if (message == null) messageSnackbarHelper.hide(this);
        else messageSnackbarHelper.showMessage(this, message);

        if (frame.getTimestamp() != 0) {
            backgroundRenderer.drawBackground(render);
        }

        if (camera.getTrackingState() == TrackingState.PAUSED) return;

        camera.getProjectionMatrix(projectionMatrix, 0, Z_NEAR, Z_FAR);
        camera.getViewMatrix(viewMatrix, 0);

        // point cloud
        try (PointCloud pointCloud = frame.acquirePointCloud()) {
            if (pointCloud.getTimestamp() > lastPointCloudTimestamp) {
                pointCloudVertexBuffer.set(pointCloud.getPoints());
                lastPointCloudTimestamp = pointCloud.getTimestamp();
            }
            Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0);
            pointCloudShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);
            render.draw(pointCloudMesh, pointCloudShader);
        }

        // planes
        planeRenderer.drawPlanes(
                render,
                session.getAllTrackables(Plane.class),
                camera.getDisplayOrientedPose(),
                projectionMatrix
        );

        // lighting
        updateLightEstimation(frame.getLightEstimate(), viewMatrix);

        // draw virtual objects into framebuffer
        render.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f);

        // Draw strokes FIRST.
        // Important: the tube drawing is now real triangle geometry and writes color/depth.
        // If it is rendered after the arrow, it can overlap the arrow and make the arrow look deformed.
        // Rendering strokes before arrows keeps arrow shape stable, while both still behave as 3D objects.
        if (!drawStrokes.isEmpty()) {

            // OLD GL_LINE rendering.
            // This is commented because GL_LINE/depth can cut/fade the stroke like pixels.
//      Matrix.setIdentityM(modelMatrix, 0);
//      Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0);
//      Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0);
//      drawLineShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);
//
//      GLES30.glLineWidth(32.0f);
//
//      for (Stroke stroke : drawStrokes) {
//        List<float[]> points = buildInterpolatedStrokePoints(stroke);
//        if (points.size() < 2) continue;
//        updateDrawLineVertexBuffer(points);
//        render.draw(drawLineMesh, drawLineShader, virtualSceneFramebuffer);
//      }

            // NEW object-style tube drawing.
            // Every segment is rendered like a normal AR object: local mesh + latest anchor model matrix.
            for (Stroke stroke : drawStrokes) {
                for (StrokeSegment segment : stroke.segments) {
                    if (segment.anchor == null
                            || segment.anchor.getTrackingState() != TrackingState.TRACKING) {
                        continue;
                    }

                    List<float[]> localPoints = buildInterpolatedStrokePoints(segment);
                    if (localPoints.size() < 2) continue;

                    segment.anchor.getPose().toMatrix(modelMatrix, 0);

                    float strokeDistance = distanceCameraToPoseM(camera.getPose(), segment.anchor.getPose());
                    float drawingRadius = DRAW_BASE_RADIUS_M * strokeDistance;
                    drawingRadius = Math.max(DRAW_MIN_RADIUS_M, Math.min(drawingRadius, DRAW_MAX_RADIUS_M));
                    tubeObjectDrawingRenderer.setRadiusMeters(drawingRadius);
                    float[] strokeColor = getRoleColor(stroke.owner);
                    tubeObjectDrawingRenderer.setColor(strokeColor[0], strokeColor[1], strokeColor[2], strokeColor[3]);

                    tubeObjectDrawingRenderer.drawStroke(
                            render,
                            localPoints,
                            modelMatrix,
                            viewMatrix,
                            projectionMatrix,
                            virtualSceneFramebuffer
                    );
                }
            }
        }

        // Draw arrows AFTER strokes.
        // This prevents drawing tube triangles from visually covering/distorting the arrow shape.
        for (WrappedAnchor wrappedAnchor : wrappedAnchors) {
            Anchor anchor = wrappedAnchor.getAnchor();
            Trackable trackable = wrappedAnchor.getTrackable();
            if (anchor.getTrackingState() != TrackingState.TRACKING) continue;

            anchor.getPose().toMatrix(modelMatrix, 0);
            float distance = distanceCameraToPoseM(camera.getPose(), anchor.getPose());

            float baseScale = 0.035f;
            float scale = baseScale * distance;
            scale = Math.max(0.08f, Math.min(scale, 0.16f));

            Matrix.scaleM(modelMatrix, 0, scale, scale, scale);
//      Matrix.scaleM(modelMatrix, 0, 0.05f, 0.05f, 0.05f);

            Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0);
            Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0);

            virtualObjectShader.setMat4("u_ModelView", modelViewMatrix);
            virtualObjectShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);

            if (trackable instanceof InstantPlacementPoint
                    && ((InstantPlacementPoint) trackable).getTrackingMethod()
                    == InstantPlacementPoint.TrackingMethod.SCREENSPACE_WITH_APPROXIMATE_DISTANCE) {
                virtualObjectShader.setTexture("u_AlbedoTexture", virtualObjectAlbedoInstantPlacementTexture);
            } else {
                virtualObjectShader.setTexture("u_AlbedoTexture", virtualObjectAlbedoTexture);
            }
            float[] arrowTint = getRoleColor(wrappedAnchor.getOwner());
            virtualObjectShader.setVec4("u_TintColor", arrowTint);

            render.draw(virtualObjectMesh, virtualObjectShader, virtualSceneFramebuffer);
        }

        backgroundRenderer.drawVirtualScene(render, virtualSceneFramebuffer, Z_NEAR, Z_FAR);
    }

    private void handleTap(Frame frame, Camera camera) {

        // PLACE mode:
        // 1) Use normal ARCore Plane behavior when a Plane exists.
        // 2) If there is no Plane (curved chair/machine/engine surface), build a hidden mini
        //    surface patch from the dense/full depth neighborhood and create a persistent Anchor.
        // 3) Fallback to DepthPoint/feature Point only if the mini patch is unavailable.
        if (currentMode == InteractionMode.PLACE_PAWN) {
            MotionEvent tap = tapHelper.poll();
            if (tap == null || camera.getTrackingState() != TrackingState.TRACKING) return;

            if (wrappedAnchors.size() >= 20) {
                wrappedAnchors.get(0).getAnchor().detach();
                wrappedAnchors.remove(0);
            }

            Anchor anchor = null;
            Trackable trackable = null;

            HitResult planeHit = pickSurfaceHitPlaneFirst(frame, camera, tap.getX(), tap.getY());
            if (planeHit != null && planeHit.getTrackable() instanceof Plane) {
                anchor = createSurfaceAnchor(planeHit, camera);
                trackable = planeHit.getTrackable();
            } else {
                MiniSurfacePatch patch = tryBuildMiniSurfacePatch(frame, camera, tap.getX(), tap.getY(), null, null);
                if (patch != null) {
                    anchor = createMiniPatchAnchor(patch);
                    trackable = null; // Custom backend patch, not an ARCore Trackable.
                    addMiniPatchDebugSample(camera, patch, "arrow");
                }

                if (anchor == null) {
                    HitResult fallbackHit = pickSurfaceHitDepthFirst(frame, camera, tap.getX(), tap.getY());
                    if (fallbackHit != null) {
                        anchor = createSurfaceAnchor(fallbackHit, camera);
                        trackable = fallbackHit.getTrackable();
                    }
                }
            }

            if (anchor != null) {
                wrappedAnchors.add(new WrappedAnchor(anchor, trackable, currentUserRole));
                runOnUiThread(this::showOcclusionDialogIfNeeded);
            }
        }

        // DRAW mode:
        // Plane drawing remains unchanged. For non-plane/curved surfaces, a backend-only mini
        // local surface patch is created from full depth and used exactly like a small hidden plane:
        // create anchor once -> store local stroke points -> never update old points from depth frames.
        if (currentMode == InteractionMode.DRAW_LINE && camera.getTrackingState() == TrackingState.TRACKING) {
            Image rawDepthImage = null;
            Image rawDepthConfidenceImage = null;

            try {
                rawDepthImage = frame.acquireRawDepthImage16Bits();
                rawDepthConfidenceImage = frame.acquireRawDepthConfidenceImage();
            } catch (NotYetAvailableException e) {
                // Optional debug/quality images are not ready. Mini patches still use full depth.
            }

            MotionEvent event;
            while ((event = tapHelper.pollMoveEvent()) != null) {
                int action = event.getActionMasked();

                if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
                    if (action == MotionEvent.ACTION_UP) {
                        finalizeCurrentStrokeAsSmoothTube();
                    }
                    currentStroke = null;
                    lastDebugFinalDistanceM = null;
                    event.recycle();
                    continue;
                }

                // Raw depth + confidence are sampled for analysis/quality only. They do not move
                // already-created stroke points.
                if (rawDepthImage != null && rawDepthConfidenceImage != null) {
                    getStableRawDepthWorldPoint(
                            frame,
                            camera,
                            rawDepthImage,
                            rawDepthConfidenceImage,
                            event.getX(),
                            event.getY());
                } else {
                    lastRawDepthSample = null;
                }

                HitResult planeHit = pickSurfaceHitPlaneFirst(frame, camera, event.getX(), event.getY());
                boolean usingRealPlane = planeHit != null && planeHit.getTrackable() instanceof Plane;

                MiniSurfacePatch miniPatch = null;
                HitResult surfaceHit = null;
                Pose stablePose = null;

                if (usingRealPlane) {
                    surfaceHit = planeHit;
                    stablePose = offsetAlongSurfaceNormal(surfaceHit, camera, DRAW_SURFACE_OFFSET_M);
                } else {
                    miniPatch = tryBuildMiniSurfacePatch(
                            frame,
                            camera,
                            event.getX(),
                            event.getY(),
                            rawDepthImage,
                            rawDepthConfidenceImage);
                    if (miniPatch != null) {
                        stablePose = miniPatch.pose;
                    } else {
                        surfaceHit = pickSurfaceHitDepthFirst(frame, camera, event.getX(), event.getY());
                        if (surfaceHit != null) {
                            stablePose = offsetAlongSurfaceNormal(surfaceHit, camera, DRAW_SURFACE_OFFSET_M);
                        }
                    }
                }

                if (stablePose == null) {
                    event.recycle();
                    continue;
                }

                float[] worldPoint = poseToWorldPoint(stablePose);

                if (action == MotionEvent.ACTION_DOWN || currentStroke == null) {
                    resetDepthDebugRows();
                    currentStroke = new Stroke();
                    currentStroke.owner = currentUserRole;
                    drawStrokes.add(currentStroke);
                    currentStroke.activeSegment = createStrokeSegmentFromSurface(surfaceHit, miniPatch, camera);
                    if (currentStroke.activeSegment != null) {
                        currentStroke.segments.add(currentStroke.activeSegment);
                    }
                }

                if (currentStroke == null || currentStroke.activeSegment == null) {
                    event.recycle();
                    continue;
                }

                StrokeSegment segment = currentStroke.activeSegment;
                if (segment.anchor == null || segment.anchor.getTrackingState() != TrackingState.TRACKING) {
                    event.recycle();
                    continue;
                }

                // Keep local coordinates small by starting a new stable patch/anchor every few cm.
                // This approximates curved surfaces as many hidden local planes.
                if (shouldStartNewSegment(segment, worldPoint)) {
                    StrokeSegment newSegment = createStrokeSegmentFromSurface(surfaceHit, miniPatch, camera);
                    if (newSegment != null) {
                        float[] previousWorldPoint = getLastWorldPoint(segment);
                        if (previousWorldPoint != null) {
                            newSegment.localPoints.add(worldToSegmentLocal(newSegment, previousWorldPoint));
                        }
                        currentStroke.segments.add(newSegment);
                        currentStroke.activeSegment = newSegment;
                        segment = newSegment;
                    }
                }

                if (shouldAddStrokePoint(segment, worldPoint)) {
                    List<float[]> worldPoints = getSegmentWorldControlPoints(segment);
                    worldPoint = stabilizeDrawPoint(worldPoint, worldPoints);
                    segment.localPoints.add(worldToSegmentLocal(segment, worldPoint));
                    addDepthDebugSample(camera, worldPoint, lastRawDepthSample, currentStroke, segment);
                }

                event.recycle();
            }

            if (rawDepthImage != null) rawDepthImage.close();
            if (rawDepthConfidenceImage != null) rawDepthConfidenceImage.close();
        }
    }


    /**
     * Commit step for drawing, same idea as the reference video:
     * live points may look rough while the user is moving, but when the finger is released
     * we freeze all already-calculated world points, remove obvious depth spikes, smooth them,
     * and rebuild the stroke as one stable tube mesh anchored in AR world space.
     *
     * Important: this method does not sample depth again. Depth/plane/feature points are used
     * only while creating points. After ACTION_UP, final rendering uses saved world coordinates.
     */
    private void finalizeCurrentStrokeAsSmoothTube() {
        if (currentStroke == null || currentStroke.finalizedSmoothTube) return;

        List<float[]> frozenWorldPoints = collectStrokeWorldPoints(currentStroke);
        if (frozenWorldPoints.size() < 2) return;

        List<float[]> cleaned = cleanStrokeWorldPoints(frozenWorldPoints);
        if (cleaned.size() < 2) return;

        List<float[]> simplified = simplifyStrokeWorldPoints(cleaned, FINAL_STROKE_SIMPLIFY_TOLERANCE_M);
        if (simplified.size() < 2) simplified = cleaned;

        List<float[]> smoothed = chaikinSmoothWorldPoints(simplified, FINAL_STROKE_SMOOTHING_ITERATIONS);
        smoothed = cleanStrokeWorldPoints(smoothed);
        if (smoothed.size() < 2) return;

        replaceStrokeWithSingleStableSegment(currentStroke, smoothed);
        currentStroke.finalizedSmoothTube = true;
    }

    private List<float[]> collectStrokeWorldPoints(Stroke stroke) {
        ArrayList<float[]> worldPoints = new ArrayList<>();
        if (stroke == null) return worldPoints;

        for (StrokeSegment segment : stroke.segments) {
            if (segment == null || segment.anchor == null) continue;
            if (segment.anchor.getTrackingState() != TrackingState.TRACKING) continue;

            for (float[] localPoint : segment.localPoints) {
                float[] world = segmentLocalToWorld(segment, localPoint);
                if (world == null) continue;

                // Avoid duplicate seam point where a new live segment started from previous point.
                if (!worldPoints.isEmpty()) {
                    float[] last = worldPoints.get(worldPoints.size() - 1);
                    if (distanceBetweenPointsM(last, world) < FINAL_STROKE_MIN_POINT_DISTANCE_M * 0.5f) {
                        continue;
                    }
                }
                worldPoints.add(world);
            }
        }
        return worldPoints;
    }

    private List<float[]> cleanStrokeWorldPoints(List<float[]> input) {
        ArrayList<float[]> output = new ArrayList<>();
        if (input == null || input.isEmpty()) return output;

        float[] previousAccepted = null;
        for (float[] p : input) {
            if (p == null) continue;
            if (previousAccepted == null) {
                output.add(p.clone());
                previousAccepted = p;
                continue;
            }

            float d = distanceBetweenPointsM(previousAccepted, p);
            if (d < FINAL_STROKE_MIN_POINT_DISTANCE_M) {
                continue;
            }

            // If one raw depth sample jumped to background, do not keep it in final tube.
            if (d > FINAL_STROKE_MAX_KEEP_JUMP_M) {
                continue;
            }

            output.add(p.clone());
            previousAccepted = p;
        }
        return output;
    }

    private List<float[]> simplifyStrokeWorldPoints(List<float[]> points, float toleranceM) {
        ArrayList<float[]> result = new ArrayList<>();
        if (points == null || points.isEmpty()) return result;
        if (points.size() <= 2 || toleranceM <= 0f) {
            for (float[] p : points) result.add(p.clone());
            return result;
        }

        boolean[] keep = new boolean[points.size()];
        keep[0] = true;
        keep[points.size() - 1] = true;
        simplifyStrokeRdp(points, 0, points.size() - 1, toleranceM, keep);

        for (int i = 0; i < points.size(); i++) {
            if (keep[i]) result.add(points.get(i).clone());
        }
        return result;
    }

    private void simplifyStrokeRdp(List<float[]> points, int start, int end, float toleranceM, boolean[] keep) {
        if (end <= start + 1) return;

        float maxDistance = -1f;
        int index = -1;
        float[] a = points.get(start);
        float[] b = points.get(end);

        for (int i = start + 1; i < end; i++) {
            float d = distancePointToSegmentM(points.get(i), a, b);
            if (d > maxDistance) {
                maxDistance = d;
                index = i;
            }
        }

        if (index >= 0 && maxDistance > toleranceM) {
            keep[index] = true;
            simplifyStrokeRdp(points, start, index, toleranceM, keep);
            simplifyStrokeRdp(points, index, end, toleranceM, keep);
        }
    }

    private float distancePointToSegmentM(float[] p, float[] a, float[] b) {
        float abx = b[0] - a[0];
        float aby = b[1] - a[1];
        float abz = b[2] - a[2];
        float apx = p[0] - a[0];
        float apy = p[1] - a[1];
        float apz = p[2] - a[2];

        float abLen2 = abx * abx + aby * aby + abz * abz;
        if (abLen2 < 1e-8f) return distanceBetweenPointsM(p, a);

        float t = (apx * abx + apy * aby + apz * abz) / abLen2;
        t = Math.max(0f, Math.min(1f, t));

        float cx = a[0] + abx * t;
        float cy = a[1] + aby * t;
        float cz = a[2] + abz * t;
        return distanceBetweenPointsM(p, new float[]{cx, cy, cz});
    }

    private List<float[]> chaikinSmoothWorldPoints(List<float[]> input, int iterations) {
        ArrayList<float[]> current = new ArrayList<>();
        if (input == null) return current;
        for (float[] p : input) current.add(p.clone());

        for (int iter = 0; iter < iterations; iter++) {
            if (current.size() < 3) break;

            ArrayList<float[]> next = new ArrayList<>();
            next.add(current.get(0).clone()); // Preserve start exactly where user started.

            for (int i = 0; i < current.size() - 1; i++) {
                float[] p0 = current.get(i);
                float[] p1 = current.get(i + 1);

                // Chaikin corner cutting: removes zigzag while keeping the path on saved world points.
                next.add(new float[]{
                        0.75f * p0[0] + 0.25f * p1[0],
                        0.75f * p0[1] + 0.25f * p1[1],
                        0.75f * p0[2] + 0.25f * p1[2]
                });
                next.add(new float[]{
                        0.25f * p0[0] + 0.75f * p1[0],
                        0.25f * p0[1] + 0.75f * p1[1],
                        0.25f * p0[2] + 0.75f * p1[2]
                });
            }

            next.add(current.get(current.size() - 1).clone()); // Preserve end exactly where user stopped.
            current = next;
        }
        return current;
    }

    private void replaceStrokeWithSingleStableSegment(Stroke stroke, List<float[]> finalWorldPoints) {
        if (session == null || stroke == null || finalWorldPoints == null || finalWorldPoints.size() < 2) return;

        float[] anchorWorld = finalWorldPoints.get(0);
        Anchor finalAnchor;
        try {
            finalAnchor = session.createAnchor(Pose.makeTranslation(anchorWorld[0], anchorWorld[1], anchorWorld[2]));
        } catch (Exception e) {
            Log.w(TAG, "Unable to create final smooth stroke anchor", e);
            return;
        }

        // Detach old live preview anchors after the final stable anchor is ready.
        for (StrokeSegment oldSegment : stroke.segments) {
            if (oldSegment != null && oldSegment.anchor != null) {
                oldSegment.anchor.detach();
            }
        }
        stroke.segments.clear();

        StrokeSegment finalSegment = new StrokeSegment();
        finalSegment.anchor = finalAnchor;
        for (float[] worldPoint : finalWorldPoints) {
            finalSegment.localPoints.add(worldToSegmentLocal(finalSegment, worldPoint));
        }

        stroke.segments.add(finalSegment);
        stroke.activeSegment = finalSegment;
    }

    private void resetDepthDebugRows() {
        depthDebugRows.clear();
        depthDebugSampleIndex = 0;
        lastDebugFinalDistanceM = null;
    }

    private void addDepthDebugSample(Camera camera, float[] finalWorldPoint, RawDepthSample rawSample,
                                     Stroke stroke, StrokeSegment segment) {
        if (camera == null || finalWorldPoint == null || rawSample == null) return;

        float finalDistanceM = distanceCameraToPointM(camera.getPose(), finalWorldPoint);
        float deltaM = lastDebugFinalDistanceM == null ? 0f : finalDistanceM - lastDebugFinalDistanceM;
        lastDebugFinalDistanceM = finalDistanceM;

        int segmentIndex = stroke == null ? 0 : stroke.segments.indexOf(segment) + 1;
        int pointCount = segment == null ? 0 : segment.localPoints.size();

        boolean mixedDepth = rawSample.depthRangeMeters() > 0.06f;
        boolean bigStep = Math.abs(deltaM) > 0.05f;
        String flag;
        if (mixedDepth && bigStep) {
            flag = "MIXED+JUMP";
        } else if (mixedDepth) {
            flag = "MIXED";
        } else if (bigStep) {
            flag = "JUMP";
        } else {
            flag = "OK";
        }

        String row = String.format(Locale.US,
                "%02d S%d P%d raw %.3fm final %.3fm Δ%+.3fm range %.3fm conf %.2f n%d %s",
                ++depthDebugSampleIndex,
                Math.max(segmentIndex, 0),
                pointCount,
                rawSample.depthMeters,
                finalDistanceM,
                deltaM,
                rawSample.depthRangeMeters(),
                rawSample.avgConfidence,
                rawSample.validCount,
                flag);

        Log.d(TAG, "DepthDebug " + row);
        depthDebugRows.add(0, row);
        while (depthDebugRows.size() > DEPTH_DEBUG_MAX_ROWS) {
            depthDebugRows.remove(depthDebugRows.size() - 1);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Depth point debug\n");
        sb.append("range>0.06m = likely object/background mix\n");
        for (String r : depthDebugRows) {
            sb.append(r).append('\n');
        }

    }



    private float[] stabilizeDrawPoint(float[] newPoint, List<float[]> points) {
        final float DRAW_ALPHA_SLOW = 0.18f;
        final float DRAW_ALPHA_FAST = 0.65f;

        final float NORMAL_MOVE_M = 0.06f;
        final float MAX_BAD_JUMP_M = 0.25f;
        if (points == null || points.isEmpty()) {
            return newPoint;
        }

        float[] last = points.get(points.size() - 1);

        float dx = newPoint[0] - last[0];
        float dy = newPoint[1] - last[1];
        float dz = newPoint[2] - last[2];

        float dist = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);

        // Very large jump is probably wrong depth
        if (dist > MAX_BAD_JUMP_M) {
            return last.clone();
        }

        float alpha;

        if (dist > NORMAL_MOVE_M) {
            // fast finger movement, follow more quickly
            alpha = DRAW_ALPHA_FAST;
        } else {
            // slow movement, smooth more
            alpha = DRAW_ALPHA_SLOW;
        }

        return new float[]{
                last[0] * (1f - alpha) + newPoint[0] * alpha,
                last[1] * (1f - alpha) + newPoint[1] * alpha,
                last[2] * (1f - alpha) + newPoint[2] * alpha
        };
    }

    private void updateModeButtonUi(Button modeButton) {
        modeButton.setText(currentMode == InteractionMode.PLACE_PAWN ? "Place" : "Draw");
    }

    private void updateRoleButtonUi(Button roleButton) {
        if (roleButton == null) return;
        roleButton.setText(currentUserRole == UserRole.SUPERVISOR ? "Supervisor" : "Technician");
    }

    private float[] getRoleColor(UserRole role) {
        return role == UserRole.TECHNICIAN ? TECHNICIAN_COLOR : SUPERVISOR_COLOR;
    }

    private void updateDrawLineVertexBuffer(List<float[]> strokePoints) {
        if (strokePoints == null || strokePoints.isEmpty()) return;

        int count = strokePoints.size();
        ByteBuffer bb = ByteBuffer.allocateDirect(count * 3 * 4).order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        for (float[] p : strokePoints) {
            fb.put(p[0]);
            fb.put(p[1]);
            fb.put(p[2]);
        }
        fb.flip();
        drawLineVertexBuffer.set(fb);
    }

    /**
     * Surface picker for AR drawing/placement.
     * <p>
     * Priority:
     * 1) DepthPoint  -> best for real objects/sides such as cup, chair, machine surface
     * 2) Plane       -> fallback for floor/table/wall planes
     * 3) Point       -> fallback feature point
     */
    private HitResult pickSurfaceHitDepthFirst(Frame frame, Camera camera, float xPx, float yPx) {
        if (camera.getTrackingState() != TrackingState.TRACKING) return null;

        List<HitResult> hits = frame.hitTest(xPx, yPx);

        HitResult bestDepthHit = null;
        HitResult bestPlaneHit = null;
        HitResult bestPointHit = null;

        final float MIN_DIST_M = 0.05f;
        final float MAX_DIST_M = 6.00f;

        for (HitResult h : hits) {
            Trackable t = h.getTrackable();

            float dist = distanceCameraToPoseM(camera.getPose(), h.getHitPose());
            if (dist < MIN_DIST_M || dist > MAX_DIST_M) continue;

            // 1) Depth hit first. This makes drawing/arrow follow real object surfaces,
            // not only detected planes.
//            if (t instanceof DepthPoint) {
//                if (bestDepthHit == null ||
//                        distanceCameraToPoseM(camera.getPose(), h.getHitPose()) <
//                                distanceCameraToPoseM(camera.getPose(), bestDepthHit.getHitPose())) {
//                    bestDepthHit = h;
//                }
            if (bestDepthHit == null && t instanceof DepthPoint) {

                bestDepthHit = h;
//        break;
            }

            // 2) Plane fallback.
            if (bestPlaneHit == null && t instanceof Plane) {
                Plane plane = (Plane) t;
                if (plane.getTrackingState() != TrackingState.TRACKING) continue;
                if (!plane.isPoseInPolygon(h.getHitPose())) continue;
                bestPlaneHit = h;
            }

            // 3) Feature point fallback.
            if (bestPointHit == null && t instanceof Point) {
                Point p = (Point) t;
                if (p.getTrackingState() != TrackingState.TRACKING) continue;
                bestPointHit = h;
            }
        }

        if (bestDepthHit != null) return bestDepthHit;
        if (bestPlaneHit != null) return bestPlaneHit;
        return bestPointHit;
    }

    private HitResult pickSurfaceHitPlaneFirst(Frame frame, Camera camera, float xPx, float yPx) {
        if (camera.getTrackingState() != TrackingState.TRACKING) return null;

        List<HitResult> hits = frame.hitTest(xPx, yPx);

        HitResult bestPlaneHit = null;
        HitResult bestDepthHit = null;
        HitResult bestPointHit = null;

        final float MIN_DIST_M = 0.05f;
        final float MAX_DIST_M = 6.00f;

        for (HitResult h : hits) {
            Trackable t = h.getTrackable();

            float dist = distanceCameraToPoseM(camera.getPose(), h.getHitPose());
            if (dist < MIN_DIST_M || dist > MAX_DIST_M) continue;

            if (bestPlaneHit == null && t instanceof Plane) {
                Plane plane = (Plane) t;
                if (plane.getTrackingState() == TrackingState.TRACKING
                        && plane.isPoseInPolygon(h.getHitPose())) {
                    bestPlaneHit = h;
                }
            }

            if (bestDepthHit == null && t instanceof DepthPoint) {
                bestDepthHit = h;

            }

            if (bestPointHit == null && t instanceof Point) {
                Point p = (Point) t;
                if (p.getTrackingState() == TrackingState.TRACKING) {
                    bestPointHit = h;
                }
            }
        }

        if (bestPlaneHit != null) return bestPlaneHit;
        if (bestDepthHit != null) return bestDepthHit;
        return bestPointHit;
    }

    /**
     * Builds one backend-only mini surface patch from the full depth image around a screen point.
     *
     * This is our custom "small plane" for non-plane/curved surfaces. It is intentionally created
     * once at input time and immediately converted into an Anchor. We do not keep updating old
     * drawing/arrow content from new depth frames, which is what gives plane-like stability.
     */
    private MiniSurfacePatch tryBuildMiniSurfacePatch(
            Frame frame,
            Camera camera,
            float screenX,
            float screenY,
            Image optionalRawDepthImage,
            Image optionalRawDepthConfidenceImage) {

        if (frame == null || camera == null || camera.getTrackingState() != TrackingState.TRACKING) return null;

        Image depthImage = null;
        boolean closeDepthImage = false;
        try {
            depthImage = frame.acquireDepthImage16Bits();
            closeDepthImage = true;
        } catch (NotYetAvailableException e) {
            return null;
        } catch (Exception e) {
            return null;
        }

        try {
            float[] viewCoordinates = new float[]{screenX, screenY};
            float[] textureCoordinates = new float[2];
            try {
                frame.transformCoordinates2d(
                        Coordinates2d.VIEW,
                        viewCoordinates,
                        Coordinates2d.TEXTURE_NORMALIZED,
                        textureCoordinates);
            } catch (Exception e) {
                return null;
            }

            float u = textureCoordinates[0];
            float v = textureCoordinates[1];
            if (Float.isNaN(u) || Float.isNaN(v) || u < 0f || u > 1f || v < 0f || v > 1f) return null;

            int width = depthImage.getWidth();
            int height = depthImage.getHeight();
            int centerX = Math.max(0, Math.min(width - 1, (int) (u * width)));
            int centerY = Math.max(0, Math.min(height - 1, (int) (v * height)));

            ArrayList<float[]> points = new ArrayList<>();
            ArrayList<Float> depths = new ArrayList<>();
            float minDepth = Float.MAX_VALUE;
            float maxDepth = 0f;

            for (int y = centerY - MINI_PATCH_RADIUS_PX; y <= centerY + MINI_PATCH_RADIUS_PX; y++) {
                if (y < 0 || y >= height) continue;
                for (int x = centerX - MINI_PATCH_RADIUS_PX; x <= centerX + MINI_PATCH_RADIUS_PX; x++) {
                    if (x < 0 || x >= width) continue;
                    int depthMm = readDepthMillimeters(depthImage, x, y);
                    if (depthMm <= 0) continue;
                    float depthM = depthMm / 1000.0f;
                    if (depthM < HIT_MIN_DISTANCE_M || depthM > HIT_MAX_DISTANCE_M) continue;

                    float[] world = unprojectDepthPixelToWorld(camera, depthImage, x, y, depthM);
                    if (world == null) continue;
                    points.add(world);
                    depths.add(depthM);
                    minDepth = Math.min(minDepth, depthM);
                    maxDepth = Math.max(maxDepth, depthM);
                }
            }

            if (points.size() < MINI_PATCH_MIN_VALID_SAMPLES) return null;
            if ((maxDepth - minDepth) > MINI_PATCH_MAX_DEPTH_RANGE_M) return null;

            Collections.sort(depths);
            float medianDepth = depths.get(depths.size() / 2);
            int centerDepthMm = readDepthMillimeters(depthImage, centerX, centerY);
            if (centerDepthMm <= 0) return null;
            float centerDepthM = centerDepthMm / 1000.0f;
            if (Math.abs(centerDepthM - medianDepth) > MINI_PATCH_MAX_CENTER_TO_MEDIAN_M) return null;

            float[] centerWorld = averageWorldPoint(points);
            float[] normal = estimateMiniPatchNormal(camera, depthImage, centerX, centerY, centerDepthM, centerWorld, points);
            if (normal == null) return null;
            orientNormalTowardCamera(normal, centerWorld, camera);

            float[] offsetCenter = new float[]{
                    centerWorld[0] + normal[0] * DRAW_SURFACE_OFFSET_M,
                    centerWorld[1] + normal[1] * DRAW_SURFACE_OFFSET_M,
                    centerWorld[2] + normal[2] * DRAW_SURFACE_OFFSET_M
            };

            MiniSurfacePatch patch = new MiniSurfacePatch();
            patch.centerWorld = offsetCenter;
            patch.normalWorld = normal;
            patch.validDepthCount = points.size();
            patch.minDepthMeters = minDepth;
            patch.maxDepthMeters = maxDepth;

            if (optionalRawDepthImage != null && optionalRawDepthConfidenceImage != null) {
                getStableRawDepthWorldPoint(frame, camera, optionalRawDepthImage, optionalRawDepthConfidenceImage, screenX, screenY);
                patch.rawConfidenceGood = lastRawDepthSample != null
                        && lastRawDepthSample.avgConfidence >= RAW_DEPTH_MIN_CONFIDENCE;
            }

            patch.pose = makePoseFromPositionAndNormal(offsetCenter, normal, camera);
            return patch;
        } finally {
            if (closeDepthImage && depthImage != null) depthImage.close();
        }
    }

    private int readDepthMillimeters(Image depthImage, int x, int y) {
        if (depthImage == null || x < 0 || y < 0 || x >= depthImage.getWidth() || y >= depthImage.getHeight()) return 0;
        Image.Plane plane = depthImage.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer().duplicate().order(ByteOrder.LITTLE_ENDIAN);
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        if (pixelStride <= 0) pixelStride = 2;
        int index = y * rowStride + x * pixelStride;
        if (index < 0 || index + 1 >= buffer.capacity()) return 0;
        return buffer.getShort(index) & 0xFFFF;
    }

    private float[] unprojectDepthPixelToWorld(Camera camera, Image depthImage, int x, int y, float depthMeters) {
        try {
            CameraIntrinsics intrinsics = camera.getTextureIntrinsics();
            int[] intrinsicsDimensions = intrinsics.getImageDimensions();
            int depthWidth = depthImage.getWidth();
            int depthHeight = depthImage.getHeight();
            float fx = intrinsics.getFocalLength()[0] * depthWidth / intrinsicsDimensions[0];
            float fy = intrinsics.getFocalLength()[1] * depthHeight / intrinsicsDimensions[1];
            float cx = intrinsics.getPrincipalPoint()[0] * depthWidth / intrinsicsDimensions[0];
            float cy = intrinsics.getPrincipalPoint()[1] * depthHeight / intrinsicsDimensions[1];

            float[] pointCamera = new float[]{
                    depthMeters * (x - cx) / fx,
                    depthMeters * (cy - y) / fy,
                    -depthMeters,
                    1f
            };
            float[] pointWorld4 = new float[4];
            float[] cameraPoseMatrix = new float[16];
            camera.getPose().toMatrix(cameraPoseMatrix, 0);
            Matrix.multiplyMV(pointWorld4, 0, cameraPoseMatrix, 0, pointCamera, 0);
            return new float[]{pointWorld4[0], pointWorld4[1], pointWorld4[2]};
        } catch (Exception e) {
            return null;
        }
    }

    private float[] averageWorldPoint(List<float[]> points) {
        float[] avg = new float[]{0f, 0f, 0f};
        if (points == null || points.isEmpty()) return avg;
        for (float[] p : points) {
            avg[0] += p[0];
            avg[1] += p[1];
            avg[2] += p[2];
        }
        avg[0] /= points.size();
        avg[1] /= points.size();
        avg[2] /= points.size();
        return avg;
    }

    private float[] estimateMiniPatchNormal(Camera camera, Image depthImage, int cx, int cy, float centerDepthM,
                                            float[] centerWorld, List<float[]> fallbackPoints) {
        int r = Math.max(2, MINI_PATCH_RADIUS_PX);
        float[] left = depthWorldAtOrNear(camera, depthImage, cx - r, cy, cx, cy);
        float[] right = depthWorldAtOrNear(camera, depthImage, cx + r, cy, cx, cy);
        float[] up = depthWorldAtOrNear(camera, depthImage, cx, cy - r, cx, cy);
        float[] down = depthWorldAtOrNear(camera, depthImage, cx, cy + r, cx, cy);

        if (left != null && right != null && up != null && down != null) {
            float[] horizontal = subtract3(right, left);
            float[] vertical = subtract3(down, up);
            float[] normal = cross3(horizontal, vertical);
            if (length3(normal) > 1e-5f) {
                normalize3(normal);
                return normal;
            }
        }

        // Fallback: use two far-enough patch points to estimate a local normal.
        if (fallbackPoints != null && fallbackPoints.size() >= 3) {
            float[] a = fallbackPoints.get(0);
            float[] b = fallbackPoints.get(fallbackPoints.size() / 2);
            float[] c = fallbackPoints.get(fallbackPoints.size() - 1);
            float[] normal = cross3(subtract3(b, a), subtract3(c, a));
            if (length3(normal) > 1e-5f) {
                normalize3(normal);
                return normal;
            }
        }

        // Last fallback: face camera. Position still uses full depth; orientation is best effort.
        float[] camT = new float[3];
        camera.getPose().getTranslation(camT, 0);
        float[] normal = new float[]{camT[0] - centerWorld[0], camT[1] - centerWorld[1], camT[2] - centerWorld[2]};
        normalize3(normal);
        return normal;
    }

    private float[] depthWorldAtOrNear(Camera camera, Image depthImage, int x, int y, int centerX, int centerY) {
        int width = depthImage.getWidth();
        int height = depthImage.getHeight();
        for (int radius = 0; radius <= 2; radius++) {
            for (int yy = y - radius; yy <= y + radius; yy++) {
                if (yy < 0 || yy >= height) continue;
                for (int xx = x - radius; xx <= x + radius; xx++) {
                    if (xx < 0 || xx >= width) continue;
                    int depthMm = readDepthMillimeters(depthImage, xx, yy);
                    if (depthMm <= 0) continue;
                    float depthM = depthMm / 1000.0f;
                    if (depthM < HIT_MIN_DISTANCE_M || depthM > HIT_MAX_DISTANCE_M) continue;
                    return unprojectDepthPixelToWorld(camera, depthImage, xx, yy, depthM);
                }
            }
        }
        return null;
    }

    private float[] subtract3(float[] a, float[] b) {
        return new float[]{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    }

    private float[] cross3(float[] a, float[] b) {
        return new float[]{
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
        };
    }

    private float length3(float[] v) {
        return (float) Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    private void orientNormalTowardCamera(float[] normal, float[] worldPoint, Camera camera) {
        float[] camT = new float[3];
        camera.getPose().getTranslation(camT, 0);
        float[] toCamera = new float[]{camT[0] - worldPoint[0], camT[1] - worldPoint[1], camT[2] - worldPoint[2]};
        normalize3(toCamera);
        if (normal[0] * toCamera[0] + normal[1] * toCamera[1] + normal[2] * toCamera[2] < 0f) {
            normal[0] *= -1f;
            normal[1] *= -1f;
            normal[2] *= -1f;
        }
    }

    private Pose makePoseFromPositionAndNormal(float[] position, float[] normal, Camera camera) {
        float[] zAxis = new float[]{normal[0], normal[1], normal[2]};
        normalize3(zAxis);

        float[] up = new float[]{0f, 1f, 0f};
        if (Math.abs(zAxis[1]) > 0.92f) {
            up = new float[]{1f, 0f, 0f};
        }

        float[] xAxis = cross3(up, zAxis);
        normalize3(xAxis);
        float[] yAxis = cross3(zAxis, xAxis);
        normalize3(yAxis);

        float[] q = quaternionFromRotationAxes(xAxis, yAxis, zAxis);
        return new Pose(position, q);
    }

    private float[] quaternionFromRotationAxes(float[] x, float[] y, float[] z) {
        // Column-major 3x3 rotation matrix with x/y/z as local axes in world space.
        float m00 = x[0], m01 = y[0], m02 = z[0];
        float m10 = x[1], m11 = y[1], m12 = z[1];
        float m20 = x[2], m21 = y[2], m22 = z[2];

        float qw, qx, qy, qz;
        float trace = m00 + m11 + m22;
        if (trace > 0f) {
            float s = (float) Math.sqrt(trace + 1.0f) * 2f;
            qw = 0.25f * s;
            qx = (m21 - m12) / s;
            qy = (m02 - m20) / s;
            qz = (m10 - m01) / s;
        } else if (m00 > m11 && m00 > m22) {
            float s = (float) Math.sqrt(1.0f + m00 - m11 - m22) * 2f;
            qw = (m21 - m12) / s;
            qx = 0.25f * s;
            qy = (m01 + m10) / s;
            qz = (m02 + m20) / s;
        } else if (m11 > m22) {
            float s = (float) Math.sqrt(1.0f + m11 - m00 - m22) * 2f;
            qw = (m02 - m20) / s;
            qx = (m01 + m10) / s;
            qy = 0.25f * s;
            qz = (m12 + m21) / s;
        } else {
            float s = (float) Math.sqrt(1.0f + m22 - m00 - m11) * 2f;
            qw = (m10 - m01) / s;
            qx = (m02 + m20) / s;
            qy = (m12 + m21) / s;
            qz = 0.25f * s;
        }
        return new float[]{qx, qy, qz, qw};
    }

    private Anchor createMiniPatchAnchor(MiniSurfacePatch patch) {
        if (session == null || patch == null || patch.pose == null) return null;
        try {
            return session.createAnchor(patch.pose);
        } catch (Exception e) {
            return null;
        }
    }

    private StrokeSegment createStrokeSegmentFromSurface(HitResult hit, MiniSurfacePatch patch, Camera camera) {
        Anchor anchor = null;
        if (patch != null) {
            anchor = createMiniPatchAnchor(patch);
        }
        if (anchor == null && hit != null) {
            anchor = createSurfaceAnchor(hit, camera);
        }
        if (anchor == null) return null;
        StrokeSegment segment = new StrokeSegment();
        segment.anchor = anchor;
        return segment;
    }

    private void addMiniPatchDebugSample(Camera camera, MiniSurfacePatch patch, String label) {
        if (camera == null || patch == null || patch.centerWorld == null) return;
        float dist = distanceCameraToPointM(camera.getPose(), patch.centerWorld);
        final String row = String.format(
                Locale.US,
                "%s miniPatch %.2fm samples=%d range=%.3fm rawConf=%s",
                label,
                dist,
                patch.validDepthCount,
                patch.maxDepthMeters - patch.minDepthMeters,
                patch.rawConfidenceGood ? "ok" : "n/a/low");
        depthDebugRows.add(0, row);
        while (depthDebugRows.size() > DEPTH_DEBUG_MAX_ROWS) depthDebugRows.remove(depthDebugRows.size() - 1);

    }

    /**
     * Returns a stable world point from raw depth + confidence near the finger.
     * This follows the Raw Depth codelab approach: raw depth is geometrically more accurate,
     * confidence is used to reject weak pixels, and a small patch/median reduces single-pixel noise.
     */
    private float[] getStableRawDepthWorldPoint(
            Frame frame,
            Camera camera,
            Image rawDepthImage,
            Image confidenceImage,
            float screenX,
            float screenY) {

        lastRawDepthSample = null;
        if (rawDepthImage == null || confidenceImage == null) return null;

        float[] viewCoordinates = new float[]{screenX, screenY};
        float[] textureCoordinates = new float[2];
        try {
            frame.transformCoordinates2d(
                    Coordinates2d.VIEW,
                    viewCoordinates,
                    Coordinates2d.TEXTURE_NORMALIZED,
                    textureCoordinates);
        } catch (Exception e) {
            return null;
        }

        float u = textureCoordinates[0];
        float v = textureCoordinates[1];
        if (Float.isNaN(u) || Float.isNaN(v) || u < 0f || u > 1f || v < 0f || v > 1f) {
            return null;
        }

        int depthWidth = rawDepthImage.getWidth();
        int depthHeight = rawDepthImage.getHeight();
        int depthX = Math.max(0, Math.min(depthWidth - 1, (int) (u * depthWidth)));
        int depthY = Math.max(0, Math.min(depthHeight - 1, (int) (v * depthHeight)));

        RawDepthSample sample = sampleRawDepth(rawDepthImage, confidenceImage, depthX, depthY);
        if (sample == null) return null;
        float depthMeters = sample.depthMeters;
        if (depthMeters < HIT_MIN_DISTANCE_M || depthMeters > HIT_MAX_DISTANCE_M) return null;
        lastRawDepthSample = sample;

        CameraIntrinsics intrinsics = camera.getTextureIntrinsics();
        int[] intrinsicsDimensions = intrinsics.getImageDimensions();
        float fx = intrinsics.getFocalLength()[0] * depthWidth / intrinsicsDimensions[0];
        float fy = intrinsics.getFocalLength()[1] * depthHeight / intrinsicsDimensions[1];
        float cx = intrinsics.getPrincipalPoint()[0] * depthWidth / intrinsicsDimensions[0];
        float cy = intrinsics.getPrincipalPoint()[1] * depthHeight / intrinsicsDimensions[1];

        // Unproject depth pixel into camera coordinates. Matches the Raw Depth codelab:
        // X = depth * (x - cx) / fx, Y = depth * (cy - y) / fy, Z = -depth.
        float[] pointCamera = new float[]{
                depthMeters * (depthX - cx) / fx,
                depthMeters * (cy - depthY) / fy,
                -depthMeters,
                1f
        };
        float[] pointWorld4 = new float[4];
        float[] cameraPoseMatrix = new float[16];
        camera.getPose().toMatrix(cameraPoseMatrix, 0);
        Matrix.multiplyMV(pointWorld4, 0, cameraPoseMatrix, 0, pointCamera, 0);

        return new float[]{pointWorld4[0], pointWorld4[1], pointWorld4[2]};
    }

    private RawDepthSample sampleRawDepth(Image rawDepthImage, Image confidenceImage, int centerX, int centerY) {
        Image.Plane depthPlane = rawDepthImage.getPlanes()[0];
        ByteBuffer depthBuffer = depthPlane.getBuffer().duplicate().order(ByteOrder.LITTLE_ENDIAN);
        int depthRowStride = depthPlane.getRowStride();
        int depthPixelStride = depthPlane.getPixelStride();
        if (depthPixelStride <= 0) depthPixelStride = 2;

        Image.Plane confidencePlane = confidenceImage.getPlanes()[0];
        ByteBuffer confidenceBuffer = confidencePlane.getBuffer().duplicate();
        int confidenceRowStride = confidencePlane.getRowStride();
        int confidencePixelStride = confidencePlane.getPixelStride();
        if (confidencePixelStride <= 0) confidencePixelStride = 1;

        int width = rawDepthImage.getWidth();
        int height = rawDepthImage.getHeight();
        ArrayList<Float> validDepths = new ArrayList<>();
        float minDepth = Float.MAX_VALUE;
        float maxDepth = 0f;
        float confSum = 0f;

        for (int y = centerY - RAW_DEPTH_PATCH_RADIUS_PX; y <= centerY + RAW_DEPTH_PATCH_RADIUS_PX; y++) {
            if (y < 0 || y >= height) continue;
            for (int x = centerX - RAW_DEPTH_PATCH_RADIUS_PX; x <= centerX + RAW_DEPTH_PATCH_RADIUS_PX; x++) {
                if (x < 0 || x >= width) continue;

                int depthIndex = y * depthRowStride + x * depthPixelStride;
                if (depthIndex < 0 || depthIndex + 1 >= depthBuffer.capacity()) continue;

                int depthMillimeters = depthBuffer.getShort(depthIndex) & 0xFFFF;
                if (depthMillimeters <= 0) continue;

                int confidenceIndex = y * confidenceRowStride + x * confidencePixelStride;
                if (confidenceIndex < 0 || confidenceIndex >= confidenceBuffer.capacity()) continue;

                int confidenceValue = confidenceBuffer.get(confidenceIndex) & 0xFF;
                float confidenceNormalized = confidenceValue / 255.0f;
                if (confidenceNormalized < RAW_DEPTH_MIN_CONFIDENCE) continue;

                float depthM = depthMillimeters / 1000.0f;
                validDepths.add(depthM);
                minDepth = Math.min(minDepth, depthM);
                maxDepth = Math.max(maxDepth, depthM);
                confSum += confidenceNormalized;
            }
        }

        if (validDepths.size() < RAW_DEPTH_MIN_VALID_SAMPLES) return null;
        Collections.sort(validDepths);

        RawDepthSample sample = new RawDepthSample();
        sample.depthMeters = validDepths.get(validDepths.size() / 2);
        sample.minDepthMeters = minDepth;
        sample.maxDepthMeters = maxDepth;
        sample.validCount = validDepths.size();
        sample.avgConfidence = confSum / validDepths.size();
        return sample;
    }

    private float[] offsetWorldPointTowardCamera(float[] worldPoint, Camera camera, float meters) {
        if (worldPoint == null || meters == 0f) return worldPoint;
        float[] cam = new float[3];
        camera.getPose().getTranslation(cam, 0);

        float vx = cam[0] - worldPoint[0];
        float vy = cam[1] - worldPoint[1];
        float vz = cam[2] - worldPoint[2];
        float len = (float) Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (len < 1e-6f) return worldPoint;

        vx /= len;
        vy /= len;
        vz /= len;

        return new float[]{
                worldPoint[0] + vx * meters,
                worldPoint[1] + vy * meters,
                worldPoint[2] + vz * meters
        };
    }

    /**
     * Offsets hit pose outward from the real surface.
     * <p>
     * Important for tube drawing:
     * If the drawing anchor is exactly on the measured depth surface, tiny depth noise can make
     * the drawing flicker/cut. A 5-10mm offset keeps it visually attached but stable.
     */
    private Pose offsetAlongSurfaceNormal(HitResult hit, Camera camera, float meters) {
        Pose hitPose = hit.getHitPose();
        float[] p = new float[3];
        hitPose.getTranslation(p, 0);

        float[] normal = estimateHitNormal(hit, camera);

        float[] outT = new float[]{
                p[0] + normal[0] * meters,
                p[1] + normal[1] * meters,
                p[2] + normal[2] * meters
        };

        float[] q = new float[4];
        hitPose.getRotationQuaternion(q, 0);
        return new Pose(outT, q);
    }

    /**
     * Best-effort surface normal for DepthPoint/Plane/Point.
     * <p>
     * For DepthPoint, ARCore gives a hit pose from the depth map. We use its local axis and
     * flip it toward the camera so the offset moves out of the real surface, not inside it.
     */
    private float[] estimateHitNormal(HitResult hit, Camera camera) {
        Pose pose = hit.getHitPose();

        float[] normal;
        Trackable t = hit.getTrackable();

        if (t instanceof Plane) {
            normal = ((Plane) t).getCenterPose().getZAxis();
        } else {
            normal = pose.getZAxis();
        }

        normalize3(normal);

        float[] hitT = new float[3];
        float[] camT = new float[3];
        pose.getTranslation(hitT, 0);
        camera.getPose().getTranslation(camT, 0);

        float vx = camT[0] - hitT[0];
        float vy = camT[1] - hitT[1];
        float vz = camT[2] - hitT[2];
        float vLen = (float) Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (vLen > 1e-6f) {
            vx /= vLen;
            vy /= vLen;
            vz /= vLen;
        }

        // Make normal face the camera. Offset should move toward visible side.
        float dot = normal[0] * vx + normal[1] * vy + normal[2] * vz;
        if (dot < 0f) {
            normal[0] *= -1f;
            normal[1] *= -1f;
            normal[2] *= -1f;
        }

        return normal;
    }

    private void normalize3(float[] v) {
        float len = (float) Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len < 1e-6f) return;
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }

    private float distanceCameraToPoseM(Pose cameraPose, Pose hitPose) {
        float[] c = new float[3];
        float[] p = new float[3];
        cameraPose.getTranslation(c, 0);
        hitPose.getTranslation(p, 0);
        float dx = p[0] - c[0];
        float dy = p[1] - c[1];
        float dz = p[2] - c[2];
        return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    private float distanceCameraToPointM(Pose cameraPose, float[] point) {
        float[] c = new float[3];
        cameraPose.getTranslation(c, 0);
        float dx = point[0] - c[0];
        float dy = point[1] - c[1];
        float dz = point[2] - c[2];
        return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    private boolean shouldAddStrokePoint(StrokeSegment segment, float[] currentWorld) {
        if (segment == null || segment.anchor == null || currentWorld == null) return false;
        if (segment.anchor.getTrackingState() != TrackingState.TRACKING) return false;
        if (segment.localPoints.isEmpty()) return true;

        float[] lastWorld = getLastWorldPoint(segment);
        if (lastWorld == null) return true;

        float d = distanceBetweenPointsM(lastWorld, currentWorld);

        // Large jump usually means raw depth jumped from object surface to background/edge.
        if (d > DRAW_MAX_BAD_DEPTH_JUMP_M) return false;

        return d >= DRAW_MIN_ANCHOR_DISTANCE_M;
    }

    private boolean shouldStartNewSegment(StrokeSegment segment, float[] worldPoint) {
        if (segment == null || segment.anchor == null || worldPoint == null) return true;
        if (segment.localPoints.size() < 2) return false;

        float[] anchorT = new float[3];
        segment.anchor.getPose().getTranslation(anchorT, 0);
        return distanceBetweenPointsM(anchorT, worldPoint) > DRAW_MAX_SEGMENT_DISTANCE_M;
    }

    /**
     * Creates a persistent ARCore anchor from a surface hit.
     * This is intentionally similar to plane placement: the pose is calculated once from the hit result,
     * then ARCore tracking owns the anchor. We do not update this anchor again from depth pixels.
     */
    private Anchor createSurfaceAnchor(HitResult hit, Camera camera) {
        if (session == null || hit == null || camera == null) return null;
        try {
            Pose pose = offsetAlongSurfaceNormal(hit, camera, DRAW_SURFACE_OFFSET_M);
            return session.createAnchor(pose);
        } catch (Exception e) {
            // Fallback to ARCore's own hit anchor if a manual pose anchor fails for any reason.
            try {
                return hit.createAnchor();
            } catch (Exception ignored) {
                return null;
            }
        }
    }

    private StrokeSegment createStrokeSegmentFromHit(HitResult hit, Camera camera) {
        Anchor anchor = createSurfaceAnchor(hit, camera);
        if (anchor == null) return null;
        StrokeSegment segment = new StrokeSegment();
        segment.anchor = anchor;
        return segment;
    }

    private float[] poseToWorldPoint(Pose pose) {
        float[] p = new float[3];
        pose.getTranslation(p, 0);
        return p;
    }

    private StrokeSegment createStrokeSegmentAtWorldPoint(float[] worldPoint) {
        if (session == null || worldPoint == null) return null;
        StrokeSegment segment = new StrokeSegment();
        segment.anchor = session.createAnchor(Pose.makeTranslation(worldPoint[0], worldPoint[1], worldPoint[2]));
        return segment;
    }

    private float[] worldToSegmentLocal(StrokeSegment segment, float[] worldPoint) {
        if (segment == null || segment.anchor == null || worldPoint == null) return new float[]{0f, 0f, 0f};
        return segment.anchor.getPose().inverse().transformPoint(worldPoint);
    }

    private float[] segmentLocalToWorld(StrokeSegment segment, float[] localPoint) {
        if (segment == null || segment.anchor == null || localPoint == null) return new float[]{0f, 0f, 0f};
        return segment.anchor.getPose().transformPoint(localPoint);
    }

    private float[] getLastWorldPoint(StrokeSegment segment) {
        if (segment == null || segment.localPoints.isEmpty()) return null;
        return segmentLocalToWorld(segment, segment.localPoints.get(segment.localPoints.size() - 1));
    }

    private List<float[]> getSegmentWorldControlPoints(StrokeSegment segment) {
        ArrayList<float[]> worldPoints = new ArrayList<>();
        if (segment == null || segment.anchor == null) return worldPoints;
        if (segment.anchor.getTrackingState() != TrackingState.TRACKING) return worldPoints;

        for (float[] localPoint : segment.localPoints) {
            worldPoints.add(segmentLocalToWorld(segment, localPoint));
        }
        return worldPoints;
    }

    private float distanceBetweenPointsM(float[] a, float[] b) {
        float dx = b[0] - a[0];
        float dy = b[1] - a[1];
        float dz = b[2] - a[2];
        return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    private List<float[]> buildInterpolatedStrokePoints(StrokeSegment segment) {
        ArrayList<float[]> pts = new ArrayList<>();
        if (segment == null || segment.localPoints.isEmpty()) return pts;

        float[] prev = null;
        for (float[] p : segment.localPoints) {
            if (prev == null) {
                pts.add(p);
            } else {
                addInterpolatedPoints(pts, prev, p, DRAW_INTERP_SEGMENTS);
            }
            prev = p;
        }
        return pts;
    }

    private Pose offsetTowardCamera(Pose pose, Pose cameraPose, float meters) {
        float[] p = new float[3];
        pose.getTranslation(p, 0);

        float[] zAxis = cameraPose.getZAxis();
        float fx = -zAxis[0], fy = -zAxis[1], fz = -zAxis[2];
        float len = (float) Math.sqrt(fx * fx + fy * fy + fz * fz);
        if (len > 1e-6f) {
            fx /= len;
            fy /= len;
            fz /= len;
        }

        float[] outT = new float[]{p[0] + fx * meters, p[1] + fy * meters, p[2] + fz * meters};
        float[] q = new float[4];
        pose.getRotationQuaternion(q, 0);
        return new Pose(outT, q);
    }

    private void addInterpolatedPoints(List<float[]> stroke, float[] a, float[] b, int segments) {
        for (int i = 1; i <= segments; i++) {
            float t = i / (float) segments;
            stroke.add(new float[]{lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)});
        }
    }

    private float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    private void showOcclusionDialogIfNeeded() {
        boolean isDepthSupported = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC);
        if (!depthSettings.shouldShowDepthEnableDialog() || !isDepthSupported) return;

        new AlertDialog.Builder(this)
                .setTitle(R.string.options_title_with_depth)
                .setMessage(R.string.depth_use_explanation)
                .setPositiveButton(R.string.button_text_enable_depth,
                        (DialogInterface dialog, int which) -> depthSettings.setUseDepthForOcclusion(true))
                .setNegativeButton(R.string.button_text_disable_depth,
                        (DialogInterface dialog, int which) -> depthSettings.setUseDepthForOcclusion(false))
                .show();
    }

    private void launchInstantPlacementSettingsMenuDialog() {
        resetSettingsMenuDialogCheckboxes();
        Resources resources = getResources();
        new AlertDialog.Builder(this)
                .setTitle(R.string.options_title_instant_placement)
                .setMultiChoiceItems(
                        resources.getStringArray(R.array.instant_placement_options_array),
                        instantPlacementSettingsMenuDialogCheckboxes,
                        (DialogInterface dialog, int which, boolean isChecked) -> instantPlacementSettingsMenuDialogCheckboxes[which] = isChecked)
                .setPositiveButton(R.string.done, (DialogInterface dialogInterface, int which) -> applySettingsMenuDialogCheckboxes())
                .setNegativeButton(android.R.string.cancel, (DialogInterface dialog, int which) -> resetSettingsMenuDialogCheckboxes())
                .show();
    }

    private void launchDepthSettingsMenuDialog() {
        resetSettingsMenuDialogCheckboxes();
        Resources resources = getResources();
        if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
            new AlertDialog.Builder(this)
                    .setTitle(R.string.options_title_with_depth)
                    .setMultiChoiceItems(
                            resources.getStringArray(R.array.depth_options_array),
                            depthSettingsMenuDialogCheckboxes,
                            (DialogInterface dialog, int which, boolean isChecked) -> depthSettingsMenuDialogCheckboxes[which] = isChecked)
                    .setPositiveButton(R.string.done, (DialogInterface dialogInterface, int which) -> applySettingsMenuDialogCheckboxes())
                    .setNegativeButton(android.R.string.cancel, (DialogInterface dialog, int which) -> resetSettingsMenuDialogCheckboxes())
                    .show();
        } else {
            new AlertDialog.Builder(this)
                    .setTitle(R.string.options_title_without_depth)
                    .setPositiveButton(R.string.done, (DialogInterface dialogInterface, int which) -> applySettingsMenuDialogCheckboxes())
                    .show();
        }
    }

    private void applySettingsMenuDialogCheckboxes() {
        depthSettings.setUseDepthForOcclusion(depthSettingsMenuDialogCheckboxes[0]);
        depthSettings.setDepthColorVisualizationEnabled(depthSettingsMenuDialogCheckboxes[1]);
        instantPlacementSettings.setInstantPlacementEnabled(instantPlacementSettingsMenuDialogCheckboxes[0]);
        configureSession();
    }

    private void resetSettingsMenuDialogCheckboxes() {
        depthSettingsMenuDialogCheckboxes[0] = depthSettings.useDepthForOcclusion();
        depthSettingsMenuDialogCheckboxes[1] = depthSettings.depthColorVisualizationEnabled();
        instantPlacementSettingsMenuDialogCheckboxes[0] = instantPlacementSettings.isInstantPlacementEnabled();
    }

    private boolean hasTrackingPlane() {
        for (Plane plane : session.getAllTrackables(Plane.class)) {
            if (plane.getTrackingState() == TrackingState.TRACKING) return true;
        }
        return false;
    }

    private void updateLightEstimation(LightEstimate lightEstimate, float[] viewMatrix) {
        if (lightEstimate.getState() != LightEstimate.State.VALID) {
            virtualObjectShader.setBool("u_LightEstimateIsValid", false);
            return;
        }
        virtualObjectShader.setBool("u_LightEstimateIsValid", true);

        Matrix.invertM(viewInverseMatrix, 0, viewMatrix, 0);
        virtualObjectShader.setMat4("u_ViewInverse", viewInverseMatrix);

        updateMainLight(
                lightEstimate.getEnvironmentalHdrMainLightDirection(),
                lightEstimate.getEnvironmentalHdrMainLightIntensity(),
                viewMatrix);

        updateSphericalHarmonicsCoefficients(lightEstimate.getEnvironmentalHdrAmbientSphericalHarmonics());
        cubemapFilter.update(lightEstimate.acquireEnvironmentalHdrCubeMap());
    }

    private void updateMainLight(float[] direction, float[] intensity, float[] viewMatrix) {
        worldLightDirection[0] = direction[0];
        worldLightDirection[1] = direction[1];
        worldLightDirection[2] = direction[2];
        Matrix.multiplyMV(viewLightDirection, 0, viewMatrix, 0, worldLightDirection, 0);
        virtualObjectShader.setVec4("u_ViewLightDirection", viewLightDirection);
        virtualObjectShader.setVec3("u_LightIntensity", intensity);
    }

    private void updateSphericalHarmonicsCoefficients(float[] coefficients) {
        if (coefficients.length != 9 * 3) {
            throw new IllegalArgumentException("The given coefficients array must be of length 27");
        }
        for (int i = 0; i < 9 * 3; ++i) {
            sphericalHarmonicsCoefficients[i] = coefficients[i] * sphericalHarmonicFactors[i / 3];
        }
        virtualObjectShader.setVec3Array("u_SphericalHarmonicsCoefficients", sphericalHarmonicsCoefficients);
    }

    private void configureSession() {
        Config config = session.getConfig();
        config.setLightEstimationMode(Config.LightEstimationMode.ENVIRONMENTAL_HDR);

        // Full depth is kept enabled for occlusion. Raw depth images are also acquired in draw mode
        // for accurate stroke point placement using confidence filtering.
        if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
            config.setDepthMode(Config.DepthMode.AUTOMATIC);
        } else {
            config.setDepthMode(Config.DepthMode.DISABLED);
        }
        config.setFocusMode(Config.FocusMode.AUTO);

        config.setPlaneFindingMode(Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL);

        // Disable instant placement for accurate depth/object-surface drawing.
        // Instant placement gives an estimated pose first, then corrects later, which can make
        // drawing anchors shift or float on machine/chair/table sides.
        config.setInstantPlacementMode(InstantPlacementMode.DISABLED);

        session.configure(config);
    }
}

/**
 * Associates an Anchor with the trackable it was attached to.
 */
class WrappedAnchor {
    private final Anchor anchor;
    private final Trackable trackable;
    private final HelloArActivity.UserRole owner;

    public WrappedAnchor(Anchor anchor, Trackable trackable, HelloArActivity.UserRole owner) {
        this.anchor = anchor;
        this.trackable = trackable;
        this.owner = owner;
    }

    public Anchor getAnchor() {
        return anchor;
    }

    public Trackable getTrackable() {
        return trackable;
    }

    public HelloArActivity.UserRole getOwner() {
        return owner;
    }
}
