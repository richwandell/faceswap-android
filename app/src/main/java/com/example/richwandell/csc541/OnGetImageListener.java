package com.example.richwandell.csc541;


import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.ImageView;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;
import com.tzutalin.dlibtest.ImageUtils;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static com.example.richwandell.csc541.ImageUtils.loadResource;
import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_COLOR;

/**
 * Class that takes in preview frames and converts the image to Bitmaps to process with dlib lib.
 */
public class OnGetImageListener implements OnImageAvailableListener {

    static {
        Loader.load(opencv_java.class);
    }

    private static final String TAG = "rdebug";

    private final float SCALE_DOWN = 0.5f;

    private final float SCALE_UP = 2F;

    private int mPreviewWidth = 0;
    private int mPreviewHeight = 0;
    private byte[][] mYUVBytes;
    private int[] mRGBBytes = null;
    private Bitmap mRGBframeBitmap = null;
    private Bitmap mMutableBitmap = null;

    private boolean mIsComputing = false;
    private Activity activity;
    private AutoFitTextureView textureView;
    private int rotation;
    private int sensorOrientation;

    private FaceDet mFaceDet;


    private Paint mFaceLandmardkPaint;
    private AssetManager assetManager;
    private ImageView mImageView;
    private Bitmap bradsFace;
    private List<VisionDetRet> bradsFaceResults;
    private ArrayList<Point> bPoints;
    private Mat bradsFaceMat;
    private Bitmap bradsFaceBitmap;
    private long start;
    private int yRowStride;
    private int uvRowStride;
    private int uvPixelStride;

    private int lastLeft = 0;
    private int lastRight = 0;
    private int lastTop = 0;
    private int lastBottom = 0;

    private int drawWidth = 0;
    private int drawHeight = 0;

    private boolean debug = false;


    public void initialize(
        final Context context,
        final AssetManager assetManager,
        Activity activity,
        AutoFitTextureView textureView,
        int rotation,
        int sensorOrientation,
        Size mPreviewSize) {
        this.assetManager = assetManager;
        this.activity = activity;

        mImageView = activity.findViewById(R.id.the_preview_image);

        this.textureView = textureView;
        this.rotation = rotation;
        this.sensorOrientation = sensorOrientation;
        mImageView.getLayoutParams().width = mPreviewSize.getWidth();
        mImageView.getLayoutParams().height = mPreviewSize.getHeight();
        mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());

        mFaceLandmardkPaint = new Paint();
        mFaceLandmardkPaint.setColor(Color.GREEN);
        mFaceLandmardkPaint.setStrokeWidth(2);
        mFaceLandmardkPaint.setStyle(Paint.Style.STROKE);

        drawWidth = mPreviewSize.getWidth();
        drawHeight = mPreviewSize.getHeight();

        activity.findViewById(R.id.picture).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                debug = !debug;
            }
        });

        try {
            bradsFaceMat = loadResource(context, R.drawable.brad_face, CV_LOAD_IMAGE_COLOR);
            Imgproc.cvtColor(bradsFaceMat, bradsFaceMat, Imgproc.COLOR_BGR2RGB);

            bradsFaceBitmap = Bitmap.createBitmap(bradsFaceMat.width(), bradsFaceMat.height(), Config.ARGB_8888);
            Utils.matToBitmap(bradsFaceMat, bradsFaceBitmap);

            FaceDet faceDet = new FaceDet(Constants.getFaceShapeModelPath());
            this.bradsFaceResults = faceDet.detect(bradsFaceBitmap);
            if(this.bradsFaceResults.size() > 0) {
                this.bPoints = this.bradsFaceResults.get(0).getFaceLandmarks();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void deInitialize() {
        synchronized (OnGetImageListener.this) {
            if (mFaceDet != null) {
                mFaceDet.release();
            }
        }
    }

    private void drawToImageView(final Bitmap bitmap) {
        this.activity.runOnUiThread(() -> {
            mImageView.getLayoutParams().width = textureView.getWidth();
            mImageView.getLayoutParams().height = textureView.getHeight();
            mImageView.setImageBitmap(bitmap);
        });
    }

    @Override
    public void onImageAvailable(final ImageReader reader) {

        Image image = null;

        try {
            image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            // No mutex needed as this method is not reentrant.
            if (mIsComputing) {
                image.close();
                return;
            }

            this.start = System.currentTimeMillis();
            mIsComputing = true;

            final Plane[] planes = image.getPlanes();

            // Initialize the storage bitmaps once when the resolution is known.
            if (mPreviewWidth != image.getWidth() || mPreviewHeight != image.getHeight()) {
                mPreviewWidth = image.getWidth();
                mPreviewHeight = image.getHeight();

                Log.d(TAG, String.format("Initializing at size %dx%d", mPreviewWidth, mPreviewHeight));
                mRGBBytes = new int[mPreviewWidth * mPreviewHeight];
                mRGBframeBitmap = Bitmap.createBitmap(mPreviewWidth, mPreviewHeight, Config.ARGB_8888);

                mYUVBytes = new byte[planes.length][];
                for (int i = 0; i < planes.length; ++i) {
                    mYUVBytes[i] = new byte[planes[i].getBuffer().capacity()];
                }
            }

            for (int i = 0; i < planes.length; ++i) {
                planes[i].getBuffer().get(mYUVBytes[i]);
            }

            yRowStride = planes[0].getRowStride();
            uvRowStride = planes[1].getRowStride();
            uvPixelStride = planes[1].getPixelStride();

            image.close();

            ((Runnable) this::postImageUpdate).run();

//            mInferenceHandler.post(this::postImageUpdate);
        } catch (final Exception e) {
            if (image != null) {
                image.close();
            }
            Log.e(TAG, "Exception!", e);
            return;
        }


    }

    private void resetLast() {
        Log.d(TAG, "resetLast");
        lastLeft = 0;
        lastRight = 0;
        lastTop = 0;
        lastBottom = 0;
    }

    private void increaseLast() {
        Log.d(TAG, "increaseLast");
        lastLeft -= 20;
        lastRight += 20;
        lastTop -= 20;
        lastBottom += 20;
    }

    private Bitmap getLocalBitmap() {
        ImageUtils.convertYUV420ToARGB8888(
            mYUVBytes[0],
            mYUVBytes[1],
            mYUVBytes[2],
            mRGBBytes,
            mPreviewWidth,
            mPreviewHeight,
            yRowStride,
            uvRowStride,
            uvPixelStride,
            false);

        Bitmap correct;
        mRGBframeBitmap.setPixels(mRGBBytes, 0, mPreviewWidth, 0, 0, mPreviewWidth, mPreviewHeight);
        if(rotation == 0 && sensorOrientation == 270) {
            Matrix matrix = new Matrix();
            matrix.postRotate(-90, mPreviewWidth / 2, mPreviewHeight / 2);
            correct = Bitmap.createBitmap(mRGBframeBitmap,
                0,
                0,
                mRGBframeBitmap.getWidth(),
                mRGBframeBitmap.getHeight(),
                matrix,
                true);
        } else {
            Matrix matrix = new Matrix();
            matrix.postScale( SCALE_DOWN, SCALE_DOWN);
            correct = Bitmap.createBitmap(mRGBframeBitmap,
                0,
                0,
                mRGBframeBitmap.getWidth(),
                mRGBframeBitmap.getHeight(),
                matrix,
                true);
        }


        if (lastLeft > 0) {
            try {
                int width = lastRight - lastLeft;
                int height = lastBottom - lastTop;

                int size = width * height;
                int[] localBytes = new int[size];
                int[] rbgBytes = new int[correct.getWidth() * correct.getHeight()];

                correct.getPixels(rbgBytes, 0, width, lastLeft, lastTop, width, height);

                System.arraycopy(rbgBytes, 0, localBytes, 0, localBytes.length);

                Bitmap local = Bitmap.createBitmap(width, height, Config.ARGB_8888);
                local.setPixels(rbgBytes, 0, width, 0, 0, width, height);
                Log.d(TAG, "worked");
                return local;
            } catch (Exception e) {
                Log.d(TAG, "didn't work");
                Log.d(TAG, e.getMessage());
                resetLast();
                return correct;
            }
        }
        return correct;
    }

    private void postImageUpdate() {
        Bitmap localBitmap = getLocalBitmap();
        List<VisionDetRet> results = mFaceDet.detect(localBitmap);

        // Draw on bitmap

        if(results != null && results.size() > 0) {
            ArrayList<Point> landmarks = null;
            int left = 0;
            int right = 0;
            int top = 0;
            int bottom = 0;
            for (final VisionDetRet ret : results) {
                landmarks = ret.getFaceLandmarks();
                left = ret.getLeft();
                right = ret.getRight();
                top = ret.getTop();
                bottom = ret.getBottom();
            }


            if(landmarks != null) {
                Mat currentImageMat = new Mat();
                Utils.bitmapToMat(localBitmap, currentImageMat);

                Imgproc.cvtColor(currentImageMat, currentImageMat, Imgproc.COLOR_RGBA2BGR);

                Bitmap newImage = processResult(landmarks, currentImageMat);

                Matrix matrix = new Matrix();
                matrix.postScale(SCALE_UP,SCALE_UP);

                newImage = Bitmap.createBitmap(newImage,
                    0,
                    0,
                    newImage.getWidth(),
                    newImage.getHeight(),
                    matrix,
                    true);

                mMutableBitmap = Bitmap.createBitmap(drawWidth, drawHeight, Config.ARGB_8888);
                Canvas c = new Canvas(mMutableBitmap);
                c.drawBitmap(newImage, lastLeft * SCALE_UP, lastTop * SCALE_UP, new Paint());

                if(debug) {
                    drawDebug(c, landmarks);
                }

                Matrix matrix1 = new Matrix();
                matrix1.postScale(
                    -1,
                    1,
                    mMutableBitmap.getWidth() / 2,
                    mMutableBitmap.getHeight() / 2);

                mMutableBitmap = Bitmap.createBitmap(mMutableBitmap,
                    0,
                    0,
                    mMutableBitmap.getWidth(),
                    mMutableBitmap.getHeight(),
                    matrix1,
                    true);

                drawToImageView(mMutableBitmap);
                updateLast(left, right, top, bottom);
            }
        } else {
            increaseLast();
        }


        long time = System.currentTimeMillis() - start;

        Log.d(TAG, Long.toString(time));
        mIsComputing = false;
    }

    private void drawDebug(Canvas c, ArrayList<Point> landmarks) {
        for(Point p : landmarks) {
            c.drawCircle(lastLeft * SCALE_UP + p.x * SCALE_UP, lastTop * SCALE_UP + p.y * SCALE_UP, 3, mFaceLandmardkPaint);
        }

        c.drawLine(0, 0, drawWidth, drawHeight, mFaceLandmardkPaint);
        c.drawLine(drawWidth, 0, 0, drawHeight, mFaceLandmardkPaint);

        c.drawLine(lastLeft * SCALE_UP, lastTop * SCALE_UP, lastRight * SCALE_UP , lastTop * SCALE_UP, mFaceLandmardkPaint);
        c.drawLine(lastRight * SCALE_UP, lastTop * SCALE_UP, lastRight * SCALE_UP, lastBottom * SCALE_UP, mFaceLandmardkPaint);
        c.drawLine(lastRight * SCALE_UP, lastBottom * SCALE_UP, lastLeft * SCALE_UP, lastBottom * SCALE_UP, mFaceLandmardkPaint);
        c.drawLine(lastLeft * SCALE_UP, lastBottom * SCALE_UP, lastLeft * SCALE_UP, lastTop * SCALE_UP, mFaceLandmardkPaint);

        c.drawCircle(drawWidth / 2, drawHeight / 2, 10, mFaceLandmardkPaint);

    }

    private void updateLast(int left, int right, int top, int bottom) {
        if(lastLeft > 0) {
            int newLeft = lastLeft + left;
            lastRight = lastLeft + right;
            lastLeft = newLeft;

            int newTop = lastTop + top;
            lastBottom = lastTop + bottom;
            lastTop = newTop;

            if (lastLeft < 0) {
                lastLeft = 1;
            }
            if(lastRight > drawWidth) {
                lastRight = drawWidth;
            }
            if(lastTop < 0) {
                lastTop = 1;
            }
            if(lastBottom > drawHeight) {
                lastBottom = drawHeight;
            }

            if(lastRight - lastLeft > 500) {
                resetLast();
            }
        } else {
            lastLeft = left;
            lastRight = right;
            lastTop = top - 20;
            lastBottom = bottom + 20;
        }
    }

    private Bitmap processResult(ArrayList<Point> dPoints, Mat currentImageMat) {
        float[][] points1 = pointsToFloat(dPoints);
        float[][] points2 = pointsToFloat(bPoints);
        FaceSwapper f = new FaceSwapper(currentImageMat, bradsFaceMat, points1, points2);
        Mat faceMask = f.getFaceMask();
        Imgproc.cvtColor(faceMask, faceMask, Imgproc.COLOR_BGRA2RGBA);
        Bitmap faceMaskBitmap = Bitmap.createBitmap(faceMask.width(), faceMask.height(), Config.ARGB_8888);

        Utils.matToBitmap(faceMask, faceMaskBitmap, true);

        return faceMaskBitmap;
    }

    private float[][] pointsToFloat(ArrayList<Point> points) {
        float[][] f = new float[points.size()][2];
        for(int i = 0; i < points.size(); i++) {
            Point p = points.get(i);
            f[i][0] = p.x;
            f[i][1] = p.y;
        }
        return f;
    }

}
