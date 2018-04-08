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
import android.graphics.Rect;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Handler;
import android.os.Trace;
import android.util.Log;
import android.widget.ImageView;

import com.twelvemonkeys.image.BufferedImageFactory;
import com.twelvemonkeys.imageio.stream.BufferedImageInputStream;
import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;
import com.tzutalin.dlibtest.ImageUtils;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.datavec.image.loader.AndroidNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static com.example.richwandell.csc541.ImageUtils.bitmapToMat;
import static com.example.richwandell.csc541.ImageUtils.loadResource;
import static com.example.richwandell.csc541.ImageUtils.matToBitmap;
import static org.bytedeco.javacv.Java2DFrameUtils.*;
import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_COLOR;

/**
 * Class that takes in preview frames and converts the image to Bitmaps to process with dlib lib.
 */
public class OnGetImageListener implements OnImageAvailableListener {

    static {
        Loader.load(opencv_java.class);
    }

    private static final String TAG = "rdebug";

    private int mPreviewWidth = 0;
    private int mPreviewHeight = 0;
    private byte[][] mYUVBytes;
    private int[] mRGBBytes = null;
    private Bitmap mRGBframeBitmap = null;
    private Bitmap mMutableBitmap = null;

    private boolean mIsComputing = false;
    private Handler mInferenceHandler;
    private Activity activity;
    private AutoFitTextureView textureView;

    private Context mContext;
    private FaceDet mFaceDet;


    private Paint mFaceLandmardkPaint;
    private AssetManager assetManager;
    private ImageView mImageView;
    private Bitmap bradsFace;
    private List<VisionDetRet> bradsFaceResults;
    private ArrayList<Point> bPoints;
    private INDArray frameIm;
    private INDArray bradIm;
    private Mat currentImageMat;
    private Mat bradsFaceMat;
    private Bitmap bradsFaceBitmap;


    public void initialize(
        final Context context,
        final AssetManager assetManager,
        final Handler handler,
        ImageView mImageView,
        Activity activity,
        AutoFitTextureView textureView
    ) {

        this.mContext = context;
        this.assetManager = assetManager;
        this.mImageView = mImageView;

        this.mInferenceHandler = handler;
        this.activity = activity;
        this.textureView = textureView;
        mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());

        mFaceLandmardkPaint = new Paint();
        mFaceLandmardkPaint.setColor(Color.GREEN);
        mFaceLandmardkPaint.setStrokeWidth(2);
        mFaceLandmardkPaint.setStyle(Paint.Style.STROKE);

        try {
            bradsFaceMat = loadResource(context, R.drawable.brad_face, CV_LOAD_IMAGE_COLOR);
            bradsFaceBitmap = matToBitmap(bradsFaceMat);

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

    public void drawToImageView(final Bitmap bitmap) {
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
            mIsComputing = true;

            final Plane[] planes = image.getPlanes();

            // Initialize the storage bitmaps once when the resolution is known.
            if (mPreviewWidth != image.getWidth() || mPreviewHeight != image.getHeight()) {
                mPreviewWidth = image.getWidth();
                mPreviewHeight = image.getHeight();

                Log.d(TAG, String.format("Initializing at size %dx%d", mPreviewWidth, mPreviewHeight));
                mRGBBytes = new int[mPreviewWidth * mPreviewHeight];
                mRGBframeBitmap = Bitmap.createBitmap(mPreviewWidth, mPreviewHeight, Config.ARGB_8888);
                mMutableBitmap = Bitmap.createBitmap(mPreviewWidth, mPreviewHeight, Config.ARGB_8888);

                mYUVBytes = new byte[planes.length][];
                for (int i = 0; i < planes.length; ++i) {
                    mYUVBytes[i] = new byte[planes[i].getBuffer().capacity()];
                }
            }

            for (int i = 0; i < planes.length; ++i) {
                planes[i].getBuffer().get(mYUVBytes[i]);
            }

            final int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();
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

            mRGBframeBitmap.setPixels(mRGBBytes, 0, mPreviewWidth, 0, 0, mPreviewWidth, mPreviewHeight);
            currentImageMat = bitmapToMat(mRGBframeBitmap, CV_LOAD_IMAGE_COLOR);


            image.close();
        } catch (final Exception e) {
            if (image != null) {
                image.close();
            }
            Log.e(TAG, "Exception!", e);
            Trace.endSection();
            return;
        }

        mInferenceHandler.post(this::postImageUpdate);

    }

    private Mat arrayToMat(int[] array) {
        Mat mat = new Mat(array.length, array.length, CvType.CV_8UC3);
        return mat;
    }

    private void postImageUpdate() {

        List<VisionDetRet> results;
        synchronized (OnGetImageListener.this) {
            results = mFaceDet.detect(mRGBframeBitmap);
        }

        // Draw on bitmap
        if (results != null) {
            if(results.size() > 0) {
//                mMutableBitmap = Bitmap.createBitmap(mPreviewWidth, mPreviewHeight, Config.ARGB_8888);

//                Canvas canvas = new Canvas(mMutableBitmap);

                ArrayList<Point> landmarks = null;
                for (final VisionDetRet ret : results) {

//                                float resizeRatio = 1.0f;
//                                Rect bounds = new Rect();
//                                bounds.left = (int) (ret.getLeft() * resizeRatio);
//                                bounds.top = (int) (ret.getTop() * resizeRatio);
//                                bounds.right = (int) (ret.getRight() * resizeRatio);
//                                bounds.bottom = (int) (ret.getBottom() * resizeRatio);
//
//
//
//                                canvas.drawRect(bounds, mFaceLandmardkPaint);
//
//                                // Draw landmark
                    landmarks = ret.getFaceLandmarks();
//                                for (Point point : landmarks) {
//                                    int pointX = (int) (point.x * resizeRatio);
//                                    int pointY = (int) (point.y * resizeRatio);
//                                    canvas.drawCircle(pointX, pointY, 2, mFaceLandmardkPaint);
//                                }

                }
                if(landmarks != null) {
                    Bitmap newImage = processResult(landmarks, currentImageMat);

//                    Matrix matrix = new Matrix();
//                    matrix.postScale(
//                        -1,
//                        1,
//                        mPreviewWidth / 2,
//                        mPreviewHeight / 2);
//                    mMutableBitmap = Bitmap.createBitmap(mMutableBitmap,
//                        0,
//                        0,
//                        mMutableBitmap.getWidth(),
//                        mMutableBitmap.getHeight(),
//                        matrix,
//                        true);

                    drawToImageView(newImage);
                }
            }

        }

        mIsComputing = false;
    }

    private Bitmap processResult(ArrayList<Point> dPoints, Mat currentImageMat) {
        float[][] points1 = pointsToFloat(dPoints);
        float[][] points2 = pointsToFloat(bPoints);
        FaceSwapper f = new FaceSwapper(currentImageMat, bradsFaceMat, points1, points2);
        Mat swappedImage = f.getSwappedImage();
        return matToBitmap(swappedImage);
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

    private INDArray pointsToNd4j(ArrayList<Point> points) {
        float[][] fPoints = new float[points.size()][2];
        for(int i = 0; i < points.size(); i ++) {
            Point p = points.get(i);
            fPoints[i][0] = p.x;
            fPoints[i][1] = p.y;
        }
        return Nd4j.create(fPoints);
    }
}
