package com.example.richwandell.csc541;

import android.content.Context;
import android.graphics.Bitmap;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class ImageUtils {

    static {
        Loader.load(opencv_java.class);
    }

    public static Mat bitmapToMat(Bitmap bitmap, int flags) {

        ByteArrayOutputStream os = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 0 /*ignored for PNG*/, os);
        try {
            Mat encoded = new Mat(1, os.size(), CvType.CV_8U);
            encoded.put(0, 0, os.toByteArray());
            os.close();

            Mat decoded = Imgcodecs.imdecode(encoded, flags);
            encoded.release();

            return decoded;
        }catch(Exception e) {
            return null;
        }
    }

    public static Bitmap matToBitmap(Mat m) {
        int mWidth = m.width();
        int mHeight = m.height();
        int[] pixels = new int[mWidth * mHeight];
        int type = m.type();

        int index = 0;
        int a = 255;
        int b = 255;
        int g = 255;
        int r = 255;
        double[] abgr;

        Bitmap bitmap = Bitmap.createBitmap(mWidth, mHeight, Bitmap.Config.ARGB_8888);
        for (int ro = 0; ro < m.rows(); ro++) {
            for (int c = 0; c < m.cols(); c++) {
                abgr = m.get(ro, c);
                if(type == CvType.CV_8UC4) {
                    a = (int)abgr[0];
                    b = (int)abgr[1];
                    g = (int)abgr[2];
                    r = (int)abgr[3];
                } else if(type == CvType.CV_8UC3) {
                    a = 255;
                    b = (int)abgr[0];
                    g = (int)abgr[1];
                    r = (int)abgr[2];
                }
                int color = ((a << 24) & 0xFF000000) +
                    ((r << 16) & 0x00FF0000) +
                    ((g << 8) & 0x0000FF00) +
                    (b & 0x000000FF);
                pixels[index] = color;
                index++;
            }
        }
        bitmap.setPixels(pixels, 0, mWidth, 0, 0, mWidth, mHeight);
        return bitmap;
    }

    public static Mat loadResource(Context context, int resourceId, int flags) throws IOException {
        InputStream is = context.getResources().openRawResource(resourceId);
        ByteArrayOutputStream os = new ByteArrayOutputStream(is.available());

        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = is.read(buffer)) != -1) {
            os.write(buffer, 0, bytesRead);
        }
        is.close();

        Mat encoded = new Mat(1, os.size(), CvType.CV_8U);
        encoded.put(0, 0, os.toByteArray());
        os.close();

        Mat decoded = Imgcodecs.imdecode(encoded, flags);
        encoded.release();

        return decoded;
    }
}