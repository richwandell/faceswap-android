package com.example.richwandell.csc541;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.bytedeco.javacpp.opencv_core.CV_REDUCE_AVG;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.*;


public class FaceSwapper {

    private final int FEATHER_AMOUNT = 11;

    private final double COLOUR_CORRECT_BLUR_FRAC = 1;

    private static List<Integer> FACE_POINTS = IntStream.rangeClosed(17, 67)
        .boxed().collect(Collectors.toList());

    private static List<Integer> MOUTH_POINTS = IntStream.rangeClosed(48, 60)
        .boxed().collect(Collectors.toList());

    private static List<Integer> RIGHT_BROW_POINTS = IntStream.rangeClosed(17, 21)
        .boxed().collect(Collectors.toList());

    private static List<Integer> LEFT_BROW_POINTS = IntStream.rangeClosed(22, 26)
        .boxed().collect(Collectors.toList());

    private static int[] RIGHT_EYE_POINTS;

    private static int[] LEFT_EYE_POINTS;

    private static List<Integer> NOSE_POINTS = IntStream.rangeClosed(27, 34)
        .boxed().collect(Collectors.toList());

    private static List<Integer> JAW_POINTS = IntStream.rangeClosed(0, 16)
        .boxed().collect(Collectors.toList());

    private static int[][] OVERLAY_POINTS;

    private static int[] ALIGN_POINTS;

    private static int CV_TYPE = CvType.CV_64FC3;
    private final Mat landmarks2Mat;
    private final Mat landmarks1Mat;

    private Mat im1;

    private Mat im2;

    private float[][] landmarks1;

    private float[][] landmarks2;


    static {
        Loader.load(opencv_java.class);

        List<Integer> leftEyePoints = IntStream.rangeClosed(42, 47)
            .boxed().collect(Collectors.toList());

        List<Integer> rightEyePoints = IntStream.rangeClosed(36, 41)
            .boxed().collect(Collectors.toList());

        ArrayList<Integer> top = new ArrayList<Integer>();

        top.addAll(leftEyePoints);
        top.addAll(rightEyePoints);
        top.addAll(LEFT_BROW_POINTS);
        top.addAll(RIGHT_BROW_POINTS);
        int[] topInts = top.stream().mapToInt(Integer::intValue).toArray();


        ArrayList<Integer> bottom = new ArrayList<Integer>();
        bottom.addAll(NOSE_POINTS);
        bottom.addAll(MOUTH_POINTS);
        int[] bottomInts = bottom.stream().mapToInt(Integer::intValue).toArray();

        OVERLAY_POINTS = new int[][]{topInts, bottomInts};

        ArrayList<Integer> alignPoints = new ArrayList<>();
        alignPoints.addAll(LEFT_BROW_POINTS);
        alignPoints.addAll(rightEyePoints);
        alignPoints.addAll(leftEyePoints);
        alignPoints.addAll(RIGHT_BROW_POINTS);
        alignPoints.addAll(NOSE_POINTS);
        alignPoints.addAll(MOUTH_POINTS);

        ALIGN_POINTS = alignPoints.stream().mapToInt(Integer::intValue).toArray();
        LEFT_EYE_POINTS = leftEyePoints.stream().mapToInt(Integer::intValue).toArray();
        RIGHT_EYE_POINTS = rightEyePoints.stream().mapToInt(Integer::intValue).toArray();
    }

    private Mat swappedImage = null;

    private Mat floatToMat(float[][] f) {
        Mat m = new Mat(f.length, f[0].length, CvType.CV_64FC1);
        for(int i = 0; i < f.length; i++) {
            for(int j = 0; j < f[i].length; j++) {
                m.put(i, j, f[i][j]);
            }
        }
        return m;
    }

    private Mat subSet(Mat m, int[] i) {
        Mat m1= new Mat();
        for(int j : i) {
            m1.push_back(m.row(j));
        }
        return m1;
    }

    public Mat getSwappedImage() {
        Mat points1 = subSet(landmarks1Mat, ALIGN_POINTS);
        Mat points2 = subSet(landmarks2Mat, ALIGN_POINTS);

        Mat M = transformationFromPoints(points1, points2);

        Mat mask1 = getFaceMask(im1, landmarks1Mat);
        Mat mask2 = getFaceMask(im2, landmarks2Mat);

        Mat warpedMask2 = warpIm(mask2, M, im1.size());

        Mat combinedMask = getCombinedMask(mask1, warpedMask2);

        Mat warpedIm2 = warpIm(im2, M, im1.size());

        Mat warpedCorrectedIm2 = correctColors(im1, warpedIm2, landmarks1);

        //ones
        Mat ones = new Mat(combinedMask.size(), CvType.CV_64FC3);
        ones.setTo(new Scalar(1, 1, 1));
        //one minus mask
        Mat omm = new Mat(combinedMask.size(), CvType.CV_64FC3);
        //warped image 2 times combined mask
        Mat wim2Tcm = new Mat(combinedMask.size(), CvType.CV_64FC3);
        //image 1 times omm
        Mat im1Tomm = new Mat(combinedMask.size(), CvType.CV_64FC3);
        //output image
        Mat output64 = new Mat(combinedMask.size(), CvType.CV_64FC3);

        Core.subtract(ones, combinedMask, omm);

        Core.multiply(im1, omm, im1Tomm, 1, CvType.CV_64FC3);

        Core.multiply(warpedCorrectedIm2, combinedMask, wim2Tcm, 1, CvType.CV_64FC3);
        Core.add(im1Tomm, wim2Tcm, output64);

        Mat outputImage = new Mat(output64.size(), CV_8UC3);
        output64.convertTo(outputImage, CV_8UC3);

        return outputImage;
    }

    private Mat correctColors(Mat im1, Mat warpedIm2, float[][] landmarks1) {
        Mat lep = subSet(landmarks1Mat, LEFT_EYE_POINTS);
        MatOfDouble leftEyeMean = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        Core.meanStdDev(lep, leftEyeMean, std);

        Mat rep = subSet(landmarks1Mat, RIGHT_EYE_POINTS);
        MatOfDouble rightEyeMean = new MatOfDouble();
        Core.meanStdDev(rep, rightEyeMean, std);

        Mat mean = new Mat();
        Core.subtract(leftEyeMean, rightEyeMean, mean);

        double[] blurAmountDouble = mean.get(0, 0);

        int blurAmount = (int)(COLOUR_CORRECT_BLUR_FRAC * blurAmountDouble[0]);
        if(blurAmount % 2 == 0) {
            blurAmount += 1;
        }

        Mat im1Blur = new Mat();
        Imgproc.GaussianBlur(im1, im1Blur, new Size(blurAmount, blurAmount), 0);

        Mat im2Blur = new Mat();
        Imgproc.GaussianBlur(warpedIm2, im2Blur, new Size(blurAmount, blurAmount), 0);

        Mat wimtim1b = new Mat();
        Core.multiply(warpedIm2, im1Blur, wimtim1b, 1, CvType.CV_64FC3);

        Mat im1tim2blurdim2blur = new Mat();
        Core.divide(wimtim1b, im2Blur, im1tim2blurdim2blur, 1, CvType.CV_64FC3);

        return im1tim2blurdim2blur;
    }

    public Mat getFaceMask() {
        Mat points1 = subSet(landmarks1Mat, ALIGN_POINTS);
        Mat points2 = subSet(landmarks2Mat, ALIGN_POINTS);

        Mat M = transformationFromPoints(points1, points2);

        Mat mask1 = getFaceMask(im1, landmarks1Mat);
        Mat mask2 = getFaceMask(im2, landmarks2Mat);

        Mat warpedMask2 = warpIm(mask2, M, im1.size());

        Mat combinedMask = getCombinedMask(mask1, warpedMask2);

        Mat warpedIm2 = warpIm(im2, M, im1.size());

        Mat warpedCorrectedIm2 = correctColors(im1, warpedIm2, landmarks1);

        //ones
        Mat ones = new Mat(combinedMask.size(), CvType.CV_64FC3);
        ones.setTo(new Scalar(1, 1, 1));
        //one minus mask
        Mat omm = new Mat(combinedMask.size(), CvType.CV_64FC3);
        //warped image 2 times combined mask
        Mat wim2Tcm = new Mat(combinedMask.size(), CvType.CV_64FC3);
        //image 1 times omm
        Mat im1Tomm = new Mat(combinedMask.size(), CvType.CV_64FC3);
        //output image
        Mat output64 = new Mat(combinedMask.size(), CvType.CV_64FC3);

        Core.subtract(ones, combinedMask, omm);

        im1.convertTo(im1, CV_TYPE);

        Core.multiply(im1, omm, im1Tomm, 1, CvType.CV_64FC3);

        Mat wim2tcombinedm = new Mat(combinedMask.size(), CV_TYPE);

        Core.multiply(warpedCorrectedIm2, combinedMask, wim2tcombinedm, 1, CV_TYPE);

        Core.add(im1Tomm, wim2tcombinedm, output64);

        Mat outputImage = new Mat(output64.size(), CV_8UC3);
        output64.convertTo(outputImage, CV_8UC3);

        Imgproc.cvtColor(outputImage, outputImage, Imgproc.COLOR_BGR2BGRA);

        combinedMask.convertTo(combinedMask, CV_8UC3);

        Imgproc.cvtColor(combinedMask, combinedMask, Imgproc.COLOR_BGR2GRAY);

        Imgproc.threshold(combinedMask, combinedMask, 0.9, 255, THRESH_BINARY);

        Mat done = new Mat();
        outputImage.copyTo(done, combinedMask);
//        Core.bitwise_xor(outputImage, outputImage, combinedMask);

        return done;
    }

    public FaceSwapper(Mat im1, Mat im2, float[][] landmarks1, float[][] landmarks2) {
        this.im1 = im1;
        this.im2 = im2;
        this.landmarks1 = landmarks1;
        this.landmarks2 = landmarks2;
        this.landmarks1Mat = floatToMat(landmarks1);
        this.landmarks2Mat = floatToMat(landmarks2);
    }

    private Mat getCombinedMask(Mat mask1, Mat warpedMask2) {

        Mat dest = new Mat();
        Core.add(mask1, warpedMask2, dest);
        return dest;
    }

    private Mat warpIm(Mat im, Mat m, Size size) {
        Mat dest = new Mat(size, im.type());
        Imgproc.warpAffine(
            im,
            dest,
            m,
            size,
            CV_WARP_INVERSE_MAP,
            5,
            new Scalar(1, 1, 1)
        );

        return dest;
    }

    private Mat getFaceMask(Mat im, Mat landmarks) {
        Mat newImage = new Mat(im.size(), CV_TYPE);

        for(int[] rowsToGet : OVERLAY_POINTS) {
            MatOfPoint points = new MatOfPoint();
            ArrayList<Point> pointList = new ArrayList<>();
            for(int i : rowsToGet) {
                double px = landmarks.get(i, 0)[0];
                double py = landmarks.get(i, 1)[0];
                Point p = new Point(px, py);
                pointList.add(p);
            }
            points.fromList(pointList);
            drawConvexHull(newImage, points);
        }

        newImage.convertTo(newImage, CV_8UC3);
        Imgproc.GaussianBlur(newImage, newImage, new Size(FEATHER_AMOUNT, FEATHER_AMOUNT), 0);
        Imgproc.threshold(newImage, newImage, 0, 1, THRESH_BINARY);
        Imgproc.GaussianBlur(newImage, newImage, new Size(FEATHER_AMOUNT, FEATHER_AMOUNT), 0);
        newImage.convertTo(newImage, CV_TYPE);

        return newImage;
    }

    private void drawConvexHull(Mat im, MatOfPoint matOfPoint) {

        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(matOfPoint, hull, true);

        MatOfPoint hullPoints = new MatOfPoint();

        ArrayList<Point> pointList = new ArrayList<>();
        pointList = new ArrayList<>();

        for(int i = 0; i < hull.size().height; i ++){
            int index = (int)hull.get(i, 0)[0];
            Point p = new Point();
            p.set(matOfPoint.get(index, 0));
            pointList.add(p);
        }
        hullPoints.fromList(pointList);

        Imgproc.fillConvexPoly(im, hullPoints, new Scalar(1, 1, 1));
    }

    private Mat transformationFromPoints(Mat points1, Mat points2) {
        Mat c1 = new Mat();
        Core.reduce(points1, c1, 0, CV_REDUCE_AVG);
        Mat c2 = new Mat();
        Core.reduce(points2, c2, 0, CV_REDUCE_AVG);

        for(int i = 0; i < points1.height(); i++) {
            Mat row1 = points1.row(i);
            Core.subtract(row1, c1, row1);

            Mat row2 = points2.row(i);
            Core.subtract(row2, c2, row2);
        }

        MatOfDouble mean = new MatOfDouble();
        MatOfDouble s1 = new MatOfDouble();
        Core.meanStdDev(points1, mean, s1);

        MatOfDouble s2 = new MatOfDouble();
        Core.meanStdDev(points2, mean, s2);

        Core.divide(points1, s1, points1);
        Core.divide(points2, s2, points2);

        Mat A = new Mat();
        Core.transpose(points1, points1);
        Core.gemm(points1, points2,1, new Mat(), 0, A);

        Mat S = new Mat(1, A.height(), A.type());
        Mat U = new Mat(A.height(), A.height(), A.type());
        Mat V = new Mat(A.width(), A.width(), A.type());

        Core.SVDecomp(A, S, U, V);

        Mat R = new Mat();
        Core.gemm(U, V, 1, new Mat(), 0, R);
        Core.transpose(R, R);

        double s1d = s1.get(0, 0)[0];
        double s2d = s2.get(0, 0)[0];
        double std = s2d / s1d;

        Mat hs1 = new Mat();
        Core.multiply(R, new Scalar(std), hs1);
        Core.transpose(c1, c1);

        Mat mul = new Mat();
        Core.gemm(hs1, c1, 1, new Mat(), 0, mul);
        Core.transpose(c2, c2);

        Mat hs2 = new Mat();
        Core.subtract(c2, mul, hs2);

        List<Mat> src = Arrays.asList(hs1, hs2);
        Mat dst = new Mat();
        Core.hconcat(src, dst);

        return dst;
    }
}
