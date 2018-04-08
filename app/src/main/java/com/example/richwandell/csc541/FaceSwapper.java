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
import static org.opencv.imgproc.Imgproc.CV_WARP_INVERSE_MAP;


public class FaceSwapper {

    private final int FEATHER_AMOUNT = 11;

    private static List<Integer> FACE_POINTS = IntStream.rangeClosed(17, 67)
        .boxed().collect(Collectors.toList());

    private static List<Integer> MOUTH_POINTS = IntStream.rangeClosed(48, 60)
        .boxed().collect(Collectors.toList());

    private static List<Integer> RIGHT_BROW_POINTS = IntStream.rangeClosed(17, 21)
        .boxed().collect(Collectors.toList());

    private static List<Integer> LEFT_BROW_POINTS = IntStream.rangeClosed(22, 26)
        .boxed().collect(Collectors.toList());

    private static List<Integer> RIGHT_EYE_POINTS = IntStream.rangeClosed(36, 41)
        .boxed().collect(Collectors.toList());

    private static List<Integer> LEFT_EYE_POINTS = IntStream.rangeClosed(42, 47)
        .boxed().collect(Collectors.toList());

    private static List<Integer> NOSE_POINTS = IntStream.rangeClosed(27, 34)
        .boxed().collect(Collectors.toList());

    private static List<Integer> JAW_POINTS = IntStream.rangeClosed(0, 16)
        .boxed().collect(Collectors.toList());

    private static int[][] OVERLAY_POINTS;

    private static int[] ALIGN_POINTS;


    static {
        Loader.load(opencv_java.class);

        ArrayList<Integer> top = new ArrayList<Integer>();

        top.addAll(LEFT_EYE_POINTS);
        top.addAll(RIGHT_EYE_POINTS);
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
        alignPoints.addAll(RIGHT_EYE_POINTS);
        alignPoints.addAll(LEFT_EYE_POINTS);
        alignPoints.addAll(RIGHT_BROW_POINTS);
        alignPoints.addAll(NOSE_POINTS);
        alignPoints.addAll(MOUTH_POINTS);

        ALIGN_POINTS = alignPoints.stream().mapToInt(Integer::intValue).toArray();
    }

    private Mat swappedImage = null;

    public Mat getSwappedImage() {
        return swappedImage;
    }

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

    public FaceSwapper(Mat im1, Mat im2, float[][] landmarks1, float[][] landmarks2) {

        Mat landmarks1Mat = floatToMat(landmarks1);
        Mat landmarks2Mat = floatToMat(landmarks2);

        Mat points1 = subSet(landmarks1Mat, ALIGN_POINTS);
        Mat points2 = subSet(landmarks2Mat, ALIGN_POINTS);



        long start = System.currentTimeMillis();

        Mat M = transformationFromPoints(points1, points2);

        Mat mask1 = getFaceMask(im1, landmarks1Mat);
        Mat mask2 = getFaceMask(im2, landmarks2Mat);

        Mat warpedMask2 = warpIm(mask2, M, im1.size());

        Mat combinedMask = getCombinedMask(mask1, warpedMask2);

        Mat warpedIm2 = warpIm(im2, M, im1.size());

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

        Core.multiply(warpedIm2, combinedMask, wim2Tcm, 1, CvType.CV_64FC3);
        Core.add(im1Tomm, wim2Tcm, output64);

        Mat outputImage = new Mat(output64.size(), CvType.CV_8UC3);
        output64.convertTo(outputImage, CvType.CV_8UC3);
        long end = System.currentTimeMillis() - start;

        System.out.println(Long.toString(end));
        Imgcodecs.imwrite("outfile3.jpg", outputImage);

        this.swappedImage = outputImage;

    }

    private Mat getCombinedMask(Mat mask1, Mat warpedMask2) {
        Mat dest = new Mat(mask1.size(), mask1.type());
        Core.max(mask1, warpedMask2, dest);
        return dest;
    }

    private Mat warpIm(Mat faceMask, Mat m, Size size) {
        Mat dest = new Mat(size, CvType.CV_64FC3);
        Imgproc.warpAffine(
            faceMask,
            dest,
            m,
            size,
            CV_WARP_INVERSE_MAP,
            5,
            new Scalar(0, 0, 0)
        );

        return dest;
    }

    private Mat getFaceMask(Mat im, Mat landmarks) {
        Mat newImage = new Mat(im.size(), CvType.CV_64FC3);

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

        Imgproc.GaussianBlur(newImage, newImage, new Size(FEATHER_AMOUNT, FEATHER_AMOUNT), 0);

        return newImage;
    }

    private void drawConvexHull(Mat im, MatOfPoint matOfPoint) {

        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(matOfPoint, hull);

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
