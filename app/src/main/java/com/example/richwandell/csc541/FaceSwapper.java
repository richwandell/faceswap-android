package com.example.richwandell.csc541;

import android.util.Log;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_java;
import org.nd4j.linalg.api.blas.Lapack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.BlasWrapper;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.bytedeco.javacpp.opencv_core.MatExpr;
import org.yaml.snakeyaml.nodes.Tag;

import static com.example.richwandell.csc541.MainActivity.TAG;
import static org.opencv.imgproc.Imgproc.CV_WARP_INVERSE_MAP;

/**
 * Created by richwandell on 4/1/18.
 */
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

    private static List<ArrayList<Integer>> OVERLAY_POINTS;

    private static List<Integer> ALIGN_POINTS;


    static {
        Loader.load(opencv_java.class);

        OVERLAY_POINTS = new ArrayList<>();
        ArrayList<Integer> top = new ArrayList<Integer>();

        top.addAll(LEFT_EYE_POINTS);
        top.addAll(RIGHT_EYE_POINTS);
        top.addAll(LEFT_BROW_POINTS);
        top.addAll(RIGHT_BROW_POINTS);
        OVERLAY_POINTS.add(top);

        ArrayList<Integer> bottom = new ArrayList<Integer>();
        bottom.addAll(NOSE_POINTS);
        bottom.addAll(MOUTH_POINTS);
        OVERLAY_POINTS.add(bottom);

        ALIGN_POINTS = new ArrayList<>();
        ALIGN_POINTS.addAll(LEFT_BROW_POINTS);
        ALIGN_POINTS.addAll(RIGHT_EYE_POINTS);
        ALIGN_POINTS.addAll(LEFT_EYE_POINTS);
        ALIGN_POINTS.addAll(RIGHT_BROW_POINTS);
        ALIGN_POINTS.addAll(NOSE_POINTS);
        ALIGN_POINTS.addAll(MOUTH_POINTS);
    }

    private final Mat swappedImage;

    public FaceSwapper(Mat im1, Mat im2, INDArray landmarks1, INDArray landmarks2) {
        int[] alignPoints = ALIGN_POINTS.stream()
            .mapToInt(Integer::intValue).toArray();

        INDArray M = transformationFromPoints(
            landmarks1.getRows(alignPoints),
            landmarks2.getRows(alignPoints)
        );


        Mat mask1 = getFaceMask(im1, landmarks1);
        Mat mask2 = getFaceMask(im2, landmarks2);

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

//        Imgcodecs.imwrite("outfile1.jpg", outputImage);

//        output_im = im1 * (1.0 - combined_mask) + warped_im2 * combined_mask
        this.swappedImage = outputImage;
    }

    public Mat getSwappedImage() {
        return this.swappedImage;
    }

    private Mat getCombinedMask(Mat mask1, Mat warpedMask2) {
        Mat dest = new Mat(mask1.size(), mask1.type());
        Core.max(mask1, warpedMask2, dest);
        return dest;
    }

    private Mat warpIm(Mat faceMask, INDArray m, Size size) {
        Mat dest = new Mat(size, CvType.CV_64FC3);
        Mat transformation = Mat.eye(2, 3, CvType.CV_64F);

        for(int i = 0; i < m.rows() -1; i++){
            INDArray row = m.getRow(i);

            transformation.put(
                i,
                0,
                new double[]{
                    row.getDouble(0),
                    row.getDouble(1),
                    row.getDouble(2)
                }
            );
        }

        Imgproc.warpAffine(
            faceMask,
            dest,
            transformation,
            size,
            CV_WARP_INVERSE_MAP,
            5,
            new Scalar(0, 0, 0)
        );

        return dest;
    }

    private Mat getFaceMask(Mat im, INDArray landmarks) {
        Mat newImage = new Mat(im.size(), CvType.CV_64FC3);

        for(ArrayList<Integer> group : OVERLAY_POINTS) {
            int[] rowsToGet = group.stream()
                .mapToInt(Integer::intValue).toArray();

            INDArray rows = landmarks.getRows(rowsToGet);

            drawConvexHull(newImage, rows);
        }

        Imgproc.GaussianBlur(newImage, newImage, new Size(FEATHER_AMOUNT, FEATHER_AMOUNT), 0);

        return newImage;
    }

    private void drawConvexHull(Mat im, INDArray points) {
        int[] shape = points.shape();
        ArrayList<Point> pointList = new ArrayList<>();
        for(int i = 0; i < shape[0]; i++){
            INDArray row = points.getRow(i);
            Point p = new Point();
            p.set(new double[]{row.getDouble(0), row.getDouble(1)});
            pointList.add(p);
        }

        MatOfPoint matOfPoint = new MatOfPoint();
        matOfPoint.fromList(pointList);

        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(matOfPoint, hull);

        MatOfPoint hullPoints = new MatOfPoint();
        pointList = new ArrayList<>();

        for(int i = 0; i < hull.size().height; i ++){
            int index = (int)hull.get(i, 0)[0];
            Point p = new Point();
            p.set(matOfPoint.get(index, 0));
            pointList.add(p);
        }
        hullPoints.fromList(pointList);

        Imgproc.fillConvexPoly(im, hullPoints, new Scalar(255, 255, 255));
    }

    private INDArray transformationFromPoints(INDArray points1, INDArray points2) {
        INDArray c1 = points1.mean(0);
        INDArray c2 = points2.mean(0);

        points1 = points1.subRowVector(c1);
        points2 = points2.subRowVector(c2);

        Number s1 = points1.stdNumber();
        Number s2 = points2.stdNumber();

        points1 = points1.div(s1);
        points2 = points2.div(s2);

        INDArray A = points1.transpose().mmul(points2);

        int nRows = A.rows();
        int nColumns = A.columns();

        INDArray S = Nd4j.zeros(1, nRows);
        INDArray U = Nd4j.zeros(nRows, nRows);
        INDArray V = Nd4j.zeros(nColumns, nColumns);


        Mat aMat = indArrayToMat(A);
        Mat sMat = indArrayToMat(S);
        Mat uMat = indArrayToMat(U);
        Mat vMat = indArrayToMat(V);

        Core.SVDecomp(aMat, sMat, uMat, vMat);

        INDArray uInd = matToIndArray(uMat);
        INDArray vInd = matToIndArray(vMat);

        INDArray R = uInd.mmul(vInd).transpose();
        INDArray hs1 = R.mul(s2.floatValue() / s1.floatValue());
        INDArray mul = hs1.mmul(c1.transpose());
        INDArray hs2 = c2.transpose().sub(mul);

        INDArray hs = Nd4j.hstack(hs1, hs2);

        INDArray done = Nd4j.vstack(
            hs,
            Nd4j.create(new float[]{0f, 0f, 1f})
        );

        return done;
    }

    private Mat indArrayToMat(INDArray a) {
        int[] shape = a.shape();
        Mat m = new Mat(shape[0], shape[1], CvType.CV_64FC1);

        for(int i = 0; i < shape[0]; i++) {
            for(int j = 0; j < shape[1]; j++) {
                m.put(i, j, a.getFloat(i, j));
            }
        }

        return m;
    }

    private INDArray matToIndArray(Mat m) {
        int width = m.width();
        int height = m.height();

        INDArray ind = Nd4j.create(height, width);

        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                double[] value = m.get(i, j);
                ind.put(i, j, value[0]);
            }
        }
        return ind;
    }
}
