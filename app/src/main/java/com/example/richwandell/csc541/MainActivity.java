package com.example.richwandell.csc541;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Point;
import android.media.Image;
import android.media.MediaMetadataRetriever;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import com.twelvemonkeys.image.ImageUtil;
import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_COLOR;

public class MainActivity extends AppCompatActivity {

    static String TAG = "rdebug";

    static final int REQUEST_CODE_PERMISSION = 1;

    private static String[] PERMISSIONS_REQ = {
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
        Manifest.permission.CAMERA
    };

    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");
//    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if(verifyPermissions(this)) {
            while(!copyLandmark()){}
            if (null == savedInstanceState) {
                Mat bFace = null;
                Bitmap bitmap = null;
                try {
                    bFace = ImageUtils.loadResource(this, R.drawable.brad_face, CV_LOAD_IMAGE_COLOR);

                    bitmap = ImageUtils.matToBitmap(bFace);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                Bitmap bradsFace = BitmapFactory.decodeResource(getResources(), R.drawable.brad_face);
                int orgWidth = bradsFace.getWidth();
                int orgHeight = bradsFace.getHeight();
                int[] pixels = new int[orgWidth * orgHeight];
                bradsFace.getPixels(pixels, 0, orgWidth, 0, 0, orgWidth, orgHeight);

                Mat m = new Mat(orgHeight, orgWidth, CvType.CV_8UC4);

                int id = 0;
                for(int row = 0; row < orgHeight; row++) {
                    for(int col = 0; col < orgWidth; col++) {
                        int color = pixels[id];
                        int a = (color >> 24) & 0xff; // or color >>> 24
                        int r = (color >> 16) & 0xff;
                        int g = (color >>  8) & 0xff;
                        int b = (color      ) & 0xff;

                        m.put(row, col, new int[]{a, r, g, b});
                        id++;
                    }
                }



//                Frame outputFrame = new Frame.Builder().setBitmap(myBitmap).build();

//                AndroidFrameConverter frameConverter = new AndroidFrameConverter();

                showCamera();
            }
        }
    }

    private void showCamera() {
        getSupportFragmentManager().beginTransaction()
            .replace(R.id.container, Camera2BasicFragment.newInstance())
            .commit();
    }

    private boolean copyLandmark () {
        final String targetPath = Constants.getFaceShapeModelPath();
        if (!new File(targetPath).exists()) {
            FileUtils.copyFileFromRawToOthers(getApplicationContext(), R.raw.shape_predictor_68_face_landmarks, targetPath);
        }
        return true;
    }



    public void testImageFace() {

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.brad_face);

        if(copyLandmark()) {


            FaceDet faceDet = new FaceDet(Constants.getFaceShapeModelPath());
            List<VisionDetRet> results = faceDet.detect(bitmap);
            for (final VisionDetRet ret : results) {
                String label = ret.getLabel();
                int rectLeft = ret.getLeft();
                int rectTop = ret.getTop();
                int rectRight = ret.getRight();
                int rectBottom = ret.getBottom();
                // Get 68 landmark points
                ArrayList<Point> landmarks = ret.getFaceLandmarks();
                for (Point point : landmarks) {
                    int pointX = point.x;
                    int pointY = point.y;

                    Log.d(TAG, Integer.toString(pointX) + Integer.toString(pointY));
                }
            }
        }

    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    private static boolean verifyPermissions(Activity activity) {
        // Check if we have write permission
        int write_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int read_persmission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int camera_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.CAMERA);

        if (write_permission != PackageManager.PERMISSION_GRANTED ||
            read_persmission != PackageManager.PERMISSION_GRANTED ||
            camera_permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                activity,
                PERMISSIONS_REQ,
                REQUEST_CODE_PERMISSION
            );
            return false;
        } else {
            return true;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_CODE_PERMISSION) {
            showCamera();
        }
    }
}
