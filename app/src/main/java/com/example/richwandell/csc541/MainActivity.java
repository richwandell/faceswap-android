package com.example.richwandell.csc541;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Point;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    static String TAG = "rdebug";

    static final int REQUEST_CODE_PERMISSION = 1;

    private static String[] PERMISSIONS_REQ = {
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
        Manifest.permission.CAMERA
    };

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
//        TextView tv = (TextView) findViewById(R.id.sample_text);
//        tv.setText(stringFromJNI());

        if(verifyPermissions(this)) {
            if (null == savedInstanceState) {
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
        if(copyLandmark()) {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.brad_face);

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
