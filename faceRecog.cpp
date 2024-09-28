#include <opencv2/opencv.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace cv;
using namespace dlib;

int main() {
    // Initialize face detector and shape predictor from Dlib
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp; // Download from Dlib website

    // Initialize Video Capture (from webcam)
    VideoCapture cap(0); // Use 0 for the default webcam
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video stream." << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;  // Capture frame-by-frame

        if (frame.empty()) {
            cerr << "Error: Empty frame." << endl;
            break;
        }

        // Convert OpenCV image to Dlib format
        cv_image<bgr_pixel> cimg(frame);

        // Detect faces
        std::vector<rectangle> faces = detector(cimg);

        // Loop over all detected faces and display landmarks
        for (auto& face : faces) {
            // Get the landmarks/parts for the face
            full_object_detection shape = sp(cimg, face);

            // Draw rectangle around the face
            rectangle(frame, Point(face.left(), face.top()), Point(face.right(), face.bottom()), Scalar(0, 255, 0), 2);

            // Draw face landmarks
            for (int i = 0; i < shape.num_parts(); ++i) {
                circle(frame, Point(shape.part(i).x(), shape.part(i).y()), 2, Scalar(0, 0, 255), FILLED);
            }
        }

        // Display the result
        imshow("Face Recognition", frame);

        // Exit the loop if 'q' is pressed
        if (waitKey(1) == 'q') break;
    }

    // Release video capture object
    cap.release();
    destroyAllWindows();
    return 0;
}
