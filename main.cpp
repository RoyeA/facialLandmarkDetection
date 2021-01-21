#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using namespace dlib;
#define MAIN_WINDOW "Main"
string DATA_PATH="";
void drawPolyline(const Mat& frame, const full_object_detection& landmarks, const int start, const int end, bool isClosed = false){
    std::vector<Point> points;
    for ( int i=start; i<=end; i++){
        points.push_back(Point(landmarks.part(i).x(),landmarks.part(i).y()));
    }
    polylines(frame, points, isClosed, Scalar(255,200,0),2,16);
}
void rednderFace(const cv::Mat& frame, const full_object_detection& landmarks){
    drawPolyline(frame, landmarks, 0, 16);
    drawPolyline(frame, landmarks, 17, 21);
    drawPolyline(frame, landmarks, 22, 26);
    drawPolyline(frame, landmarks, 27, 30);
    drawPolyline(frame, landmarks, 30, 35, true );
    drawPolyline(frame, landmarks, 36, 41, true );
    drawPolyline(frame, landmarks, 42, 47, true );
    drawPolyline(frame, landmarks, 48, 59, true );
    drawPolyline(frame, landmarks, 60, 67, true );

}
int main(int argc, char* argv[])
{
    namedWindow(MAIN_WINDOW);
	int camId = 0;
	if ( argc == 2 ) {
		camId = atoi(argv[1]);
	}
    VideoCapture webcam(camId);
    shape_predictor landmarkDetector;
    string PREDICTOR_PATH = DATA_PATH+"shape_predictor_68_face_landmarks.dat";
    deserialize(PREDICTOR_PATH) >> landmarkDetector;
    if ( !webcam.isOpened()){
        cout<<"Failed to open webcam ["<<camId<<"]"<<endl;
        exit(1);
    }
    frontal_face_detector faceDetector = get_frontal_face_detector();
    Mat frame;
    bool detectFace = true;
    bool detectLandmarks = true;
    bool renderFace = true;
    long start;
    int fps;
    while ( true ){
        start=getTickCount();
        webcam>>frame;
        cv::flip(frame,frame,1);
        cv_image<bgr_pixel> dlibIm(frame);

        std::vector<dlib::rectangle> faces ;
        if ( detectFace )
            faces = faceDetector(dlibIm);
        for(auto face: faces){
            cv::rectangle(frame,
                          Point(face.left(), face.top()),
                          Point(face.right(),face.bottom()),
                          Scalar(0,0,255));
        }
        if ( detectLandmarks ){
            for ( auto face: faces){
                full_object_detection landmarks = landmarkDetector(dlibIm, face);
                if ( renderFace )
                    rednderFace(frame, landmarks);
            }
        }
        double totalTicks = getTickCount()-start;
        fps = round(getTickFrequency()/totalTicks); // time laps in second per 1 frame;

        cv::rectangle(frame, Rect(10,3,300,80),Scalar(100,100,100),-1);
        cv::putText(frame, detectFace?"DetectFace: On":"DetectFace: Off", Point(10,20),FONT_HERSHEY_COMPLEX, .7, detectFace?Scalar(255,0,0):Scalar(0,0,255));
        cv::putText(frame, detectLandmarks?"DetectLandmarks: On":"DetectLandmarks: Off", Point(10,40),FONT_HERSHEY_COMPLEX, .7, detectLandmarks?Scalar(255,0,0):Scalar(0,0,255));
        cv::putText(frame, renderFace?"Render: On":"Render: Off", Point(10,60),FONT_HERSHEY_COMPLEX, .7, renderFace?Scalar(255,0,0):Scalar(0,0,255));
        cv::putText(frame, "FPS: "+to_string(fps), Point(10,80),FONT_HERSHEY_COMPLEX, .7, renderFace?Scalar(255,0,0):Scalar(0,0,0));

        imshow(MAIN_WINDOW,frame);
        int key = waitKey(25);
        if ( key == 'q' ) break;
        if ( key == 'r' ) renderFace=!renderFace;
        if ( key == 'f' ) detectFace=!detectFace;
        if ( key == 'l' ) detectLandmarks=!detectLandmarks;
    }
    webcam.release();
//    waitKey(0);
    return 0;
}
