
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/ml.h>
#include "iostream"
#include <stdio.h>




using namespace cv;


int g_slider_pos = 0;
CvCapture* g_capture = NULL;


void faceDetect(IplImage* frameFromCam = NULL){
    IplImage* faceImg;
    if (frameFromCam == NULL){
        faceImg = cvLoadImage("face.jpg");
    }
    else faceImg = frameFromCam;



    CvHaarClassifierCascade *clCascade = 0;
    CvMemStorage *mStorage = 0;
    CvSeq *faceRectSeq;

    mStorage = cvCreateMemStorage(0);

    clCascade = (CvHaarClassifierCascade *) cvLoad ("haarcascade_frontalface_default.xml", 0, 0, 0);


    if ( !faceImg || !mStorage || !clCascade )
       {
           printf("Initilization error : %s" , (!faceImg)? "cant load image" : (!clCascade)?
               "cant load haar cascade" :
               "unable to locate memory storage");

           return;
       }

    faceRectSeq = cvHaarDetectObjects(faceImg,clCascade,mStorage,
            1.2,
            3,
            CV_HAAR_DO_CANNY_PRUNING,
            cvSize(25,25));


    for ( int i = 0; i < (faceRectSeq? faceRectSeq->total:0); i++ )
        {

            CvRect *r = (CvRect*)cvGetSeqElem(faceRectSeq,i);
            CvPoint p1 = { r->x, r->y };
            CvPoint p2 = { r->x + r->width, r->y + r->height };

             printf(" %d %d %d %d\n", r->x, r->y, r->width, r->height);
            cvRectangle(faceImg,p1,p2,CV_RGB(0,255,0),1,4,0);
        }

    cvShowImage("Display Face", faceImg);


}





//------------------ IMAGE  ----------
void show_image()
{
//    cv::Mat img = cv::imread("C://2a.jpg");
//    cv::imshow("Test", img);

    IplImage* img2 = cvLoadImage("C://2a.jpg");
    cvShowImage("DisplayPicha", img2);
}




int camera1(){
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    IplImage* image;
    namedWindow("edges",1);
    while(true)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera

        image = new IplImage(frame);

        cvtColor(frame, edges, CV_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
//        imshow("edges", edges);
        cvShowImage("original", image);

        faceDetect(image);

        char c = cvWaitKey(33);
        if (c == 27) { // нажата ESC
            break;
        }
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

int camera2(){

    CvCapture* capture = cvCreateCameraCapture(CV_CAP_ANY);
    assert( capture );

    IplImage* frame;

    // узнаем ширину и высоту кадра
    double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    printf("[i] %.0f x %.0f\n", width, height );


    cvNamedWindow("capture", CV_WINDOW_AUTOSIZE);


    while(true){
            // получаем кадр
            frame = cvQueryFrame( capture );



            // показываем
            cvShowImage("capture", frame);

            char c = cvWaitKey(33);
            if (c == 27) { // нажата ESC
                break;
            }
        }
//    cv::waitKey(0);

    cvDestroyAllWindows();
    return 0;
}



main(int argc, char ** argv)
{
//    show_image();

//    camera2();

    camera1();

//    faceDetect();


    cvDestroyAllWindows();
    return 0;
}

