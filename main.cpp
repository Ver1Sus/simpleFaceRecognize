
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/ml.h>
#include "iostream"
#include <stdio.h>

/*

 Поверх лица выводится маска
 -несколько масок + поворот головы


*/
using namespace cv;


Mat monokl2      = imread("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\monokl2.png");
Mat testFace     = imread("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\face.jpg");
Mat mous1        = imread("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\mous1.png");
Mat hat1         = imread("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\hat.png");

IplImage* monokl2_2 = new IplImage(monokl2);
IplImage* mous1_ = new IplImage(mous1);
IplImage* hat1_ = new IplImage(hat1);

int g_slider_pos = 0;
CvCapture* g_capture = NULL;

///---------- подгоняем размер входязего изображения под заданную ширину
IplImage* resizeImage(IplImage* src, int width){

//    float scale = (width*0.1)/src->width;
//    printf("%d %d", src->width, width);
    float scale = (width*1.0)/src->width;
    IplImage* dest = cvCreateImage(cvSize(src->width*scale, src->height*scale), src->depth, src->nChannels);
    cvResize(src, dest);


    return dest;
}


///---- отобразить элемент на изображении
void showElement(IplImage* Iimage, Rect RoiZone, IplImage* faceImg, Mat frameFromCam, CvScalar borderColor= CV_RGB(255,0,0) ){

    //---- меняем размер элемента под размер ширины объекта
    Iimage = resizeImage(Iimage, RoiZone.width);
    Mat Mimage = cvarrToMat(Iimage);
    RoiZone.height = Mimage.rows;
    RoiZone.width = Mimage.cols;

//    RoiZone = Rect(RoiZone.x, RoiZone.y, RoiZone.width, RoiZone.height);

//    cvRectangle(faceImg, {RoiZone.x, RoiZone.y}, {RoiZone.x+RoiZone.width, RoiZone.y+RoiZone.height},borderColor,1,4,0);


    ///////------------------------- не выходить за рамки -------------------------
    if ((RoiZone.x + Iimage->width) < faceImg->width && (RoiZone.y + Iimage->height) < faceImg->height
            && RoiZone.x  > 1 && RoiZone.y > 1)
    {
//       frameFromCam(RoiZone) *= 0.5;
//       frameFromCam(RoiZone) += Mimage;

        Mimage.copyTo(frameFromCam(cv::Rect(RoiZone.x,RoiZone.y, Mimage.cols, Mimage.rows)));
//       Mimage.copyTo(frameFromCam(RoiZone));
    }
    else{
//       printf("Border error");
    }
}




void faceDetect(Mat frameFromCam){

    IplImage* faceImg;
//    if (frameFromCam == NULL){
//        faceImg = cvLoadImage("face.jpg");
//    }
//    else faceImg = frameFromCam;


    faceImg = new IplImage(frameFromCam);

    CvHaarClassifierCascade *clCascade, *clCascadeEyeRight, *clCascadeMouth;
    CvMemStorage *mStorage = 0;
    CvSeq *faceRectSeq, *eyeRightSeq, *mouthSeq;

    mStorage = cvCreateMemStorage(0);

    clCascade           = (CvHaarClassifierCascade *) cvLoad ("haarcascade_frontalface_default.xml", 0, 0, 0);
    clCascadeEyeRight   = (CvHaarClassifierCascade *) cvLoad ("haarcascade_righteye_2splits.xml", 0, 0, 0);
    clCascadeMouth   = (CvHaarClassifierCascade *) cvLoad ("haarcascade_mcs_mouth.xml", 0, 0, 0);



    if ( !faceImg || !mStorage || !clCascade || !clCascadeEyeRight || !clCascadeMouth )
       {
           printf("Initilization error : %s" , (!faceImg)? "cant load image" : (!clCascade || !clCascadeEyeRight || !clCascadeMouth)?
               "cant load haar cascade" :
               "unable to locate memory storage");

           return;
       }




    //------------------------------------------------
    //----- find face
    //------------------------------------------------
    faceRectSeq = cvHaarDetectObjects(faceImg,clCascade,mStorage,
            1.2,
            3,
            CV_HAAR_DO_CANNY_PRUNING,
            cvSize(25,25));

    for ( int i = 0; i < (faceRectSeq? faceRectSeq->total:0); i++ )
    {

        CvRect *faceR = (CvRect*)cvGetSeqElem(faceRectSeq,i);
        CvPoint p1 = { faceR->x, faceR->y };
        CvPoint p2 = { faceR->x + faceR->width, faceR->y + faceR->height };
        cvRectangle(faceImg,p1,p2,CV_RGB(0,255,255),1,4,0);

//        Rect hatZone(faceR->x + faceR->width, faceR->y - hat1_->height, faceR->width, hat1_->height);
        Rect hatZone(faceR->x + faceR->width*0.2,
                     faceR->y - hat1_->height*0.3,
                     hat1_->width*0.3,
                     hat1_->height*0.3);
        showElement(hat1_, hatZone, faceImg, frameFromCam);

/*
//        printf(" %d %d %d %d\n", faceR->x, faceR->y, faceR->width, faceR->height);
        cvRectangle(faceImg,p1,p2,CV_RGB(0,255,255),1,4,0);

        //---- меняем размер шляпы -  чуть меньше чем размер ширины лица
        hat1_ = resizeImage(hat1_, faceR->width*0.6);
        hat1 = cvarrToMat(hat1_);

        //---- Область для расположения шляпы - чуть выше лица
        Rect hatZone(faceR->x + hat1_->width*0.3, faceR->y - hat1_->height, faceR->width*0.6, hat1_->height);
        p1 = { hatZone.x, hatZone.y };
        p2 = { hatZone.x+hatZone.width, hatZone.y+hatZone.height };
        cvRectangle(faceImg,
            { hatZone.x, hatZone.y},
            {hatZone.x+hatZone.width, hatZone.y+hatZone.height},CV_RGB(0,0,0),1,4,0);


        frameFromCam(hatZone) -= hat1*2;
//        hat1.copyTo(frameFromCam(hatZone));
//        cvRectangle(faceImg,p1,p2,CV_RGB(0,0,0),1,4,0);
*/
        //------------------------------------------------
        //------------------------ find eye
        //------------------------------------------------
        Mat img2 = frameFromCam;
        eyeRightSeq = cvHaarDetectObjects(faceImg, clCascadeEyeRight, mStorage,
            1.2,
            3,
            CV_HAAR_DO_CANNY_PRUNING,
            cvSize(25,25));

        for ( int i = 0; i < (eyeRightSeq? eyeRightSeq->total:0); i++ )
        {
            CvRect *r = (CvRect*)cvGetSeqElem(eyeRightSeq,i);
            CvPoint p1 = { r->x, r->y };
            CvPoint p2 = { r->x + r->width, r->y + r->height };

//            printf(" %d %d %d %d\n", r->x, r->y, r->width, r->height);

            //---- не показывать, если глаз находится в проавой половине лица.
            if (((r->x - faceR->x)*1.0)/faceR->width > 0.4){
//                printf("No");
                continue;
            }


            Rect eyeMonoklZone(r->x, r->y, r->width, monokl2_2->height);
            showElement(monokl2_2, eyeMonoklZone, faceImg, frameFromCam);

            /*
            //---- выделить глаз квадратом
//            cvRectangle(faceImg,p1,p2,CV_RGB(255,0,0),1,4,0);

            //---- меняем размер монокля под размер ширины глаза
            monokl2_2 = resizeImage(monokl2_2, r->width);
            monokl2 = cvarrToMat(monokl2_2);


            Rect eyeMonoklZone(r->x, r->y, monokl2_2->width, monokl2_2->height);
            p1 = { r->x, r->y };
            p2 = { r->x + monokl2_2->width, r->y + monokl2_2->height };

//            cvRectangle(faceImg,p1,p2,CV_RGB(255,0,0),1,4,0);

            ///////------------------------- не выходить за рамки -------------------------
            if ((r->x + monokl2_2->width) < faceImg->width && (r->y + monokl2_2->height) < faceImg->height)
            {
               printf("%d %d, %d\n\n",r->y, monokl2_2->height,faceImg->height );
//               frameFromCam(eyeMonoklZone) *= 0.5;
               frameFromCam(eyeMonoklZone) += monokl2;
//               monokl2.copyTo(frameFromCam(eyeMonoklZone));
            }
            else{
               printf("Border error");
            }*/
        }


        //------------------------------------------------
        //-- Find mouth
        //------------------------------------------------
        mouthSeq = cvHaarDetectObjects(faceImg, clCascadeMouth, mStorage,
            1.1,
            3,
            0,
            cvSize(10, 10));

//        cvRectangle(faceImg, {faceR->x + faceR->width*0.2, faceR->y + (faceR->height*0.65)}, {faceR->x + faceR->width*0.8, faceR->y + faceR->height*0.95}, CV_RGB(0,0,0),1,4,0);

        for ( int i = 0; i < (mouthSeq? mouthSeq->total:0); i++ )
        {

            CvRect *mouthR = (CvRect*)cvGetSeqElem(mouthSeq,i);
            CvPoint p1 = { mouthR->x, mouthR->y };
            CvPoint p2 = { mouthR->x + mouthR->width, mouthR->y + mouthR->height };

            //---- выбираем только те, которые в области рта
            if(mouthR->x >= faceR->x + faceR->width*0.2 && mouthR->x + mouthR->width <= faceR->x + faceR->width*0.8
                && mouthR->y >= faceR->y + (faceR->height*0.65) && mouthR->y + mouthR->height <= faceR->y + faceR->height*0.95)
            {
//                cvRectangle(faceImg,p1,p2,CV_RGB(0,255,0),1,4,0);

                Rect mousZone(mouthR->x, mouthR->y - mous1_->height/2, mouthR->width, mous1_->height);
                showElement(mous1_, mousZone, faceImg, frameFromCam, CV_RGB(255, 255, 255));


                /*                //---- меняем размер moustache под размер ширины mouth
                mous1_ = resizeImage(mous1_, mouthR->width);
                mous1 = cvarrToMat(mous1_);

                //---- Область для расположения усов - наполовину выходят за верхний уровень рта
                Rect mousZone(mouthR->x, mouthR->y - mous1_->height/2, mous1_->width, mous1_->height);
                p1 = { mousZone.x, mousZone.y };
                p2 = { mousZone.x+mousZone.width, mousZone.y+mousZone.height };

//                cvRectangle(faceImg,p1,p2,CV_RGB(0,0,0),1,4,0);

                frameFromCam(mousZone) -= mous1;
//                mous1.copyTo(frameFromCam(mousZone));
                */

            }

        }



    }


//    printf("show");
    imshow("faceMonokl", frameFromCam);
//    imshow("rect", monokl2);
//    cvShowImage("Display Face", faceImg);


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
    while(true)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera


//        cvtColor(frame, edges, CV_BGR2GRAY);
//        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
//        Canny(edges, edges, 0, 30, 3);
//        imshow("edges", edges);
//        cvShowImage("original", frame);

        faceDetect(frame);

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




int test(){
    const int kNewWidth = 200;
    const int kNewHeight = 400;
    cvShowImage("faceMonokl", monokl2_2);

    IplImage* new_img;/*
    = cvCreateImage(cvSize(monokl2_2->width/2, monokl2_2->height/2), monokl2_2->depth, monokl2_2->nChannels);
    cvResize(monokl2_2, new_img);*/


    new_img = resizeImage(monokl2_2, 150);
    cvShowImage("new",new_img);

    cvWaitKey(0);

}


main(int argc, char ** argv)
{
//    show_image();

//    camera2();

    camera1();

//    test();

//    faceDetect(testFace);

    cvWaitKey(0);

    cvDestroyAllWindows();
    return 0;
}

