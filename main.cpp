
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
- задека


///gloval, gray, resize(cvpyrdown), mashtab x,y of cascade
/// /// timing of functions

*/
using namespace cv;


Mat testFace     = imread("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\face.jpg");

//---- различные маски
//Mat Mask1_hat = imread("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask1_hat.png");
//IplImage* Mask1_hat_ = new IplImage(Mask1_hat);
IplImage* Mask1_hat_    = cvLoadImage("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask1_hat.png");
IplImage* Mask1_eye_    = cvLoadImage("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask1_eye.png");
IplImage* Mask1_mouth_  = cvLoadImage("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask1_mouth.png");

IplImage* Mask2_eye_    = cvLoadImage("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask2_eye.png");
IplImage* Mask2_mouth_  = cvLoadImage("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask2_mouth.png");

IplImage* Mask3_face_   = cvLoadImage("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask3_face.png");

IplImage* Mask4_mouth_  = cvLoadImage("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask4_mouth.png");
IplImage* Mask4_eye_    = cvLoadImage("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask4_eye.png");

//--- Каскады хаара для лица, глаз и рта
CvMemStorage *mStorage =  cvCreateMemStorage(0);
CvSeq *faceRectSeq, *faceRectSeq2, *eyeRightSeq, *mouthSeq;

CvHaarClassifierCascade *clCascade  = (CvHaarClassifierCascade *) cvLoad ("haarcascade_frontalface_default.xml", 0, 0, 0);
CvHaarClassifierCascade *clCascadeEyeRight   = (CvHaarClassifierCascade *) cvLoad ("haarcascade_righteye_2splits.xml", 0, 0, 0);
CvHaarClassifierCascade *clCascadeMouth   = (CvHaarClassifierCascade *) cvLoad ("haarcascade_mcs_mouth.xml", 0, 0, 0);




//---- номер выбанной маски
char maskType = 52;

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

       for (int i=0; i < Mimage.cols; i++){
           for (int j = 0; j < Mimage.rows; j++){
               if (Mimage.at<Vec3b>(Point(i,j)).val[0] + Mimage.at<Vec3b>(Point(i,j)).val[1] + Mimage.at<Vec3b>(Point(i,j)).val[2] >= 10 )
                    frameFromCam.at<Vec3b>(Point(RoiZone.x+i, RoiZone.y+j))= Mimage.at<Vec3b>(Point(i,j));
           }
       }

//        Mimage.copyTo(frameFromCam(cv::Rect(RoiZone.x,RoiZone.y, Mimage.cols, Mimage.rows)));
//       Mimage.copyTo(frameFromCam(RoiZone));
    }
    else{
//       printf("Border error");
    }
}




int faceDetect(Mat frameFromCam){


//
//    if (frameFromCam == NULL){
//        faceImg = cvLoadImage("face.jpg");
//    }
//    else faceImg = frameFromCam;

    //--- из MAT а IplImage
    IplImage* faceImg = new IplImage(frameFromCam);

    //--- если что-то отсутствует
    if ( !faceImg || !mStorage || !clCascade || !clCascadeEyeRight || !clCascadeMouth )
    {
       printf("Initilization error : %s" , (!faceImg)? "cant load image" : (!clCascade || !clCascadeEyeRight || !clCascadeMouth)?
           "cant load haar cascade" :
           "unable to locate memory storage");

       return 0;
    }


    //--- обесцвечиваем изображение
    IplImage* gray = cvCreateImage(cvSize(faceImg->width, faceImg->height), 8, 1);
    cvCvtColor(faceImg, gray, CV_BGR2GRAY);
    faceImg = gray;

    //--- сжимаем обесцвеченное в 2 раза
    IplImage* frameDown = cvCreateImage(cvSize(faceImg->width/2, faceImg->height/2), 8, 1);
    cvPyrDown(faceImg, frameDown);







    //------------------------------------------------
    //----- find face
    //------------------------------------------------
   /* faceRectSeq = cvHaarDetectObjects(faceImg,clCascade,mStorage,
            1.2,
            3,
            CV_HAAR_DO_CANNY_PRUNING,
            cvSize(25,25));
    */
    faceRectSeq2 = cvHaarDetectObjects(frameDown,clCascade,mStorage,
            1.2,
            3,
            CV_HAAR_DO_CANNY_PRUNING,
            cvSize(25,25));

    for ( int i = 0; i < (faceRectSeq2? faceRectSeq2->total:0); i++ )
    {

        CvRect *faceR = (CvRect*)cvGetSeqElem(faceRectSeq2,i);

        //--- Создаем изображение, где только лицо
        //--- Прилшось в ROI области глаз и рта для Х и У добавить Х и У позиции лица
        Rect faceZone(faceR->x,
                      faceR->y,
                      faceR->width,
                      faceR->height);
        IplImage* faceImg_Cut = cvCreateImage(cvSize(faceR->width, faceR->height), faceImg->depth, faceImg->nChannels);
        cvSetImageROI(frameDown, faceZone);
        cvCopy(frameDown, faceImg_Cut);
//        cvShowImage("t",faceImg_Cut);

//        CvPoint p1 = { faceR->x, faceR->y };
//        CvPoint p2 = { faceR->x + faceR->width, faceR->y + faceR->height };
//        cvRectangle(faceImg,p1,p2,CV_RGB(0,255,255),1,4,0);
        *faceR = CvRect(faceR->x*2, faceR->y*2, faceR->width*2, faceR->height*2); //---  умножаем на 2, т.к. получаем положение лица из сжатого изображения

        cv::Rect rect(faceR->x, faceR->y, faceR->width, faceR->height);
        cv::rectangle(frameFromCam, rect, cv::Scalar(255, 255, 0));

//        Rect hatZone(faceR->x + faceR->width, faceR->y - hat1_->height, faceR->width, hat1_->height);

//                     hat1_->width*0.3,
//                     hat1_->height*0.3);
//        printf("%d %d %d %d", faceR->y , hat1_->height , faceR->width, faceR->y - hat1_->height * faceR->width*0.001);


//        Mat faceImag_mat = cvarrToMat(faceImg);
//        IplImage* faceImg_Cut = new IplImage(frameFromCam(faceZone));


        if (maskType==49){
            Rect Mask1_hatZone(faceR->x + faceR->width*0.2, faceR->y - Mask1_hat_->height * faceR->width*0.001, //-- зависимость от ширины лица
                        faceR->width*0.6, faceR->height);
            showElement(Mask1_hat_, Mask1_hatZone, faceImg, frameFromCam);
        }

        if (maskType==51){
            Rect MaskZone(faceR->x, faceR->y, //-- зависимость от ширины лица
                         faceR->width*1.6, 0); //--- высота вычилсяется из ширины, пожтому не имеет значения какая тут задана высота
            showElement(Mask3_face_, MaskZone, faceImg, frameFromCam);
        }


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
        eyeRightSeq = cvHaarDetectObjects(faceImg_Cut, clCascadeEyeRight, mStorage,
            1.1,
            3,
            CV_HAAR_MAGIC_VAL,
            cvSize(0,0));

        for ( int i = 0; i < (eyeRightSeq? eyeRightSeq->total:0); i++ )
        {
            CvRect *r = (CvRect*)cvGetSeqElem(eyeRightSeq,i);
//            CvPoint p1 = { r->x, r->y };
//            CvPoint p2 = { r->x + r->width, r->y + r->height };

            *r = CvRect(r->x*2, r->y*2, r->width*2, r->height*2);

//            printf(" %d %d %d %d\n", r->x, r->y, r->width, r->height);

            //---- не показывать, если глаз находится в проавой половине лица.
            if ((r->x*1.0)/faceR->width > 0.45){
//                printf("No");
                continue;
            }

//            cvRectangle(faceImg_Cut,p1,{r->width, monokl2_2->height},CV_RGB(255,0,0),1,4,0);

            if(maskType==49){
                Rect eyeMonoklZone(r->x+faceR->x, r->y+faceR->y, r->width, 0);
                showElement(Mask1_eye_, eyeMonoklZone, faceImg, frameFromCam);
            }

            if(maskType==50){
                Rect eyeMonoklZone(r->x+faceR->x*0.98, r->y+faceR->y, Mask2_eye_->width * faceR->width*0.001, 0);
                showElement(Mask2_eye_, eyeMonoklZone, faceImg, frameFromCam);

            }

            if(maskType==52){
                Rect eyeMonoklZone(faceR->x + r->x*0.5, faceR->y + r->y*1.1,
                                   faceR->width*0.8, 0);
                showElement(Mask4_eye_, eyeMonoklZone, faceImg, frameFromCam);

            }
        }


        //------------------------------------------------
        //-- Find mouth
        //------------------------------------------------
        mouthSeq = cvHaarDetectObjects(faceImg_Cut, clCascadeMouth, mStorage,
            1.1,
            3,
            CV_HAAR_DO_ROUGH_SEARCH,
            cvSize(0, 0));

//        cvRectangle(faceImg, {faceR->x + faceR->width*0.2, faceR->y + (faceR->height*0.65)}, {faceR->x + faceR->width*0.8, faceR->y + faceR->height*0.95}, CV_RGB(0,0,0),1,4,0);

        for ( int i = 0; i < (mouthSeq? mouthSeq->total:0); i++ )
        {

            CvRect *mouthR = (CvRect*)cvGetSeqElem(mouthSeq,i);
            *mouthR = CvRect(mouthR->x*2, mouthR->y*2, mouthR->width*2, mouthR->height*2);
//            CvPoint p1 = { mouthR->x + faceR->x, mouthR->y + faceR->y };
//            CvPoint p2 = { mouthR->x + mouthR->width  + faceR->x, mouthR->y + mouthR->height  + faceR->y};

            //---- выбираем только те, которые в области рта
            if(mouthR->x >= faceR->width*0.2 && mouthR->x + mouthR->width <=  faceR->width*0.8
                && mouthR->y >= (faceR->height*0.65) && mouthR->y + mouthR->height <= faceR->height*0.95)
            {
//                cvRectangle(faceImg,p1,p2,CV_RGB(0,255,0),1,4,0);

                if (maskType==49){
                    Rect mousZone(mouthR->x + faceR->x, mouthR->y + faceR->y  - mouthR->height/4,
                                  mouthR->width, 0);
                    showElement(Mask1_mouth_, mousZone, faceImg, frameFromCam, CV_RGB(255, 255, 255));
                }

                if(maskType==50){
                    Rect mouthZone(mouthR->x + faceR->x, mouthR->y + faceR->y  - mouthR->height/4,
                                   mouthR->width, 0);
                    showElement(Mask2_mouth_, mouthZone, faceImg, frameFromCam);
                }

                if(maskType==52){
                    Rect mouthZone(faceR->x + mouthR->x * 0.3, mouthR->y + faceR->y  - mouthR->height*0.6,
                                   faceR->width*0.8, /*mouthR->width*2.3,*/ 0);
                    showElement(Mask4_mouth_, mouthZone, faceImg, frameFromCam);
                }
            }            
        }
    }


//    printf("show");
    imshow("faceMonokl", frameFromCam);
//    imshow("rect", monokl2);
//    cvShowImage("Display Face", frameDown);

    return 0;
}



int camera1(){
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    while(true)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera


//        cvtColor(frame, edges, CV_BGR2GRAY);
//        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
//        Canny(edges, edges, 0, 30, 3);
//        imshow("edges", edges);
//        cvShowImage("original", frame);

        //---- определяем маску по нажатой клавише
        //--- так же играт роль задержки
        int save = maskType;
        maskType = cvWaitKey(1000/60);
        if (maskType == -1) maskType = save;
    //    printf("%d",maskType);
        switch ( maskType )
        {
            case 49:
                break;
            case 50:
                break;
            case 51:
                break;
            case 52:
                break;
            case 27:
                return 0;
            default:
                maskType = 49;

        }
//        char c = cvWaitKey(1000/60);
//        if (c == 27) { // нажата ESC
//            break;
//        }

        faceDetect(frame);
//        imshow("test", frame);

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






//------------------ IMAGE  ----------
void show_image()
{
//    cv::Mat img = cv::imread("C://2a.jpg");
//    cv::imshow("Test", img);

    IplImage* img2 = cvLoadImage("C://2a.jpg");
    cvShowImage("DisplayPicha", img2);
}


int test(){
    const int kNewWidth = 200;
    const int kNewHeight = 400;

    Mat eye = imread("C:\\Users\\VerSus\\Documents\\QtProjects\\OPENCV\\opencv_Individual\\Mask1_eye.png");
    IplImage* eye_2 = new IplImage(eye);
    cvShowImage("faceMonokl", eye_2);

    IplImage* new_img;/*
    = cvCreateImage(cvSize(monokl2_2->width/2, monokl2_2->height/2), monokl2_2->depth, monokl2_2->nChannels);
    cvResize(monokl2_2, new_img);*/


    new_img = resizeImage(eye_2, 150);
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

