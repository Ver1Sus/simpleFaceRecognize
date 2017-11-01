

CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt



SOURCES += main.cpp

INCLUDEPATH += C:/OpenCV/build/modles/videoio/include \
    C:/OpenCV/build/include


LIBS += C:/OpenCV/build/bin/libopencv_calib3d320.dll
LIBS += C:/OpenCV/build/bin/libopencv_core320.dll
LIBS += C:/OpenCV/build/bin/libopencv_features2d320.dll
LIBS += C:/OpenCV/build/bin/libopencv_flann320.dll
LIBS += C:/OpenCV/build/bin/libopencv_highgui320.dll
LIBS += C:/OpenCV/build/bin/libopencv_imgcodecs320.dll
LIBS += C:/OpenCV/build/bin/libopencv_imgproc320.dll
LIBS += C:/OpenCV/build/bin/libopencv_ml320.dll
LIBS += C:/OpenCV/build/bin/libopencv_objdetect320.dll
LIBS += C:/OpenCV/build/bin/libopencv_photo320.dll
LIBS += C:/OpenCV/build/bin/libopencv_shape320.dll
LIBS += C:/OpenCV/build/bin/libopencv_stitching320.dll
LIBS += C:/OpenCV/build/bin/libopencv_superres320.dll
LIBS += C:/OpenCV/build/bin/libopencv_video320.dll
LIBS += C:/OpenCV/build/bin/libopencv_videoio320.dll
LIBS += C:/OpenCV/build/bin/libopencv_videostab320.dll
LIBS += C:/OpenCV/build/bin/opencv_ffmpeg320.dll


# more correct variant, how set includepath and libs for mingw
# add system variable: OPENCV_SDK_DIR=D:/opencv/build
# read http://doc.qt.io/qt-5/qmake-variable-reference.html#libs

#INCLUDEPATH += $$(OPENCV_SDK_DIR)/include

#LIBS += -L$$(OPENCV_SDK_DIR)/x86/mingw/lib \
#        -lopencv_core320        \
#        -lopencv_highgui320     \
#        -lopencv_imgcodecs320   \
#        -lopencv_imgproc320     \
#        -lopencv_features2d320  \
#        -lopencv_calib3d320

DISTFILES += \
    haarcascade_frontalface_default.xml

