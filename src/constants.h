/*
 * constants.h
 *
 *  Created on: 02/08/2014
 *      Author: Guillermo Villamayor
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_
#define N 4
#define PI 3.14159265
#define SHIFT 5000
#define W 640
#define H 480
#define ANGLE(rad) (rad*180.0/PI)
#define GREEN CV_RGB(0,255,0)
#define RED CV_RGB(255,0,0)
#define CYAN CV_RGB(0,255,255)









const bool video = false;
const std::string home = "C:/Users/guillermo/Desktop";
const std::string vidDir = home + "/Videos/";
const std::string imgDir = home+ "/Pictures/";
const std::string vidCodec = "DIVX";
const std::string imgFmt = "jpg";
const bool rotacion = false;
const bool archivo = false;


// Constantes geometricas
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;
const int kNosePercentTop = 45;
const int kNosePercentSide = 30;
const int kNosePercentHeight = 35;
const int kNosePercentWidth = 25;

//Camaras
const int camInterior = 0;
const int focalPoint = 600;

const double qualityLevel = 0.3;
const int MAX_COUNT = 10;
const int MAX_INIT = 5;

//Features Shi-Tomasi
const bool ShiTomasi = true;
const double minDistance = 10;
const int blockSize = 3;
const bool useHarrisDetector = false;
const double k = 0.04;


//Features Harris
const bool Harris = false;
const int threshHarris = 180;

//Markers

const double fontScale = 0.5;
const int thickness = 0.5;
const int markerThickness = 2;

//Files
const cv::String lbpName = "../resources/lbpcascade_frontalface.xml";
const cv::String haarName1 = "../resources/haarcascade_frontalface_alt.xml";
const cv::String haarName2 = "../resources/haarcascade_frontalface_alt2.xml";
const cv::String haarName3 = "../resources/haarcascade_frontalface_default.xml";
const cv::String haarName4 = "../resources/haarcascade_frontalface_alt_tree.xml";





#endif /* CONSTANTS_H_ */
