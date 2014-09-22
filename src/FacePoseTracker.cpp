#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>


#include <iostream>
#include <ctype.h>
#include <fstream>



#include "constants.h"

using namespace cv;
using namespace std;


double _intrinsics[9] =
	{ 360, 0., W/2,
	  0., 360, H/2,
	  0., 0., 1. };

CvMat intrinsics = cvMat(4,4,CV_64F, _intrinsics);


void rotateImage(const Mat, Mat, double, double, double, double, double, double, double);


void posit(int n, CvPoint3D32f *model, CvPoint2D32f *projection, double *tvec, double *rvec,CvPOSITObject *positObject) {

	float _prmat[9];
	CvMat prmat = cvMat(3,3,CV_32F,_prmat);

	float _prvec[3];
	CvMat prvec = cvMat(3,1,CV_32F,_prvec);
	float _ptvec[3];
	//CvPOSITObject *positObject = cvCreatePOSITObject( &model[0], 4 );
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 400, 1.0e-5f);
	cvPOSIT( positObject, &projection[0], focalPoint, criteria, _prmat, _ptvec );
	cvRodrigues2(&prmat, &prvec);
	for(int i = 0; i < 3; i++)
	{
		tvec[i] = _ptvec[i];
		rvec[i] = _prvec[i];
	}

}


void help()
{
	cout << "\nFlujo optico Lukas-Kanade,\n"
	                "Usando OpenCV %s\n" << CV_VERSION << "\n"
	                << endl;

	cout << "\nComandos: \n"
			"\tESC - salir del programa\n"
	        "\tr - auto inicializar seguimiento\n"
	        "\tc - borrar todos los puntos\n"
	        "\tn - habilitar/desabilitar el modo  \"nocturno\"\n"
	        "Para agregar o quitar un punto hacer click\n" << endl;
}
Point2f pt;
bool addRemovePt = false;

void onMouse( int event, int x, int y, int flags, void* param )
{
    if( event == CV_EVENT_LBUTTONDOWN )
	{
    	pt = Point2f((float)x,(float)y);
	    addRemovePt = true;
	}
}

vector<cv::Rect> detectFace(Mat frame_gray)
{
	vector<cv::Rect> faces;
	String face_cascade_name = haarName2;
	CascadeClassifier face_cascade;
	if( !face_cascade.load( face_cascade_name ) ){ cout<<"--(!)Error cargando la base de datos."<<endl; }
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
	return faces;
}

vector<Point2f> cambiarescala(vector<Point2f> corners, int x, int y)
{
	for (unsigned i = 0; i<corners.size();i++)
	{
		corners[i].x=corners[i].x+x;
		corners[i].y=corners[i].y+y;
	}
	return corners;
}

int main( int argc, char** argv )
{
	namedWindow( "Head tracking", 1 );
	setMouseCallback( "Head tracking", onMouse, 0 );
	moveWindow("Head tracking",0,0);
	namedWindow( "Mascara", 1 );
	moveWindow("Mascara",600,0);
	namedWindow( "Imagen gris", 1 );
	moveWindow("Imagen gris",0,600);
		//namedWindow( "Matriz Back", 1 );
	char s[1024], *t;
	double _prvec[3] = { 0, 0, 0 };
	double _ptvec[3] = { 0, 0, 0 };
	CvPoint2D32f projectedPoints[N];
	CvPoint3D32f modelPoints[2*N];
	vector<Point2f> points[2];
	CvFont defFont;
	vector<cv::Rect> faces;
	VideoCapture cap;
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	Size winSize(10,10);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, W);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, H);
	bool needToInit = false;
	bool nightMode = false;
	float xCM, yCM;
	CvPOSITObject *positObject;

	/*CvPoint3D32f modelPoints[N] = {
			{ 0.0f, 0.0f, 0.0f },
			{0.0f, 0.0f,0.0f},
			{0.0f, 0.0f, 0.0f},
			{ 0.0f, 0.0f, 0.0f}
		};*/





	if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
	        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
	else if( argc == 2 )
		cap.open(argv[1]);
	if( !cap.isOpened() )
	{
		cout << "No se encontro la camara...\n";
	    return 0;
	}

	help();
	Mat gray, prevGray, image;








	for(;;)
	{
	    Mat frame;
	    cap >> frame;
	    if( frame.empty() )
	    cout<<"No se encontro frame"<<endl;
	    frame.copyTo(image);
	    std::vector<cv::Mat> rgbChannels(3);
	    split(image, rgbChannels);
	    Mat gray = rgbChannels[2];
	    Mat maskNose (gray.size(), CV_8UC1);
	    Mat maskLeft (gray.size(), CV_8UC1);

	    for( unsigned i = 0; i < faces.size(); i++ )
	    {
	    	rectangle(frame, faces[i], 1234);
	    }
	    if( nightMode )
	    	image = Scalar::all(0);

	    if( needToInit )
	    {
	        // Inicializacion automática
	        faces = detectFace(gray);



	        if(faces.size()!=0)
	        {
	        	//cvWaitKey(99);
	        	Scalar color(255,255,255);
	        	//-- Encontrar y dibujar la region de los ojos
	        	int eye_region_width = faces[0].width * (kEyePercentWidth/100.0);
	        	int eye_region_height = faces[0].width * (kEyePercentHeight/100.0);
	        	int eye_region_top = faces[0].height * (kEyePercentTop/100.0)+faces[0].y;
	        	cv::Rect leftEyeRegion(faces[0].width*(kEyePercentSide/100.0)+faces[0].x,eye_region_top,eye_region_width,eye_region_height);
	        	cv::Rect rightEyeRegion(faces[0].width - eye_region_width - faces[0].width*(kEyePercentSide/100.0), eye_region_top,eye_region_width,eye_region_height);
	        	maskLeft.setTo(Scalar::all(0));
	        	rectangle(maskLeft,leftEyeRegion,color,CV_FILLED);

	        	goodFeaturesToTrack(gray, points[1], MAX_COUNT, qualityLevel, 10, maskLeft, 3, useHarrisDetector, 0.04);
	        	cornerSubPix(gray, points[1], winSize, Size(-1,-1), termcrit);
/*	        	for(unsigned i=0;i<points[1].size();i++)
	        	{
	        		if (points[1][i].x<leftEyeRegion.x||points[1][i].x>leftEyeRegion.x+leftEyeRegion.width)

	        	}*/
	        	for(unsigned i=0;i<3&&i<points[0].size();i++)
	        	{
	        		modelPoints[i].x=points[0][i].x;
	        		modelPoints[i].y=points[0][i].y;
	        		modelPoints[i].z=0.0f;
	        	}
	        	cout<<sizeof modelPoints<<endl;
	        	//-- Encontrar y dibujar la region de la nariz
	        	int nose_region_width = faces[0].width * (kNosePercentWidth/100.0);
	        	int nose_region_height = faces[0].width * (kNosePercentHeight/100.0);
	        	int nose_region_top = faces[0].height * (kNosePercentTop/100.0)+faces[0].y;
	        	cv::Rect noseRegion(faces[0].width*(kNosePercentSide/100.0)+faces[0].x,nose_region_top,nose_region_width,nose_region_height);
	        	maskNose.setTo(Scalar::all(0));
	        	rectangle(maskNose,leftEyeRegion,color,CV_FILLED);

	            goodFeaturesToTrack(gray, points[1], MAX_COUNT, qualityLevel, 10, maskNose, 3, useHarrisDetector, 0.04);
	            //goodFeaturesToTrack(gray, points[1], MAX_COUNT, qualityLevel, 10, Mat(), 3, useHarrisDetector, 0.04);
	            cornerSubPix(gray, points[1], winSize, Size(-1,-1), termcrit);

	            addRemovePt = false;
	            cout<<"llegue aqui 2 "<<points[0].size()<<endl;
	            for(unsigned i=0;i<3&&i<points[0].size();i++)
	            {
	            	modelPoints[i].x=points[0][i].x;
	            	modelPoints[i].y=points[0][i].y;
	            	modelPoints[i].z=3.0f;
	            }
	            cout<<sizeof modelPoints<<endl;
	            cout<<"llegue aqui 3"<<endl;
	            positObject = cvCreatePOSITObject( modelPoints, 6 );
	            imshow("Imagen gris",gray(cvRect(faces[0].x,faces[0].y, faces[0].width, faces[0].height)));
	            cvInitFont(&defFont, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.8f, 0.8f, 0, 1, 1);
	            IplImage *img = cvCreateImage(cvSize(W,H),8,3);
	            cvZero(img);
	        }
	        else putText(image, "No se detecto la cara", cvPoint(30,30), 1, 0.8, cvScalar(200,200,250), 1, 0);
	    }
	    else if( !points[0].empty() )
	    {
	    	vector<uchar> status;
	        vector<float> err;
	        if(prevGray.empty())
	        	gray.copyTo(prevGray);
	        calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termcrit, 0);
//	        cornerSubPix(gray, points[1], winSize, Size(-1,-1), termcrit);


	        for(unsigned i=0;i<N&&i<points[0].size();i++)
	        {
	        	projectedPoints[i].x=points[1][i].x;
	        	projectedPoints[i].y=points[1][i].y;
	        }



	        posit(N, modelPoints, projectedPoints, _ptvec, _prvec, positObject);
	        for(int i = 0; i < 3; i++)
	        {
	        	_prvec[i] *= -1;
	        }
	        size_t i, k;

	        for( i = k = 0; i < points[1].size(); i++ )
	        {
	        	if( addRemovePt )
	            {
	        		if( norm(pt - points[1][i]) <= 5 )
	                {
	        			addRemovePt = false;
	                    continue;
	                }
	            }

	            if( !status[i] )
	                continue;

	            points[1][k++] = points[1][i];

	            circle( image, points[1][i], markerThickness, Scalar(0,255,0), -1, 8);
	            circle( maskNose, points[1][i], markerThickness, Scalar(128,128,128), -1, 8);
	        }
	        ofstream myfile ("datos.txt",ios::app);
//	        cout<<modelPoints[0].x<<"	"<<modelPoints[1].x<<"	"<<modelPoints[2].x<<"	"<<modelPoints[3].x<<endl;
//	        cout<<projectedPoints[0].x<<"	"<<projectedPoints[1].x<<"	"<<projectedPoints[2].x<<"	"<<projectedPoints[3].x<<endl;
//	       	cout <<ANGLE(_prvec[0])<<"	"<<ANGLE(_prvec[1])<<"	"<<ANGLE(_prvec[2])<<endl;
	       	t = s;
	       	t += sprintf(s,"T  %.1f %.1f %.1f ",_ptvec[0],_ptvec[1],_ptvec[2]);
	       	t += sprintf(t,"R %.0f %.0f %.0f ",ANGLE(_prvec[0]),ANGLE(_prvec[1]),ANGLE(_prvec[2]));
	       	putText(image, s, cvPoint(10,20), 1, 0.8, cvScalar(25,25,25), 1, 0);
	       	myfile<<_prvec[0]<<","<<_prvec[1]<<","<<_prvec[2]<<endl;
	       	myfile.close();
	        points[1].resize(k);
	    }

	    if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
	    {
	    	vector<Point2f> tmp;
	        tmp.push_back(pt);
	        cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
	        points[1].push_back(tmp[0]);
	        addRemovePt = false;
	    }
	    if (faces.size()!=0 && points[0].size()>MAX_INIT)
	    	needToInit = false;
	    if(points[0].size()<2)
	    	needToInit = true;




	    if(!points[0].empty())
	    {
	    	Moments m = moments(points[0],false);
	    	xCM=m.m10/m.m00;
	    	yCM=m.m01/m.m00;
//	    	cout<<"x del CM: "<<xCM<<"	y del CM"<<yCM<<endl;
//	    	circle( image, cvPoint(xCM,yCM), 5, Scalar(255,0,0), -1, 8);
//	    	circle(maskNose, cvPoint(xCM,yCM), 5, Scalar(255,0,0), -1, 8);
	    }
	    if(!frame.empty())
	   	{
	   	    imshow("Head tracking", image);
	   	    imshow("Mascara",maskNose);

	   	}


	    char c = (char)waitKey(10);
	    if( c == 27 )
	    	break;
	    switch( c )
	    {
	    	case 'r':
	            needToInit = true;
	            break;
	        case 'c':
	            points[1].clear();
	            break;
	        case 'n':
	            nightMode = !nightMode;
	            break;
	        default:
	            ;
	        }


	    std::swap(points[1], points[0]);
	    swap(prevGray, gray);
	}

	return 0;
}
