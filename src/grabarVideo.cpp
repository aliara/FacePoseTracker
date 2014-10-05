#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "constants.h"

using namespace cv;
using namespace std;


string currentTimeString(time_t *sec)
{
    struct tm *ptm = gmtime (sec);
    ostringstream ostr;
    ostr<<setfill('0')<<setw(4)<< (ptm->tm_year + 1900)<< "-" << setw(2)<< (ptm->tm_mon+1)
            << "-" << setw(2)<<ptm->tm_mday << "T"<<setw(2)<<ptm->tm_hour
            << "h" <<setw(2) << ptm->tm_min << "m" << setw(2) << ptm->tm_sec;
    return ostr.str();
}


VideoWriter createVideoFile(String vidDir, int width, int height, int fps, int fourcc, time_t sec)
{

    string timeStr = currentTimeString(&sec);
    string fileName = vidDir + timeStr + ".avi";
	cout << "Video file name = " << fileName << endl;
	Size frameSize = Size(width, height);
    VideoWriter vidWriter = VideoWriter(fileName, fourcc, fps,
        frameSize);
	return vidWriter;
}

void writeImageFile(string imgDir, Mat frame, string imgFmt, time_t sec)
{
    string timeStr = currentTimeString(&sec);
    string fileName = imgDir + timeStr + "." + imgFmt;
    cout << "Image file = " << fileName << endl;
    imwrite(fileName, frame);
}



void grabarVideo(Mat frame, VideoCapture cap)
{
	bool static isRecording = false;
	VideoWriter static writer;
	time_t static vidDelta = 0;


	int vidFps = 10;
	int fourcc = CV_FOURCC(vidCodec[0],vidCodec[1],vidCodec[2], vidCodec[3]);
	int imgInterval = 60; // seconds
	int imgNum = 0;
	time_t sec;
	long static frameNum = 0;
	bool isDisplayEnabled = false;
//	int delay = 1;
	int vidNum = 1;
	bool isRecordingEnabled = vidNum > 0 ? true : false;

	bool isImageCaptureEnabled = imgNum > 0 ? true : false;

	time_t vidTime = 20;

	int vidTotal = 0;
	time_t imgTime = 0;
	time_t imgDelta = 0;
	int imgTotal = 0;

	int vidInterval = 60; // seconds
	double fps = 0.0;

	    	sec = time(NULL);
	        frameNum++;

	        if (isDisplayEnabled)
	        {
	        	if(!frame.empty())
	        	imshow("Current Frame", frame);
	        }



	        // Decide whether to create new video file
	        if ((isRecordingEnabled) && (!isRecording))
	        {
	            int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	            int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	            writer = createVideoFile(vidDir, width, height, vidFps, fourcc, sec);
	            if(writer.isOpened())
	            {
	            	vidTime = sec;
	            	isRecording = true;
	            	frameNum = 0;
	            }
	            else
	            {
	            	cout<< "No se pudo abrir el directorio: "<<vidDir<<endl;
	            	isRecordingEnabled=false;
	            }
	        }

	        // Write frame to video, calculate time interval and whether or not to create new video file
	        if (isRecordingEnabled)
	        {
	            writer.write(frame);
	            vidDelta = sec - vidTime;
//	            cout << "vidDelta "<<vidDelta<<" >= "<<vidInterval<<endl;

	            if (vidDelta >= vidInterval) {
	//                isRecording = false;
	                vidTotal = vidTotal + 1;
//	                cout << "Videos recorded =" << vidTotal << "/" << vidNum << endl;
//	                cout << "vidTotal="<<vidTotal<<" vidNum="<<vidNum<<endl;

	                if (vidTotal >= vidNum) {
	                    isRecordingEnabled = false;

	                    if (vidDelta > 0) {
	                            fps = frameNum / vidDelta;
	                            frameNum = 0;
	                    }

//	                    cout << "Recording completed fps=" << fps << endl;

	                    if (isDisplayEnabled) {
	                            writer = VideoWriter();
	                    }

	                }
	            }

	        }

	        if (isImageCaptureEnabled) {
	            imgDelta = (sec - imgTime);

	            if (imgDelta >= imgInterval) {
	                writeImageFile(imgDir, frame, imgFmt, sec);
	                imgTime = sec;
	                imgTotal = imgTotal + 1;

	                if (imgTotal >= imgNum) {
	                    isImageCaptureEnabled = false;
	                }

	            }
	        }




}



