#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;





PCA loadPCA(const char* fileName, int& rows, int& cols,Mat& pcaset){
    FILE* in = fopen(fileName,"r");
    int a;
    fscanf(in,"%d%d",&rows,&cols);

    pcaset = Mat::eye(rows,cols,CV_64F);
    int i,j;
    i=j=0;

    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            fscanf(in,"%d",&a);
            pcaset.at<double>(i,j) = a;
        }
    }
    cout << pcaset << endl;

    PCA pca(pcaset, // pass the data
        Mat(), // we do not have a pre-computed mean vector,
        // so let the PCA engine to compute it
        CV_PCA_DATA_AS_ROW, // indicate that the vectors
        // are stored as matrix rows
        // (use CV_PCA_DATA_AS_COL if the vectors are
        // the matrix columns)
        pcaset.cols// specify, how many principal components to retain
        );
    return pca;
}
