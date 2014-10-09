#include "opencv/cv.h"
#include "opencv2/ml//ml.hpp"

// The neural network
CvANN_MLP machineBrain;

// Read the training data and train the network.
void trainMachine()
{
   int i;
   //The number of training samples.
   int train_sample_count;

   //The training data matrix.
   //Note that we are limiting the number of training data samples to 1000 here.
   //The data sample consists of two inputs and an output. That's why 3.
   //td es la matriz dinde se cargan las muestras
   float td[3000][7];

   //Read the training file
   /*
    A sample file contents(say we are training the network for generating
    the mean given two numbers) would be:

    5
    12 16 14
    10 5  7.5
    8  10 9
    5  4  4.5
    12 6  9

    */
   FILE *fin;
   fin = fopen("train.txt", "r");

   //Get the number of samples.
   fscanf(fin, "%d", &train_sample_count);
   printf("Found training file with %d samples...\n", train_sample_count);

   //Create the matrices

   //Input data samples. Matrix of order (train_sample_count x 2)
   CvMat* trainData = cvCreateMat(train_sample_count, 6, CV_32FC1);

   //Output data samples. Matrix of order (train_sample_count x 1)
   CvMat* trainClasses = cvCreateMat(train_sample_count, 1, CV_32FC1);

   //The weight of each training data sample. We'll later set all to equal weights.
   CvMat* sampleWts = cvCreateMat(train_sample_count, 1, CV_32FC1);

   //The matrix representation of our ANN. We'll have four layers.
   CvMat* neuralLayers = cvCreateMat(2, 1, CV_32SC1);

   CvMat trainData1, trainClasses1, neuralLayers1, sampleWts1;

   cvGetRows(trainData, &trainData1, 0, train_sample_count);
   cvGetRows(trainClasses, &trainClasses1, 0, train_sample_count);
   cvGetRows(trainClasses, &trainClasses1, 0, train_sample_count);
   cvGetRows(sampleWts, &sampleWts1, 0, train_sample_count);
   cvGetRows(neuralLayers, &neuralLayers1, 0, 2);

   //Setting the number of neurons on each layer of the ANN
   /*
    We have in Layer 1: 2 neurons (6 inputs)
               Layer 2: 3 neurons (hidden layer)
               Layer 3: 3 neurons (hidden layer)
               Layer 4: 1 neurons (1 output)
    */
   cvSet1D(&neuralLayers1, 0, cvScalar(6));
   //cvSet1D(&neuralLayers1, 1, cvScalar(3));
   //cvSet1D(&neuralLayers1, 2, cvScalar(3));
   cvSet1D(&neuralLayers1, 1, cvScalar(1));

   //Read and populate the samples.
   for (i=0;i<train_sample_count;i++)
       fscanf(fin,"%f %f %f %f",&td[i][0],&td[i][1],&td[i][2],&td[i][3]);

   fclose(fin);

   //Assemble the ML training data.
   for (i=0; i<train_sample_count; i++)
   {
       //Input 1
       cvSetReal2D(&trainData1, i, 0, td[i][0]);
       //Input 2
       cvSetReal2D(&trainData1, i, 1, td[i][1]);
       cvSetReal2D(&trainData1, i, 2, td[i][2]);
       cvSetReal2D(&trainData1, i, 3, td[i][3]);
       cvSetReal2D(&trainData1, i, 4, td[i][4]);
       cvSetReal2D(&trainData1, i, 5, td[i][5]);
       //Output
       cvSet1D(&trainClasses1, i, cvScalar(td[i][6]));
       //Weight (setting everything to 1)
       cvSet1D(&sampleWts1, i, cvScalar(1));
   }

   //Create our ANN.
   machineBrain.create(neuralLayers);

   //Train it with our data.
   //See the Machine learning reference at http://www.seas.upenn.edu/~bensapp/opencvdocs/ref/opencvref_ml.htm#ch_ann
   machineBrain.train(
       trainData,
       trainClasses,
       sampleWts,
       0,
       CvANN_MLP_TrainParams(
           cvTermCriteria(
               CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
               100000,
               1.0
               ),
           CvANN_MLP_TrainParams::BACKPROP,
           0.01,
           0.05
           )
       );
}

// Predict the output with the trained ANN given the two inputs.
void Predict(float data0, float data1, float data2, float data3, float data4, float data5)
{
   float _sample[6];
   CvMat sample = cvMat(1, 6, CV_32FC1, _sample);
   float _predout[1];
   CvMat predout = cvMat(1, 1, CV_32FC1, _predout);
   sample.data.fl[0] = data0;
   sample.data.fl[1] = data1;
   sample.data.fl[2] = data2;
   sample.data.fl[3] = data3;
   sample.data.fl[4] = data4;
   sample.data.fl[5] = data5;

   machineBrain.predict(&sample, &predout);

   printf("%f \n",predout.data.fl[0]);

}



