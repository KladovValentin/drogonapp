#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "/home/localadmin_jmesschendorp/onnxruntime-linux-x64-1.14.1/include/onnxruntime_cxx_api.h"


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <sstream>
#include <vector>

#include "TString.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TAxis.h"
#include "TGaxis.h"
#include "TStyle.h"
#include "TDatime.h"
#include "TLegend.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TH1.h"

using namespace std;


class NeuralNetwork {
    protected:
        Ort::Session* mSession;
        size_t inputTensorSize;
        size_t outputTensorSize;
        std::vector<int64_t> mInputDims;
        std::vector<int64_t> mOutputDims;
        vector<double> meanValues;
	    vector<double> stdValues;

        vector<int> additionalFilter;

    public:
        NeuralNetwork();
        ~NeuralNetwork();

        void setupNNPredictions();
        float getPrediction(vector<float> inputTensorValues);   

        void drawInputTargetCorrelations(TGraphErrors* target, vector< vector<double> > inputs);
        void drawTargetStability(TGraphErrors* targetsAll, TGraphErrors* targetsPP, TGraphErrors* targetsHH);
        void drawTargetSectorComparison();

        void remakeInputDataset();
        void retrainModel();

        vector<float> formNNInput(vector<double> db, vector<double> tr);
};


#endif