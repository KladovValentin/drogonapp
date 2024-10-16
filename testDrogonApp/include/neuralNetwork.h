#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <Python.h>

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
#include "TH2.h"
#include "TProfile.h"

using namespace std;


class NeuralNetwork {
    protected:
        Ort::Session* mSession;
        Ort::Env* env;
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
        vector<float> getRawPredictionPython(vector<float> normalizedInput);
        vector<float> getPrediction(vector<float> inputTensorValues);   

        void drawInputTargetCorrelations(TGraphErrors* target, vector< vector<double> > inputs);
        void drawInputTargetCorrelations(vector< vector<double> > targets, vector< vector<double> > inputs);
        void drawTargetStability(TGraphErrors* targetsAll, TGraphErrors* targetsPP, TGraphErrors* targetsHH);
        void drawTargetSectorComparison();
        TCanvas* drawTargetDimensionsComp(std::map< int, vector<double> > meanToTModSec, vector<int> shape);

        int remakeInputDataset(bool draw);
        void retrainModel();

        vector<float> formNNInput(vector<double> db, vector<double> tr);
        vector<float> formNNInput(vector<double> db);
};


#endif