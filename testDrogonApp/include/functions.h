#ifndef FUNCTIONS_H
#define FUNCTIONS_H


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


namespace stringFunctions {
    extern vector<string> tokenize(string str, const char *delim);
};

namespace vectorFunctions {
    extern vector<double> runningMeanVector(vector<double> vect, int avRange);
    extern vector<double> normalizeVectorNN(vector<double> vect);
    extern double normalizeVector(vector<double>& vect);
    extern vector<float> transposeTensorDimensions(vector<float> inputTensor, vector<long int> dims, vector<size_t> permutation);
};


namespace dateRunF {
    extern double dateToNumber(string date);
    extern string runToDate(int run);
    extern double runToDateNumber(int run);
    extern TString runToFileName(int run);

    extern int getRunIdFromAnyFileName(string fName, size_t leftCut, size_t rightCut);
    extern int getRunIdFromFileName(string fName);
    extern int getRunIdFromDQFileName(string fName);
    extern int getRunFromFullDQFile(TString inFile);

    extern vector<double> timeVectToDateNumbers(vector<double> vect);

    extern void saveRunNumbers(string expFilesLocation);
    extern vector<int> loadrunlist(int run1, int run2);

    extern vector< pair<int,int> > loadrunlistWithEnds(int run1, int run2);

    extern vector<TString> getlistOfFileNames(TString inDir); 
};

namespace preTrainFunctions {
    extern vector< pair< int, vector<double> > > readClbTableFromFile(string fName);
    extern vector< pair< int, vector<double> > > readClbTableFromFileExtended(string fName);
    extern vector< pair< string, vector<double> > > readDBTableFromFile(string fName);
    extern vector< vector<double> > clbTableToVectorsTarget(vector< pair< int, vector<double> > > intable);
    extern map< int, vector<double> > clbTableToVectorsTarget(vector< pair< int, vector<double> > > intable, vector<int> shape);
    extern vector< vector<double> > clbTableToVectorsInput(vector< pair< int, vector<double> > > intable);
    extern vector< vector<double> > dbTableToVectorsAveraged(vector< pair< string, vector<double> > > table);
};


#endif