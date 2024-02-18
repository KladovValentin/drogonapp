#ifndef HISTSUPDATER_H
#define HISTSUPDATER_H

#include "controllerBase.h"
#include "globals.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <sstream>
#include <vector>
#include <filesystem>

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
#include "THttpServer.h"
#include "TThread.h"
#include "TApplication.h"


using namespace std;

class HistsUpdater {
    protected:
        THttpServer* _serv;
        std::shared_ptr<ControllerBase> controllerBase;

        /// ___ ( Hists definitions ) ____
        //TGraph* trainingCheckGraph;
        TGraph* trainingTargetsGraph;
        TGraph* trainingPredictionsGraph;
        TGraph* predictionGraph;
        TGraph* predictionVsTargetGraph;
        TGraph* predictionGraphNodes[24];
        TGraphErrors* targetsGtaphNodes[24];
        TGraph* targetsPredictions;
        TGraph* targetsPredictionsNodes[24];
        //TCanvas* trainingCheckCanvas;


    public:
        //HistsUpdater(THttpServer* serv, ControllerBase* controllerBase);
        HistsUpdater(int argc, char** argv, std::shared_ptr<ControllerBase>& icontrollerBase);
        ~HistsUpdater();

        void updateTrainingCheckGraph();
        void updateHists(vector< pair<int, vector<float> > > newPredictions);
        void updateTargetGraphs();
        vector<int> loadProcessedrunlist(int run1, int run2);
        void updateProcessedRunList(vector<int> newRunNumbers);

        int updateMainCicle();
};


#endif