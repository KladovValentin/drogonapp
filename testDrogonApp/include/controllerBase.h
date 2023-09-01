#ifndef CONTROLLERBASE_H
#define CONTROLLERBASE_H

#include "workWithTrigger.h"
#include "workWithDB.h"
#include "neuralNetwork.h"
#include "serverData.h"

// ROOT
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TTree.h"
#include <TLorentzVector.h>
#include "TProfile.h"

class ControllerBase {
    private:
        TriggerDataManager* triggerManager;
        EpicsDBManager* epicsManager;
        NeuralNetwork* neuralNetwork;
        ServerData* serverData;

        vector<float> runningNnInpTens;

        TFile* out ;  // pointer to outputfile
        TH1F* hTriggerDataTime;
        TH1F* hEpicsDataTime;
        TH1F* hNetworkDataTime;

    public:
        ControllerBase(TriggerDataManager* triggerManager, EpicsDBManager* epicsManager, NeuralNetwork* neuralNetwork, ServerData* serverData);
        ~ControllerBase();

        void compareTargetPredictionFromTraining();

        vector<float> makeNNInputTensor(int run);

        float moveForwardCurrentNNInput();

        void drawManyPredictions();

        void changeRunList();
        
        void writeData();

};

#endif