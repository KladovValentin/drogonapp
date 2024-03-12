#ifndef CONTROLLERBASE_H
#define CONTROLLERBASE_H

#include "workWithTrigger.h"
#include "workWithDB.h"
#include "neuralNetwork.h"
#include "serverData.h"
//#include "histsUpdater.h"
//#include "continuousPredictor.h"

// ROOT
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TTree.h"
#include <TLorentzVector.h>
#include "TProfile.h"

class ControllerBase {
    private:
        vector<float> runningNnInpTens;
        int sentenceLength;
        int nodesLength;
        int inChannelsLength;

        TFile* out ;  // pointer to outputfile
        TH1F* hTriggerDataTime;
        TH1F* hEpicsDataTime;
        TH1F* hNetworkDataTime;

    public:
        TriggerDataManager* triggerManager;
        EpicsDBManager* epicsManager;
        NeuralNetwork* neuralNetwork;
        ServerData* serverData;
        //std::shared_ptr<ServerData> serverData;

        //HistsUpdater* fHistsUpdater;
        //ContinuousPredictor* fContinuousPredictor;

        //ControllerBase(TriggerDataManager* triggerManager, EpicsDBManager* epicsManager, NeuralNetwork* neuralNetwork, ServerData* serverData);
        ControllerBase();
        ~ControllerBase();

        void checkNewSettingsConfig();

        void setNewSettingsConfig();

        void compareTargetPredictionFromTraining();

        vector<float> makeNNInputTensor(int run);

        vector<float> moveForwardCurrentNNInput();

        void drawManyPredictions();

        void changeRunList();
        
        void writeData();

        int getSentenceLength() { return sentenceLength; }
        int getNodesLength() { return nodesLength; }
        int getInChannelsLength() { return inChannelsLength; }

};

#endif