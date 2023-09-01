#ifndef SERVERDATA_H
#define SERVERDATA_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <map>

using namespace std;


class ServerData {
    private:
        string triggerHistsPath;
        int epicsDBPort;

        vector<int> trigChannels;
        vector<string> dbNames;

        vector<int> runBorders;
        vector<float> currentNNInput;
        int currentRunIndex;

        map<int, float> predictedValues;


    public:
        ServerData();
        ~ServerData();

        void readSettings();
        void writeSettings();

        void readPredictedValues();
        void writePredictedValues();

        string getTrHistsLoc(){ return triggerHistsPath; }
        int getDbListeningPort(){ return epicsDBPort; }
        vector<int> getTrChannels(){ return trigChannels; }
        vector<string> getDbChannels(){ return dbNames; } 
        vector<int> getRunList(){ return runBorders; }
        vector<float> getCurrentNNInput(){ return currentNNInput; }
        int getCurrentRun(){ return runBorders[currentRunIndex]; }
        int getCurrentRunIndex(){ return currentRunIndex; }
        map<int, float> getPredictedValues() { return predictedValues; }


        void setTrHistsLoc(string newLoc){ triggerHistsPath = newLoc; }
        void setDbListeningPort(int newPort){ epicsDBPort = newPort; }
        void setTrChannels(vector<int> newChannels){ trigChannels = newChannels; }
        void setDbChannels(vector<string> newChannels){ dbNames = newChannels; }
        void setCurrentNNInput(vector<float> newNNInput) {currentNNInput = newNNInput; }
        void setCurrentRunIndex(int newCurrentRunIndex) {currentRunIndex = newCurrentRunIndex; } 
        void setPredictedValues(map<int, float> newValues) { predictedValues = newValues;}
};

#endif
