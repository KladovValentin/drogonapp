#ifndef SERVERDATA_H
#define SERVERDATA_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <map>
#include <iomanip>

#include <jsoncpp/json/json.h>

using namespace std;


class ServerData {
    private:

        /// @brief settings or ~constants
        string triggerHistsPath;
        int epicsDBPort;
        vector<int> trigChannels;
        vector< pair <vector<int>, string> > dbNames;
        vector<size_t> dbShape;


        /// @brief Dinamic information
        vector<int> runBorders;
        vector<float> currentNNInput;
        int currentRunIndex;
        bool continuousPredictionRunning;
        map<int, float> predictedValues;


    public:
        ServerData();
        ~ServerData();

        void readSettings();
        void writeSettings();
        void writeSettingsJSON();
        void readSettingsJSON();

        void readPredictedValues();
        void writePredictedValues();

        string getTrHistsLoc(){ return triggerHistsPath; }
        int getDbListeningPort(){ return epicsDBPort; }
        vector<int> getTrChannels(){ return trigChannels; }
        vector< pair <vector<int>, string> > getDbChannels(){ return dbNames; } 
        vector<size_t> getDbShape(){  return dbShape; }
        vector<int> getRunList(){ return runBorders; }
        vector<float> getCurrentNNInput(){ return currentNNInput; }
        int getCurrentRun(){ return runBorders[currentRunIndex]; }
        int getCurrentRunIndex(){ return currentRunIndex; }
        map<int, float> getPredictedValues() { return predictedValues; }

        bool getContinuousPredictionRunning() {return continuousPredictionRunning;}


        void setTrHistsLoc(string newLoc){ triggerHistsPath = newLoc; }
        void setDbListeningPort(int newPort){ epicsDBPort = newPort; }
        void setTrChannels(vector<int> newChannels){ trigChannels = newChannels; }
        void setDbChannels(vector<size_t> newShape, vector< pair <vector<int>, string> > newChannels){ dbNames = newChannels; dbShape = newShape; }
        void setCurrentNNInput(vector<float> newNNInput) {currentNNInput = newNNInput; }
        void setCurrentRunIndex(int newCurrentRunIndex) {currentRunIndex = newCurrentRunIndex; } 
        void setPredictedValues(map<int, float> newValues) { predictedValues = newValues;}

        void setContinuousPredictionRunning(bool set) { continuousPredictionRunning = set; }
};

#endif
