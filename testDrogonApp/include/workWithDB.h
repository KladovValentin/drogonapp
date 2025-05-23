#ifndef WORKWITHDB_H
#define WORKWITHDB_H


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <sstream>
#include <vector>
#include <filesystem>
#include <numeric>

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

#include <pqxx/pqxx>

using namespace std;

class EpicsDBManager {
    protected:
        vector<string> channel_ids;
        vector<string> channel_names;
        vector<size_t> shape;
        vector< pair <vector<int>, string> > dbnames;
        vector< vector<int> > allToUniqueMapping;
        pqxx::connection* pqxxConnection;
        string base_string;
        string chan_string;
        int listeningPort;
    public:
        EpicsDBManager(int port);
        ~EpicsDBManager();

        void connectToDB();
        
        void remakeChannelIds();
        void changeChannelNames(vector<size_t> shapeNew, vector< pair <vector<int>, string> > dbnamesNew);
        void changeListeningPort(int newPort);

        vector<double> getDBdataBase(string command, string dateLeft, string dateLimit);        
        vector<double> getDBdata(int nLast, string channel);
        vector<double> getDBdata(int run, int runnext);
        vector<double> getDBdata(string date1, string date2);
        vector<double> getDBdata(string date1, string date2, string dateLimit);

        void appendDBTable(string mode, int runl, vector<double> dbPars);
        void makeTableWithEpicsData(string mode, int runl, int runr);
        void makeTableWithEpicsDataExtended(string mode, int runl, int runr, string fileName);
        void makeTableWithEpicsDataContinuousSplit(string mode, int runl, int runr, string fileName);
};


#endif