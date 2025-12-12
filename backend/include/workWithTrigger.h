#ifndef WORKWITHTRIGGER_H
#define WORKWITHTRIGGER_H

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


class TriggerDataManager {
    protected:
        vector<int> channel_ids;
        string histsLocation;
    public:
        TriggerDataManager();
        ~TriggerDataManager();
        
        void changeChannels(vector<int> newIDs);
        void changeHistsLocation(string newLocation);
        vector<double> getTriggerDataBase(TString qualityFile);        
        vector< vector<double> > getTriggerData(int run);
        vector< vector<double> > getTriggerDataList(TString list);
        vector< vector<double> > getTriggerDataLists(TString listOfLists);
        vector< vector<double> > getTriggerData(string date1, string date2);

        void make_list(const char *dirname, const char *outFName);
        void make_lists();
        void makeTableWithTriggerData();
};

#endif