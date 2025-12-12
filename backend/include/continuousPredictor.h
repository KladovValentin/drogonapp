#ifndef CONTINUOUSPREDICTOR_H
#define CONTINUOUSPREDICTOR_H

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
#include <thread>
#include <chrono>

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

class ContinuousPredictor {
    protected:
        std::shared_ptr<ControllerBase> controllerBase;

    public:
        ContinuousPredictor(std::shared_ptr<ControllerBase>& icontrollerBase);
        ~ContinuousPredictor();

        void start();
};


#endif