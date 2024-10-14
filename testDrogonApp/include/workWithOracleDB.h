#ifndef WORKWITHORACLEDB_H
#define WORKWITHORACLEDB_H

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

#include <oci.h>


using namespace std;

class OracleDBManager {
    protected:
        string base_string;
        int listeningPort;

        OCIEnv *envhp;
        OCIError *errhp;
        OCISvcCtx *svchp;
        OCIStmt *stmthp;
        OCIDefine *defnp = (OCIDefine *) 0;

    public:
        OracleDBManager();
        ~OracleDBManager();

        void checkerr(OCIError *errhp, sword status);

        void connectToDB();

        
        
        int getDBdataBasic(); 
};


#endif