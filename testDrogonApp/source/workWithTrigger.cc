#include "../include/workWithTrigger.h"

#include "../include/functions.h"

#include "../include/constants.h"


#include "TFile.h"
#include <TSystemDirectory.h>
#include <TSystemFile.h>
#include <TList.h>

using namespace std;

using namespace dateRunF;


TriggerDataManager::TriggerDataManager(){
    channel_ids = {78,79,80,117,118}; 
    histsLocation = "/home/localadmin_jmesschendorp/lustre/hades/dst/feb22/gen2/";
}
TriggerDataManager::~TriggerDataManager(){}


void TriggerDataManager::changeChannels(vector<int> newIds){
    channel_ids = newIds;
}
void TriggerDataManager::changeHistsLocation(string newLocation){
    histsLocation = newLocation;
}


vector<double> TriggerDataManager::getTriggerDataBase(TString qualityFile){
    vector<double> result;

    TFile *file = new TFile(qualityFile);
    if(!file->IsOpen()){
        cout << "file " << qualityFile << " is not opened!" << endl;
        return result;
    }

    int run = getRunFromFullDQFile(qualityFile);
    file->cd();
    TH1F * hist = (TH1F*)file->Get("daqscl/histCTSScalerCh")->Clone();
    //TH1F * hist_time1 = (TH1F*)file->Get("daqscl/histTBoxScalerFileTime")->Clone();
    //TH1F * hist_time2 = (TH1F*)file->Get("daqscl/histEventHeaderFileTime")->Clone();
 
    /*
    float startOR=hist->GetBinContent(86);

    float PT1in=hist->GetBinContent(78);
    float PT1down=hist->GetBinContent(97);
    float PT1acc=hist->GetBinContent(116);
    
    float PT2in=hist->GetBinContent(79);
    float PT2down=hist->GetBinContent(98);
    float PT2acc=hist->GetBinContent(117);

    float PT3in=hist->GetBinContent(80);
    float PT3down=hist->GetBinContent(99);
    float PT3acc=hist->GetBinContent(118);

    int scal_t1=hist_time1->GetBinContent(1);
    int scal_t2=hist_time2->GetBinContent(1);
    */

    vector<double> valuesF;
    for (int id: channel_ids){
        valuesF.push_back(hist->GetBinContent(id));
    }
    bool badNumber = false;
    int zeroValues = 0;
    for (size_t k = 0; k < valuesF.size(); k++){
        //cout << valuesF[k] << endl;
        if (valuesF[k] == 0)
            zeroValues += 1;
        if (isnan(valuesF[k]) || isnan(-valuesF[k]) || isinf(valuesF[k]) || isinf(-valuesF[k]))
            badNumber = true;
    }
    if (zeroValues >= 2)
        badNumber = true;
    //cout << badNumber << endl;

    if (!badNumber){
        result.push_back(run);
        for (double value: valuesF)
            result.push_back(value);
    }
    file->Close();
    return result;
}


vector< vector<double> > TriggerDataManager::getTriggerData(string date1, string date2){
    //string command = "SELECT smpl_time,float_val FROM sample WHERE channel_id="+channel_ids[0]+" AND smpl_time >= '" + date1 + "' AND smpl_time <= '" + date2 + "' ORDER BY smpl_time ASC";
    return getTriggerData(1232143);
}

vector< vector<double> > TriggerDataManager::getTriggerData(int run){
    TString fileName = runToFileName(run);
    TString day = (TString)(((string)fileName).substr(4,3));
    vector<double> x = getTriggerDataBase((TString)histsLocation+ day +"/01/qa/"+fileName+"01.hld_dst_feb22_hist.root");
    vector< vector<double> > result;
    //for (size_t j = 0; j < x.size(); j++)
    //    cout << x[j] << endl;
    result.push_back(x);
    return result;
}

vector< vector<double> > TriggerDataManager::getTriggerDataList(TString list){
    vector< vector<double> > result;
    ifstream inFile1;
    inFile1.open(list);
    while (!inFile1.eof()){
        TString sFile;
        inFile1>>sFile;
        vector<double> tResult = getTriggerDataBase(sFile);
        if (tResult.size()>=1)
            result.push_back(tResult);
    }
    inFile1.close();
    return result;
}

vector< vector<double> > TriggerDataManager::getTriggerDataLists(TString listOfLists){
    vector< vector<double> > result;

    ifstream inFile;
    inFile.open(listOfLists);
    vector<TString> sLists;
    while (!inFile.eof()){
        TString sList;
        inFile >> sList;    
        sLists.push_back(sList);
    }    
    inFile.close();
    
    for (TString x: sLists){
        cout << "new day " << x << " out of " << sLists.size() << endl;
        vector< vector<double> > tResult = getTriggerDataList(x);
        for (vector<double> i: tResult)
            result.push_back(i);
    }

    return result;
}


void TriggerDataManager::make_list(const char *dirname, const char *outFName){
    TSystemDirectory dir(dirname, dirname);
    ofstream fout1; fout1.open(outFName);
    vector<string> filelist;

    /// iterating over files
    TSystemFile *file;
    TIter next(dir.GetListOfFiles());
    while ((file=(TSystemFile*)next())) {
        TString fname(file->GetName());
        if (file->IsDirectory() || !fname.EndsWith("_hist.root"))
            continue;

        string dirNameStr = (string)dirname; string fileNameStr = (string)fname.Data();
        filelist.push_back(dirNameStr+fileNameStr);
    }

    std::sort(filelist.begin(), filelist.end());

    for (size_t i = 0; i < filelist.size(); i++){
        if (i != 0)
            fout1 << endl;
        fout1 << filelist[i];
    }
    fout1.close();
    return;
}

void TriggerDataManager::make_lists(){

    TSystemDirectory dir("directory", (TString)(histsLocation));
    ofstream fout1; fout1.open((saveLocation+"histLists/listOfLists.list").c_str());
    TSystemFile *file;
    TIter next(dir.GetListOfFiles());
    int count = 0;
    vector<string> daylist;
    /// getting the list of directories
    while ((file=(TSystemFile*)next())) {
        TString fname(file->GetName());
        if (!file->IsDirectory() || !fname.Contains("0"))
            continue;
        daylist.push_back((string)fname.Data());
    }

    std::sort(daylist.begin(), daylist.end());

    for (size_t i = 0; i < daylist.size(); i++){
        if (i != 0)
            fout1 << endl;

        string dayTStr = daylist[i];
        string dirFName = (histsLocation + dayTStr + "/01/qa/");
        string outFName = (saveLocation+"histLists/filelist" + dayTStr + ".list");

        cout << ((string)("histLists/filelist" + dayTStr + ".list")).c_str() << endl;
        fout1 << ((string)(saveLocation+"histLists/filelist" + dayTStr + ".list")).c_str();
        make_list(dirFName.c_str(), outFName.c_str());
    }

    fout1.close();
    return;
}


void TriggerDataManager::makeTableWithTriggerData(){
    vector< vector<double> > test = getTriggerDataLists((TString)(saveLocation+"histLists/listOfLists.list"));
    ofstream outFile((saveLocation+"info_tables/trigger_data2.dat").c_str());
    for (size_t i = 0; i< test.size(); i++){
        outFile << (int)test[i][0] << " ";
        for (size_t j = 1; j < test[i].size(); j++){
            outFile << test[i][j] << " ";
        }
        outFile << endl;
    }
    outFile.close();
    return;
}
