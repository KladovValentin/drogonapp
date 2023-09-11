#include "TFile.h"
#include <TH1.h>
#include <TH2.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TGraphErrors.h>
#include <TLine.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TLegend.h>

//#include "include/functions.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>

using namespace std;

vector<string> tokenize(string str, const char *delim){
    vector<string> result;
    TString line = str;
    TString tok;
    Ssiz_t from = 0;
    while (line.Tokenize(tok, from, delim))
        result.push_back((string)tok);
    return result;
}

vector<double> runningMeanVector(vector<double> vect, int avRange){

    double maxdiff = *max_element(vect.begin(), vect.end()) - *min_element(vect.begin(), vect.end());
    TH1 *hStep = new TH1F("hStep",";diff;counts",300,-maxdiff/5,maxdiff/5);
    for (size_t i = 0; i < vect.size()-1; i++){
        if (vect[i] > 0 && vect[i+1] > 0){
            hStep->Fill(vect[i+1]-vect[i]);
        }
    }
    double stepCut = 1.5*hStep->GetRMS();
    double stepMean = hStep->GetMean();
    
    cout << "std and mean   " << hStep->GetRMS() << "    " << hStep->GetMean() << endl;
    //hStep->Draw();
    //TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);
    //c->Update();
    //cin.get();cin.get();

    vector<bool> goodForAveraging;
    goodForAveraging.push_back(false);
    for (size_t i = 0; i < vect.size()-1; i++){
        if (fabs(vect[i+1]-vect[i] - stepMean) > stepCut){
            goodForAveraging.push_back(false);            
        }
        else {
            goodForAveraging.push_back(true);
        }
    }
    vector<double> result;
    for (size_t i = 0; i < vect.size(); i++){
        if (!goodForAveraging[i]){
            result.push_back(vect[i]);
            continue;
        }

        double mean2 = 0;
        int count1 = 0;
        for (size_t j = 0; j < avRange+1; j++){
            if ((i+j-avRange/2<0) || (i+j-avRange/2>=vect.size()) || (fabs(vect[i+j-avRange/2]-vect[i] - stepMean) > stepCut))
                continue;
            count1 +=1;
            mean2+=vect[i+j-avRange/2];
        }
        mean2 = mean2/count1;
        if (count1 == 0)
            mean2 = vect[i];
        result.push_back(mean2);
    }

    return result;
}

double dateToNumber(string date){
    vector<string> tokens = tokenize(date, "[-]");
    vector<double> segments;
    for (size_t i = 0; i < tokens.size(); i++)
        segments.push_back(atof(tokens[i].c_str()));
    double baseTime = 22*30*12 + 30 + 2 + 15./24. + 58./24./60.;
    return ((segments[0]-2000)*30*12 +(segments[1]-1)*30.+ (segments[2]-1) + segments[3]/24. + segments[4]/24./60. + segments[5]/24./60./60. - baseTime)*24*60*60;
}

string runToDate(int run){
    int day = (((int)(run-443670000)/86400) - 28*(((int)(run-443670000)/86400)/28) + 1);
    int month = day/29+2;
    int runMDay = (run-443670000)%86400;
    int hour = (int)(runMDay/3600);
    int minute = (runMDay%3600)/60;
    int second = (runMDay%3600)%60;
    string hourstr = hour<10 ? "0"+to_string(hour) : to_string(hour);
    string minutestr = minute<10 ? "0"+to_string(minute) : to_string(minute);
    string secondstr = second<10 ? "0"+to_string(second) : to_string(second);
    string monthstr = month<10 ? "0"+to_string(month) : to_string(month);
    string daystr = day<10 ? "0"+to_string(day) : to_string(day);
    string date = "2022-" + monthstr + "-" + daystr + " " + hourstr + ":" + minutestr + ":" + secondstr;
    return date;
}

double runToDateNumber(int run){
    int day = ((int)(run-443670000)/86400 + 1);
    int runMDay = (run-443670000)%86400;
    int hour = (int)(runMDay/3600);
    int minute = (runMDay%3600)/60;
    int second = (runMDay%3600)%60;
    string date = "2022-02-" + to_string(day)+"-"+to_string(hour)+"-"+to_string(minute)+"-"+to_string(second);
    //cout << run << "    " << date << endl;
    return dateToNumber(date);
}


void remakeMDCAll1(){
    std::ifstream fi("serverData/info_tables/MDCALL1.dat");
    std::ofstream fo("serverData/info_tables/MDCALL12.dat");

    string line1;
    while (std::getline(fi, line1)) {

        std::istringstream iss1(line1);
        int run;
        vector<double> pars;
        iss1 >> run;
        double par;
        while (iss1 >> par){
            pars.push_back(par);
        }
        fo << run << "\t";
	int indexLinks[7] = {0,1,2,5,3,4,6};
        for (size_t i = 0; i < 7; i++){
            for (size_t j = 0; j < 6; j++){
                fo << pars[7*j+indexLinks[i]] << "\t";
            }
        }
        fo << endl;

    }
    fi.close();
    fo.close();
}

void remakeMDCALL(){

    std::ifstream fileSep("serverData/info_tables/MDCALLSec.dat");
    std::ifstream fileCom("serverData/info_tables/MDCALL.dat");
    std::ofstream outputFile("serverData/info_tables/MDCALL1.dat");

    if (!fileSep.is_open() || !fileCom.is_open() || !outputFile.is_open()) {
        std::cerr << "Failed to open files." << std::endl;
        return;
    }

    std::string line1, line2;
    while (std::getline(fileCom, line1) && std::getline(fileSep, line2)) {
        // Assuming lines are separated by tabs

        std::istringstream iss1(line1);
        std::string cell1;
        int columnNumber = 0;
        string strCom = "";
        string run = "";
        while (iss1 >> cell1) {
            // Select and append 1st, 3rd, and 4th columns
            if (columnNumber == 1 || columnNumber == 2 || columnNumber == 3 || columnNumber == 5 || columnNumber == 6) {
                //partsStrCom.push_back(cell1);
                strCom = strCom + cell1 + "\t";
                //outputFile << cell1 << "\t";
            }
            if (columnNumber == 0)
                run = cell1;
            columnNumber++;
        }


        std::istringstream iss2(line2);
        string value;
        
        // Read the 1st, 3rd, and 4th columns
        iss2 >> value; // 1st column
        if (value != run)
            cout << value << "  " << run << endl;
        
        outputFile << run << "\t";
        for (size_t i = 0; i < 6; i++){
            outputFile << strCom << "\t";
            iss2 >> value;
            outputFile << value << "\t";
            iss2 >> value;
            outputFile << value << "\t";
        }
        outputFile << endl;
    }

    // Close the files
    fileSep.close();
    fileCom.close();
    outputFile.close();
}



void drawTimeConsumptions(){
    TFile *f1 = TFile::Open("outHome.root");
    TFile *f2 = TFile::Open("out.root");
    TH1F *h1 = (TH1F*)f2->Get("hEpicsDataTime");
    TH1F *h2 = (TH1F*)f2->Get("hTriggerDataTime");
    TH1F *h3 = (TH1F*)f1->Get("hNetworkDataTime");

    h1->SetLineWidth(2);
    h2->SetLineWidth(2);
    h3->SetLineWidth(2);

    //h1->SetStats(false);
    //h2->SetStats(false);
    //h3->SetStats(false);

    TCanvas *canvas = new TCanvas("canvas", "Histograms on Pads", 800, 600);
    canvas->Divide(3, 1);
    canvas->cd(1);
    h1->Draw();
    canvas->cd(2);
    h2->Draw();
    canvas->cd(3);
    h3->Draw();
    canvas->Update();
}

void drawTargetPredictionComparison(){
    const std::string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";
    /// Reading target values and averaging -> map
    ifstream fin2;
	fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCSecAllNew.dat").c_str());
    std::map<int, pair<double,double> > targets;
    vector<int> runsTargets;
    vector<double> valuesTargets,runsTargetsErr,valuesTargetsErr;
    while (!fin2.eof()){
        int run, sector;
        double target, targetErr;
        fin2 >> run >> sector >> target >> targetErr;
        if (sector == 0){
            //cout << run << endl;
            targets[run] = make_pair(target,targetErr);
            runsTargets.push_back(run);
            valuesTargets.push_back(target);
            runsTargetsErr.push_back(0);
            valuesTargetsErr.push_back(targetErr);
        }
    }
    fin2.close();

    TGraphErrors* gr2 = new TGraphErrors();
    gr2->SetMarkerStyle(21);
    gr2->SetMarkerSize(0.4);
    gr2->SetMarkerColor(4);
    gr2->SetLineColor(4);
    gr2->SetLineWidth(1);

    //valuesTargets = runningMeanVector(valuesTargets, 20);
    for (size_t i = 0; i < runsTargets.size(); i++){
        //targets[runsTargets[i]] = valuesTargets[i];
        int n = gr2->GetN();
        //gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),targets[runsTargets[i]].first);
        gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),valuesTargets[i]);
        gr2->SetPointError(n, runsTargetsErr[i], valuesTargetsErr[i]);
    }
    cout << gr2->GetN() << endl;
    cout << "finished with target " << endl;

    ifstream fin1;
    fin1.open((saveLocation+ "function_prediction/predictedSec1tSepLSTM.txt").c_str());
	//fin1.open((saveLocation+ "function_prediction/predicted.txt").c_str());
    vector<double> runsPredicted, runsPredictedRuns;
    vector<double> valuesPredicted;
    while (!fin1.eof()){
        double run, prediction;
        double p2,p3;
        fin1 >> run >> prediction >> p2 >> p3;
        runsPredictedRuns.push_back(run);
        runsPredicted.push_back((double)runToDateNumber(run));
        valuesPredicted.push_back(prediction);
    }
    fin1.close();

    fin2.open((saveLocation+ "function_prediction/predicted1Sec1tSepLSTM.txt").c_str());
    //fin2.open((saveLocation+ "function_prediction/predicted1.txt").c_str());
    vector<double> runsPredictedP, runsPredictedRunsP;
    vector<double> valuesPredictedP;
    while (!fin2.eof()){
        double run, prediction;
        double p2,p3;
        fin2 >> run >> prediction >> p2 >> p3;
        runsPredictedRunsP.push_back(run);
        runsPredictedP.push_back((double)runToDateNumber(run));
        valuesPredictedP.push_back(prediction);
    }
    fin2.close();

    //fin1.open((saveLocation+ "function_prediction/predictedSec1tSepLSTM.txt").c_str());
    fin1.open((saveLocation+ "function_prediction/predicted.txt").c_str());
    vector<double> runsPredictedL, runsPredictedRunsL;
    vector<double> valuesPredictedL;
    while (!fin1.eof()){
        double run, prediction;
        fin1 >> run >> prediction;
        runsPredictedRunsL.push_back(run);
        runsPredictedL.push_back((double)runToDateNumber(run));
        valuesPredictedL.push_back(prediction);
    }
    fin1.close();

    //fin2.open((saveLocation+ "function_prediction/predicted1Sec1tSepLSTM.txt").c_str());
    fin2.open((saveLocation+ "function_prediction/predicted1.txt").c_str());
    vector<double> runsPredictedPL, runsPredictedRunsPL;
    vector<double> valuesPredictedPL;
    while (!fin2.eof()){
        double run, prediction;
        fin2 >> run >> prediction;
        runsPredictedRunsPL.push_back(run);
        runsPredictedPL.push_back((double)runToDateNumber(run));
        valuesPredictedPL.push_back(prediction);
    }
    fin2.close();

    vector<double> commCopy = runningMeanVector(valuesPredicted, 20);
    vector<double> singCopy = runningMeanVector(valuesPredictedL, 20);


    /// _____ making graphErrors for prediction-target
    vector<double> runsPredictedTarget, diffPredictedTarget, runsPredictedTargetErr, diffPredictedTargetErr;
    for (size_t i = 0; i < runsPredicted.size(); i++){
        //cout << (int)runsPredictedRuns[i] << endl;
        if (targets.find((int)runsPredictedRuns[i]) == targets.end())
            continue;
        if (targets[runsPredictedRuns[i]].second == 0)
            continue;
        runsPredictedTarget.push_back(runsPredicted[i]);
        runsPredictedTargetErr.push_back(0);
        diffPredictedTarget.push_back( (valuesPredicted[i] - targets[runsPredictedRuns[i]].first)/targets[runsPredictedRuns[i]].second );
        diffPredictedTargetErr.push_back(0);
    }
    vector<double> runsPredictedTargetP, diffPredictedTargetP, runsPredictedTargetErrP, diffPredictedTargetErrP;
    for (size_t i = 0; i < runsPredictedP.size(); i++){
        if (targets.find(runsPredictedRunsP[i]) == targets.end())
            continue;
        if (targets[runsPredictedRunsP[i]].second == 0)
            continue;
        runsPredictedTargetP.push_back(runsPredictedP[i]);
        runsPredictedTargetErrP.push_back(0);
        diffPredictedTargetP.push_back( (valuesPredictedP[i] - targets[runsPredictedRunsP[i]].first)/targets[runsPredictedRunsP[i]].second );
        diffPredictedTargetErrP.push_back(0);
    }

    TGraphErrors* grPT = new TGraphErrors(runsPredictedTarget.size(),&runsPredictedTarget[0], &diffPredictedTarget[0], &runsPredictedTargetErr[0], &diffPredictedTargetErr[0]);
    TGraphErrors* grPTP = new TGraphErrors(runsPredictedTargetP.size(),&runsPredictedTargetP[0], &diffPredictedTargetP[0], &runsPredictedTargetErrP[0], &diffPredictedTargetErrP[0]);
    grPT->SetMarkerStyle(22);    grPT->SetMarkerSize(1.0);    grPT->SetLineColor(3);    grPT->SetMarkerColor(3);    grPT->SetLineWidth(4);
    grPTP->SetMarkerStyle(22);   grPTP->SetMarkerSize(1.0);   grPTP->SetLineColor(1);   grPTP->SetMarkerColor(1);   grPTP->SetLineWidth(4);


    TGraph* gr1 = new TGraph(runsPredicted.size(),&runsPredicted[0], &valuesPredicted[0]);
    TGraph* grP = new TGraph(runsPredictedP.size(),&runsPredictedP[0], &valuesPredictedP[0]);
    TGraph* gr1L = new TGraph(runsPredictedL.size(),&runsPredictedL[0], &valuesPredictedL[0]);
    TGraph* grPL = new TGraph(runsPredictedPL.size(),&runsPredictedPL[0], &valuesPredictedPL[0]);


    gr1->SetMarkerStyle(22);
    gr1->SetMarkerSize(1.0);
    gr1->SetLineColor(2);
    gr1->SetMarkerColor(2);
    gr1->SetLineWidth(4);

    grP->SetMarkerStyle(22);
    grP->SetMarkerSize(1.0);
    grP->SetLineColor(2);
    grP->SetMarkerColor(2);
    grP->SetLineWidth(4);

    gr1L->SetMarkerStyle(22);
    gr1L->SetMarkerSize(1.0);
    gr1L->SetLineColor(3);
    gr1L->SetMarkerColor(3);
    gr1L->SetLineWidth(4);

    grPL->SetMarkerStyle(22);
    grPL->SetMarkerSize(1.0);
    grPL->SetLineColor(1);
    grPL->SetMarkerColor(1);
    grPL->SetLineWidth(4);

    TDatime da(2022,2,3,15,58,00);
    gStyle->SetTimeOffset(da.Convert());
    gr2->GetXaxis()->SetTimeDisplay(1);
    gr2->GetXaxis()->SetTimeFormat("%d/%m");

    gr1->GetXaxis()->SetTimeDisplay(1);
    gr1->GetXaxis()->SetTimeFormat("%d/%m");

    grP->GetXaxis()->SetTimeDisplay(1);
    grP->GetXaxis()->SetTimeFormat("%d/%m");

    gr1L->GetXaxis()->SetTimeDisplay(1);
    gr1L->GetXaxis()->SetTimeFormat("%d/%m");

    grPL->GetXaxis()->SetTimeDisplay(1);
    grPL->GetXaxis()->SetTimeFormat("%d/%m");

    grPT->GetXaxis()->SetLimits(grPT->GetX()[0]-100000, grPT->GetX()[runsPredictedTarget.size()-1]*1.3);
    grPT->GetYaxis()->SetRangeUser(-6,6);
    grPT->GetXaxis()->SetTimeDisplay(1); grPT->GetXaxis()->SetTimeFormat("%d/%m");
    grPTP->GetXaxis()->SetTimeDisplay(1); grPTP->GetXaxis()->SetTimeFormat("%d/%m");

    TCanvas* canvas = new TCanvas("canvas", "Prediction vs Target calibration comparison", 1920, 950);
    TPad* pad = new TPad("pad", "Pad", 0.01, 0.01, 0.99, 0.99);
    pad->Draw();
    pad->cd();
    pad->SetGridy();

    gr2->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    gr1->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    grP->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    grPT->SetTitle("Prediction - Target;date;#sigma");
    grPTP->SetTitle("Prediction - Target;date;#sigma");

    TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    //legend->AddEntry(gr2, "target calibration", "l");
    //legend->AddEntry(gr1, "predicted, train dataset", "l");
    //legend->AddEntry(grP, "predicted, test dataset", "l");
    legend->AddEntry(grPT, "train dataset", "l");
    legend->AddEntry(grPTP, "test dataset", "l");

    //gr2->GetYaxis()->SetRangeUser(-5,5);
    //gr2->Draw("AP");
    //gr1->Draw("Psame");
    //grP->Draw("Psame");
    //gr1L->Draw("Psame");
    //grPL->Draw("Psame");
    grPT->Draw("AP");
    grPTP->Draw("Psame");
    legend->Draw();

    TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);
    c->Update();
}
