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
	fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCAllNew.dat").c_str());
    std::map<int, pair<double,double> > targets;
    vector<int> runsTargets;
    vector<double> valuesTargets,runsTargetsErr,valuesTargetsErr;
    while (!fin2.eof()){
        int run;
        double target, targetErr;
        fin2 >> run >> target >> targetErr;
        targets[run] = make_pair(target,targetErr);
        runsTargets.push_back(run);
        valuesTargets.push_back(target);
        runsTargetsErr.push_back(0);
        valuesTargetsErr.push_back(targetErr);
    }
    fin2.close();

    TGraphErrors* gr2 = new TGraphErrors();
    gr2->SetMarkerStyle(21);
    gr2->SetMarkerSize(0.5);
    gr2->SetMarkerColor(4);
    gr2->SetLineColor(4);
    gr2->SetLineWidth(4);

    //valuesTargets = runningMeanVector(valuesTargets, 20);
    for (size_t i = 0; i < runsTargets.size(); i++){
        //targets[runsTargets[i]] = valuesTargets[i];
        int n = gr2->GetN();
        gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),targets[runsTargets[i]].first);
        gr2->SetPointError(n, runsTargetsErr[i], valuesTargetsErr[i]);
    }
    cout << gr2->GetN() << endl;
    cout << "finished with target " << endl;

    ifstream fin1;
	fin1.open((saveLocation+ "function_prediction/predicted.txt").c_str());
    vector<double> runsPredicted, runsPredictedRuns;
    vector<double> valuesPredicted;
    while (!fin1.eof()){
        double run, prediction;
        fin1 >> run >> prediction;
        runsPredictedRuns.push_back(run);
        runsPredicted.push_back((double)runToDateNumber(run));
        valuesPredicted.push_back(prediction);
    }
    fin1.close();

    fin2.open((saveLocation+ "function_prediction/predicted1.txt").c_str());
    vector<double> runsPredictedP, runsPredictedRunsP;
    vector<double> valuesPredictedP;
    while (!fin2.eof()){
        double run, prediction;
        fin2 >> run >> prediction;
        runsPredictedRunsP.push_back(run);
        runsPredictedP.push_back((double)runToDateNumber(run));
        valuesPredictedP.push_back(prediction);
    }
    fin2.close();

    /// _____ making graphErrors for prediction-target
    vector<double> runsPredictedTarget, diffPredictedTarget, runsPredictedTargetErr, diffPredictedTargetErr;
    for (size_t i = 0; i < runsPredicted.size(); i++){
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

    gr1->SetMarkerStyle(22);
    gr1->SetMarkerSize(1.0);
    gr1->SetLineColor(3);
    gr1->SetMarkerColor(3);
    gr1->SetLineWidth(4);

    grP->SetMarkerStyle(22);
    grP->SetMarkerSize(1.0);
    grP->SetLineColor(1);
    grP->SetMarkerColor(1);
    grP->SetLineWidth(4);

    TDatime da(2022,2,3,15,58,00);
    gStyle->SetTimeOffset(da.Convert());
    gr2->GetXaxis()->SetTimeDisplay(1);
    gr2->GetXaxis()->SetTimeFormat("%d/%m");

    gr1->GetXaxis()->SetTimeDisplay(1);
    gr1->GetXaxis()->SetTimeFormat("%d/%m");

    grP->GetXaxis()->SetTimeDisplay(1);
    grP->GetXaxis()->SetTimeFormat("%d/%m");

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

    gr2->GetYaxis()->SetRangeUser(-5,5);
    //gr2->Draw("AP");
    //gr1->Draw("Psame");
    //grP->Draw("Psame");
    grPT->Draw("AP");
    grPTP->Draw("Psame");
    legend->Draw();

    TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);
    c->Update();
}


void go(const char* filename = "mergedHistsExp34-48.root"){
    TFile* file = new TFile(filename, "READ");
    TH1F* hist = (TH1F*)file->Get("hMKNKP");

    TF1* f1 = new TF1("f1", "[0]*exp(-0.5*((x-[1])/[2])**2)+[3]*exp(-0.5*((x-[4])/[5])**2) + [7]*(x-[6]) + [8]*(x-[6])*(x-[6])",980,1400);
    f1->SetParameters(30, 1020, 10, 15, 1250, 50, 1000, -1, 1);
    f1->SetParLimits(0,10,100);
    f1->SetParLimits(1,1000,1050);
    f1->SetParLimits(2,1,10);
    f1->SetParLimits(3,1,50);
    f1->SetParLimits(4,1100,1300);
    f1->SetParLimits(5,30,150);
    hist->Fit("f1","","",980,1400);
    hist->Draw();
    f1->Draw("same");
    cout << (f1->GetParameter(0) * fabs(f1->GetParameter(2))) * (sqrt(3.14159 * 2.)) << endl;
}


void makeMixingMatrix(){
    TFile *f1 = TFile::Open("mergedHistsSim005_3.root");
    TH2I *h2 = (TH2I*)f1->Get("hMixingNN");
    TH1F *h1 = (TH1F*)f1->Get("hNNEvtCorrupt");
    int sums[6];
    for (size_t i = 0; i < 6; i++){
        sums[i] = 0;
        for (size_t j = 0; j < 6; j++){
            sums[i] += h2->GetBinContent(i+1,j+1);
        }
    }
    for (size_t j = 6; j >= 1; j--){
        for (size_t i = 0; i < 6; i++){
            if (sums[i]==0)
                cout << "0  ";
            else{
                cout << h2->GetBinContent(i+1,j)/sums[i] << " ";
            }
        }
        cout << endl;
    }
    h2->Draw();
}


double interpolateChiSquarePDF(double* x, double* p) {
    int lowerDof = static_cast<int>(p[1]);
    int upperDof = lowerDof + 1;

    int lowerDof1 = static_cast<int>(p[3]);
    int upperDof1 = lowerDof + 1;

    double lowerPDF = TMath::Prob(*x,lowerDof);//TMath::GammaDist(*x, lowerDof / 2.0, 0.5);
    double upperPDF = TMath::Prob(*x,upperDof);//TMath::GammaDist(*x, upperDof / 2.0, 0.5);

    double lowerPDF1 = TMath::Prob(*x,lowerDof1);//TMath::GammaDist(*x, lowerDof1 / 2.0, 0.5);
    double upperPDF1 = TMath::Prob(*x,upperDof1);//TMath::GammaDist(*x, upperDof1 / 2.0, 0.5);

    double chi0 = p[0] * (lowerPDF + (p[1] - lowerDof) * (upperPDF - lowerPDF));
    double chi1 = p[2] * (lowerPDF1 + (p[3] - lowerDof1) * (upperPDF1 - lowerPDF1));

    // Linear interpolation
    return chi0 + chi1 + p[4] + (*x)*p[5];
}

void fit4cChi2(){
    TFile *file0 = TFile::Open("mergedHistsExp34-48_N.root");
    TH1F *h1 = (TH1F*)file0->Get("h4cChi2");
    vector<double> x, y, xErr, yErr;
    for (size_t i = 1; i < 101; i++){
        x.push_back(h1->GetBinCenter(i));
        y.push_back((float)(h1->GetBinContent(i))/pow(10.,(float)i/20.-1.));
        //y.push_back(h1->GetBinContent(i));
        xErr.push_back(h1->GetBinWidth(i)/2);
        yErr.push_back(h1->GetBinError(i)/pow(10.,(float)i/20.-1.));
        //yErr.push_back(h1->GetBinError(i));
        h1->SetBinContent(i,(float)(h1->GetBinContent(i))/pow(10.,(float)i/20.-1.));
    }
    TGraphErrors* gr = new TGraphErrors(x.size(),&x[0],&y[0],&xErr[0],&yErr[0]);
    gr->SetTitle("#chi^{2} fit probability distribution");

    TF1* f1 = new TF1("f1", interpolateChiSquarePDF,0.1,10, 6);
    TF1* f2 = new TF1("f2", "[0]*x + [1]",0.1,200);
    TF1* f3 = new TF1("f3", "[0]*x + [1]",0.1,200);
    f1->SetParameters(200, 4, 600, 3000, 1, 10);
    f1->SetParLimits(0,10,1000);
    f1->SetParLimits(1,0,20);
    //f1->SetParLimits(2,100,100000);
    //f1->SetParLimits(3,500,10000);
    f1->FixParameter(2,0);
    f1->SetParLimits(4,0,100);
    f1->SetParLimits(5,0,50);

    gStyle->SetOptLogx();
    //gStyle->SetOptLogy();

    gr->Fit("f2","","",30,200);
    gr->Draw();
    TLine* l = new TLine(20,0,20,700);
    l->Draw("same");
    //f2->SetParameters(0.2562,64.3231);
    f2->Draw("same");

    //f1->Draw("same");
    //f2->SetParameters(f1->GetParameter(2), f1->GetParameter(3));
    //f2->Draw("same");
    //file0->Close();
    cout << "S/B approximate ratio  " << (h1->Integral(1,46) - (f2->GetParameter(0)*20./2.+f2->GetParameter(1))*20.)/((f2->GetParameter(0)*20./2.+f2->GetParameter(1))*20.) << endl;
}
