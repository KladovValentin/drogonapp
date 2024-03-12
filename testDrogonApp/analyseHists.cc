#include "TFile.h"
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
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
    //TFile *f2 = TFile::Open("out.root");
    TH1F *h1 = (TH1F*)f1->Get("hEpicsDataTime");
    TH1F *h2 = (TH1F*)f1->Get("hTriggerDataTime");
    TH1F *h3 = (TH1F*)f1->Get("hNetworkDataTime");

    std::ifstream infile("serverData/function_prediction/elapsed_times.txt");
    std::vector<float> elapsed_times;
    float time;
    while (infile >> time) {
        elapsed_times.push_back(time);
    }
    infile.close();
    TH1F* hPythonTimes = new TH1F("hPythonTimes", "NN propagation time, python; t, ms", 100, 0, 100);
    for (float elapsed_time : elapsed_times) {
        hPythonTimes->Fill(elapsed_time*1000);
    }
    hPythonTimes->SetLineWidth(2);
    hPythonTimes->SetLineColor(1);

    h1->SetLineWidth(2);
    h2->SetLineWidth(2);
    h3->SetLineWidth(2);

    //h1->SetStats(false);
    //h2->SetStats(false);
    //h3->SetStats(false);

    TCanvas *canvas = new TCanvas("canvas", "Histograms on Pads", 800, 600);
    canvas->Divide(2, 1);
    canvas->cd(1);
    h1->Draw();
    //canvas->cd(2);
    //h3->Draw();
    canvas->cd(2);
    hPythonTimes->Draw();
    canvas->Update();
}

void mergePredictions(){
    const std::string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";
    ofstream ofstr((saveLocation+ "function_prediction/predicted_all.txt").c_str());
    for (int i = 0; i < 1; i++){
        ifstream fin1;
        fin1.open((saveLocation+ "function_prediction/predicted_" + to_string(i) + ".txt").c_str());
        while (!fin1.eof()){
            double run;
            double prediction;
            vector<double> predictionNodes;
            fin1 >> run;
            ofstr << (int)run;
            for (size_t i = 0; i < 24; i ++){
                fin1 >> prediction;
                predictionNodes.push_back(prediction);
                //cout << prediction << endl;
                ofstr << "  " << prediction;
            }
            ofstr << endl;
        }
        fin1.close();
    }
    ofstr.close();

    ofstr.open((saveLocation+ "function_prediction/predicted1_all.txt").c_str());
    for (int i = 0; i < 100; i++){
        ifstream fin1;
        fin1.open((saveLocation+ "function_prediction/predicted1_" + to_string(i) + ".txt").c_str());
        while (!fin1.eof()){
            double run;
            double prediction;
            vector<double> predictionNodes;
            fin1 >> run;
            ofstr << (int)run;
            for (size_t i = 0; i < 24; i ++){
                fin1 >> prediction;
                predictionNodes.push_back(prediction);
                //cout << prediction << endl;
                ofstr << "  " << prediction;
            }
            ofstr << endl;
        }
        fin1.close();
    }
    ofstr.close();
}

void drawTargetPredictionComparison(){
    const std::string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";

    /// Reading mean and std
    ifstream fin1;
    vector<double> meanValues, stdValues;
	fin1.open((saveLocation+"function_prediction/meanValues.txt").c_str());
    double meanV = 0;
    while (fin1 >> meanV){
        meanValues.push_back(meanV);
    }
    fin1.close();
    fin1.open((saveLocation+"function_prediction/stdValues.txt").c_str());
    double stdV = 0;
    while (fin1 >> stdV){
        stdValues.push_back(stdV);
    }
    fin1.close();

    /// Reading target values and averaging -> map
    ifstream fin2;
    TH1F* hdTarget1D = new TH1F("hdTarget1D","target1-target2;#errors;counts",1000,-100,100);
	fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCSecModPrecise.dat").c_str());
    std::map<int, vector< pair<double,double> > > targets;
    vector<int> runsTargets;
    vector<double> valuesTargets,runsTargetsErr,valuesTargetsErr;
    vector< pair<double, double> >  predTargetNodes;
    while (!fin2.eof()){
        int run, sector, mod;
        double target, targetErr;
        vector< pair<double, double> > targetNodes;
        double meanTarget, meanTargetErr;
        for (size_t i = 0; i < 24; i++){
            fin2 >> run >> sector >> mod >> target >> targetErr;
            if (predTargetNodes.size()>0){
                if (fabs(target - predTargetNodes[i].first) < predTargetNodes[i].second*5)
                    targetErr = fabs(target - predTargetNodes[i].first)/2.;
                //else
                //    targetErr *= 2;
                if (targetErr == 0){
                    targetErr = predTargetNodes[i].second;
                }
                hdTarget1D->Fill((target - predTargetNodes[i].first)/targetErr );
            }
            meanTarget+=(target-meanValues[meanValues.size()-24*2+i])/stdValues[meanValues.size()-24*2+i];
            meanTargetErr += pow(targetErr/stdValues[meanValues.size()-24*2+i],2);
            if (i == 0){
                runsTargets.push_back(run);
                //valuesTargets.push_back((target-meanValues[meanValues.size()-24*2])/stdValues[meanValues.size()-24*2]);
                valuesTargets.push_back(target);
                runsTargetsErr.push_back(0);
                //valuesTargetsErr.push_back(targetErr/stdValues[meanValues.size()-24*2]);
                valuesTargetsErr.push_back(targetErr);
            }
            //cout << run << endl;
            targetNodes.push_back(make_pair(target,targetErr));
        }
        //meanTarget = meanTarget/24.;
        //meanTargetErr = sqrt(meanTargetErr)/24.;
        //runsTargets.push_back(run);
        //valuesTargets.push_back(meanTarget);
        //runsTargetsErr.push_back(0);
        //valuesTargetsErr.push_back(meanTargetErr);

        targets[run] = targetNodes;
        predTargetNodes = targetNodes;
    }
    fin2.close();

    /// Reading MDC epics data
    fin2.open((saveLocation + "info_tables/MDCModSecPrecise.dat").c_str());
    std::map<int, vector< double > > mdcChanChamb0;
    vector<double> runsChans;
    vector<double> valuesChans[7];
    while (!fin2.eof()){
        int run;
        fin2 >> run;
        vector<double> valuesChan1;
        for (size_t i = 0; i < 7; i++){
            double valueBuffChan;
            double valueChan;
            //cout << valueChan << endl;
            for (size_t j = 0; j < 24; j++){
                fin2 >> valueBuffChan;
                if (j == 0){
                    if (i == 0)
                        runsChans.push_back((double)(runToDateNumber(run)));
                    if (i == 0)
                        valuesChans[i].push_back(-(valueBuffChan-meanValues[i*24])/stdValues[i*24]);
                    else{
                        valuesChans[i].push_back((valueBuffChan-meanValues[i*24])/stdValues[i*24]); }
                    //valuesChans.push_back(valueBuffChan);
                    valuesChan1.push_back(valueBuffChan);
                }
            }
        }
        mdcChanChamb0[run] = valuesChan1;
    }
    fin2.close();
    TGraph* grMDCchan[7];
    for (size_t i = 0; i < 7; i++){
        grMDCchan[i] = new TGraph(runsChans.size(),&runsChans[0],&valuesChans[i][0]);
        grMDCchan[i]->SetMarkerStyle(22);    grMDCchan[i]->SetMarkerSize(1.0);    grMDCchan[i]->SetLineColor(2);    grMDCchan[i]->SetMarkerColor(2);    grMDCchan[i]->SetLineWidth(1);
        if (i == 2){
            grMDCchan[i]->SetMarkerStyle(22);    grMDCchan[i]->SetMarkerSize(1.0);    grMDCchan[i]->SetLineColor(3);    grMDCchan[i]->SetMarkerColor(3);    grMDCchan[i]->SetLineWidth(1);}
        grMDCchan[i]->GetXaxis()->SetTimeDisplay(1); grMDCchan[i]->GetXaxis()->SetTimeFormat("%d/%m");
    }



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

    fin1.open((saveLocation+ "function_prediction/predicted.txt").c_str());
    std::map<int, vector< double > > predictions;
    while (!fin1.eof()){
        double run;
        double prediction;
        vector<double> predictionNodes;
        fin1 >> run;
        for (size_t i = 0; i < 24; i ++){
            fin1 >> prediction;
            predictionNodes.push_back(prediction);
            //cout << prediction << endl;
        }
        predictions[(int)run] = predictionNodes;
    }
    fin1.close();

    fin2.open((saveLocation+ "function_prediction/predicted1.txt").c_str());
    std::map<int, vector< double > > predictionsTest;
    while (!fin2.eof()){
        double run;
        double prediction;
        vector<double> predictionNodes;
        fin2 >> run;
        for (size_t i = 0; i < 24; i ++){
            fin2 >> prediction;
            predictionNodes.push_back(prediction);
        }
        predictionsTest[(int)run] = predictionNodes;
    }
    fin2.close();

    cout << "lasjfdksf;" << endl;

    /// _____ making graphErrors for prediction-target
    vector<double> runsPredictedTarget, diffPredictedTarget, runsPredictedTargetErr, diffPredictedTargetErr;
    TH2F* hpredictedTarget = new TH2F("hpredictedTarget","(predicted-target)/targetErr;run;(P-T)/T_{err} [#sigma]",1000,runToDateNumber(443654673),runToDateNumber(446359647), 100, -10, 10);
    TProfile* ppredictedTarget = new TProfile("ppredictedTarget","(predicted-target)/targetErr average;run;(P-T)/T_{err} [#sigma]",1000,runToDateNumber(443654673),runToDateNumber(446359647));
    TH1F* hpredictedTarget1D = new TH1F("hpredictedTarget1D","predicted-target;#std;counts",100,-10,10);
    for (auto x: predictions){
        if (targets.find(x.first) == targets.end())
            continue;
        for (size_t i = 0; i < 1; i ++){
            runsPredictedTarget.push_back((double)(runToDateNumber(x.first)));
            runsPredictedTargetErr.push_back(0);
            //diffPredictedTarget.push_back( (x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            //diffPredictedTarget.push_back( (x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24*2+i] );
            diffPredictedTarget.push_back( x.second[i] );
            diffPredictedTargetErr.push_back(0);
            hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            if (fabs((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second ) < 10)
                ppredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            //hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24*2+i] );
            hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
        }
    }
    vector<double> runsPredictedTargetP, diffPredictedTargetP, runsPredictedTargetErrP, diffPredictedTargetErrP;
    for (auto x: predictionsTest){
        if (targets.find(x.first) == targets.end())
            continue;
        for (size_t i = 0; i < 1; i ++){
            runsPredictedTargetP.push_back((double)(runToDateNumber(x.first)));
            runsPredictedTargetErrP.push_back(0);
            //diffPredictedTargetP.push_back( (x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            //diffPredictedTargetP.push_back( (x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24*2+i] );
            diffPredictedTargetP.push_back( x.second[i] );
            diffPredictedTargetErrP.push_back(0);
            hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            if (fabs((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second ) < 10)
                ppredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            //hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24*2+i] );
        }
    }

    int nbinsx = hpredictedTarget->GetNbinsX();
    int nbinsy = hpredictedTarget->GetNbinsY();
    TGraphErrors* mean_graph = new TGraphErrors(nbinsx);
    vector<double> x,y,x1,y1;
    // Calculate mean and standard deviation for each x bin
    for (int i = 1; i <= nbinsx; ++i) {
        double mean = 0.0;
        double std_dev = 0.0;
        int count = 0;
        // Calculate mean and standard deviation for the y values in the bin
        for (int j = 1; j <= nbinsy; ++j) {
            double bin_content = hpredictedTarget->GetBinContent(i, j);
            double ypos = hpredictedTarget->GetYaxis()->GetBinCenter(j);
            mean += ypos * bin_content;
            std_dev += ypos * ypos * bin_content;
            count += bin_content;
        }
        if (count > 0) {
            mean /= count;
            std_dev = std::sqrt(std_dev / count - mean * mean);
            // Fill the TGraphErrors with mean and standard deviation values
            mean_graph->SetPoint(i - 1, hpredictedTarget->GetXaxis()->GetBinCenter(i), mean);
            mean_graph->SetPointError(i - 1, 0, 0);

            if (mean+std_dev!=0){
                x.push_back(hpredictedTarget->GetXaxis()->GetBinCenter(i));
                x1.push_back(hpredictedTarget->GetXaxis()->GetBinCenter(i));
                y.push_back(mean+std_dev);
                y1.push_back(mean-std_dev);
                //std_dev_graph->SetPoint(i - 1, hpredictedTarget->GetXaxis()->GetBinCenter(i), mean+std_dev);
                //std_dev_graph->SetPointError(i - 1, 0, 0);
                //std_dev_graph1->SetPoint(i - 1, hpredictedTarget->GetXaxis()->GetBinCenter(i), mean-std_dev);
                //std_dev_graph1->SetPointError(i - 1, 0, 0);
            }
        }
    }
    TGraph* std_dev_graph = new TGraph(x.size(),&x[0],&y[0]);
    TGraph* std_dev_graph1 = new TGraph(x1.size(),&x1[0],&y1[0]);
    mean_graph->SetMarkerStyle(22);   mean_graph->SetMarkerSize(1.0);   mean_graph->SetLineColor(1);   mean_graph->SetMarkerColor(1);   mean_graph->SetLineWidth(4);


    cout << runsPredictedTarget.size() << endl;

    TGraphErrors* grPT = new TGraphErrors(runsPredictedTarget.size(),&runsPredictedTarget[0], &diffPredictedTarget[0], &runsPredictedTargetErr[0], &diffPredictedTargetErr[0]);
    TGraphErrors* grPTP = new TGraphErrors(runsPredictedTargetP.size(),&runsPredictedTargetP[0], &diffPredictedTargetP[0], &runsPredictedTargetErrP[0], &diffPredictedTargetErrP[0]);
    grPT->SetMarkerStyle(22);    grPT->SetMarkerSize(1.0);    grPT->SetLineColor(3);    grPT->SetMarkerColor(3);    grPT->SetLineWidth(4);
    grPTP->SetMarkerStyle(22);   grPTP->SetMarkerSize(1.0);   grPTP->SetLineColor(1);   grPTP->SetMarkerColor(1);   grPTP->SetLineWidth(4);
    


    TDatime da(2022,2,3,15,58,00);
    gStyle->SetTimeOffset(da.Convert());

    grPT->GetXaxis()->SetLimits(grPT->GetX()[0]-100000, grPT->GetX()[runsPredictedTarget.size()-1]*1.3);
    //grPT->GetYaxis()->SetRangeUser(-6,6);
    gr2->GetXaxis()->SetTimeDisplay(1); gr2->GetXaxis()->SetTimeFormat("%d/%m");
    grPT->GetXaxis()->SetTimeDisplay(1); grPT->GetXaxis()->SetTimeFormat("%d/%m");
    grPTP->GetXaxis()->SetTimeDisplay(1); grPTP->GetXaxis()->SetTimeFormat("%d/%m");

    hpredictedTarget->GetXaxis()->SetTimeDisplay(1); hpredictedTarget->GetXaxis()->SetTimeFormat("%d/%m");
    ppredictedTarget->GetXaxis()->SetTimeDisplay(1); ppredictedTarget->GetXaxis()->SetTimeFormat("%d/%m");
    mean_graph->GetXaxis()->SetTimeDisplay(1); mean_graph->GetXaxis()->SetTimeFormat("%d/%m");
    std_dev_graph->GetXaxis()->SetTimeDisplay(1); std_dev_graph->GetXaxis()->SetTimeFormat("%d/%m");

    TCanvas* canvas = new TCanvas("canvas", "Prediction vs Target calibration comparison", 1920, 950);
    TPad* pad = new TPad("pad", "Pad", 0.01, 0.01, 0.99, 0.99);
    pad->Draw();
    pad->cd();
    pad->SetGridy();

    gr2->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    //gr1->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    //grP->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    grPT->SetTitle("Prediction - Target;date;#sigma");
    grPTP->SetTitle("Prediction - Target;date;#sigma");

    TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    //legend->AddEntry(gr2, "target calibration", "l");
    //legend->AddEntry(gr1, "predicted, train dataset", "l");
    //legend->AddEntry(grP, "predicted, test dataset", "l");
    legend->AddEntry(grPT, "train dataset", "l");
    legend->AddEntry(grPTP, "test dataset", "l");

    //hdTarget1D->SetLineColor(2);
    //hdTarget1D->Draw();
    //hpredictedTarget1D->DrawNormalized("same",hdTarget1D->GetEntries());
    //hpredictedTarget1D->Draw();
    //gr2->GetYaxis()->SetRangeUser(-5,5);
    gr2->Draw("AP");
    //grMDCchan[0]->Draw("Psame");
    //grMDCchan[2]->Draw("Psame");
    //gr1->Draw("Psame");
    //grP->Draw("Psame");
    //gr1L->Draw("Psame");
    //grPL->Draw("Psame");
    grPT->Draw("Psame");
    grPTP->Draw("Psame");
    //hpredictedTarget->SetDrawOption("colz");
    //hpredictedTarget->Draw();
    //mean_graph->Draw("Psame");
    //std_dev_graph->GetYaxis()->SetRangeUser(-10,10);
    //std_dev_graph->Draw("APL");
    //std_dev_graph1->Draw("PLsame");
    //ppredictedTarget->Draw("same");
    //legend->Draw();

    TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);
    c->Update();
}
