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
#include <TRandom3.h>
#include <TPaveText.h>
#include <TLatex.h>

//#include "include/functions.h"
#include "source/functions.cc"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>

using namespace std;
using namespace dateRunF;

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

/*double dateToNumber(string date){
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
*/

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

void drawTargetSectorComparison(){
    cout << "drawing target sector divided..." << endl;
    const std::string saveLocation1 = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";
    vector< pair< int, vector<double> > > tableClbAll = preTrainFunctions::readClbTableFromFile(saveLocation1+"info_tables/run-mean_dEdxMDCSecModPreciseFit1.dat");     
    cout << " a  " << endl;
    const size_t split = 4;
    vector< vector<double> > arr[split];
    for (size_t s = 0; s < split; s++){
        arr[s].resize(4,vector<double>(0));
        for (size_t i = 0; i < tableClbAll.size(); i++){
            if (tableClbAll[i].second[0] != s || tableClbAll[i].second[1] != 2)
                continue;
            arr[s][0].push_back((tableClbAll[i].first));
            arr[s][1].push_back(tableClbAll[i].second[2]);
            arr[s][2].push_back(0);
            arr[s][3].push_back(tableClbAll[i].second[3]);
        }
    }

    TGraphErrors* grClbAll[split];
    double mean0[split];
    for (size_t i = 0; i < split; i++){
        arr[i][0] = dateRunF::timeVectToDateNumbers(arr[i][0]);
        arr[i][1] = runningMeanVector(arr[i][1],20);
        //if (i == 0){
            mean0[i] = 0;
            for (size_t j = 0; j < arr[i][0].size(); j++){
                mean0[i]+=arr[i][1][j];
            }
            mean0[i] = mean0[i]/arr[i][0].size();
        //}

        if (i>10){
            for (size_t j = 0; j < arr[i][0].size(); j++){
                arr[i][1][j] = arr[i][1][j]/arr[0][1][j];
            }
        }
    }
    for (size_t i = 0; i < split; i++){
        //arr[i][1] = normalizeVectorNN(arr[i][1]);
        for (size_t j = 0; j < arr[i][0].size(); j++){
            arr[i][1][j] = arr[i][1][j]/mean0[i];
        }
    }

    for (size_t i = 0; i < split; i++){
        grClbAll[i]  = new TGraphErrors( arr[i][0].size(), &arr[i][0][0], &arr[i][1][0], &arr[i][2][0], &arr[i][2][0]);
        grClbAll[i]->SetMarkerStyle(22);
        grClbAll[i]->SetMarkerSize(0.5);
        grClbAll[i]->SetMarkerColor(1);
        grClbAll[i]->GetYaxis()->SetRangeUser(0.85,1.15);
        //if (i == 0)
        //    grClbAll[i]->GetYaxis()->SetRangeUser(-2,2);
        grClbAll[i]->GetXaxis()->SetTimeDisplay(1);
        grClbAll[i]->GetXaxis()->SetTimeFormat("%d/%m");
        grClbAll[i]->SetTitle(Form("plane%zu;time;fluctuations %zu",i+1,i+1));
        grClbAll[i]->GetXaxis()->SetLabelSize(0.05);
        grClbAll[i]->GetYaxis()->SetLabelSize(0.05);
        grClbAll[i]->GetXaxis()->SetTitleSize(0.08);
        grClbAll[i]->GetYaxis()->SetTitleSize(0.08);
        grClbAll[i]->GetXaxis()->SetTitleOffset(0.65);
        grClbAll[i]->GetYaxis()->SetTitleOffset(0.5);
        if (i == 0)
            grClbAll[i]->SetTitle(Form("plane%zu;time;relative change",i+1));
    }

    gStyle->SetTitleFontSize(0.1); // Adjust the value as needed
    gStyle->SetLabelSize(0.45, "xy"); // Adjust the value as needed

    TCanvas* canvas = new TCanvas("canvas", "Canvas Title", 1920, 950);
    canvas->Divide(2, 2); // Divide canvas into 3 rows and 1 column
    for (size_t i = 0; i < split; i++){
        canvas->cd(i+1);
        TPad* pad = new TPad(Form("pad%zu", i + 1), Form("Pad %zu", i + 1), 0.0, 0.0, 1.0, 1.0);
        pad->SetBottomMargin(0.15);
        pad->Draw();
        pad->cd();
        pad->SetGridy();
        grClbAll[i]->Draw("AP");
        canvas->Update();
        cin.get();
    }
    //delete canvas;
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
    int nodes_length = 24;
    //const std::string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";
    ofstream ofstr((saveLocation+ "function_prediction/predicted/predicted_all.txt").c_str());
    for (int i = 0; i < 1; i++){
        ifstream fin1;
        fin1.open((saveLocation+ "function_prediction/predicted/predicted_" + to_string(i) + ".txt").c_str());
        int runPrevT = 0;
        bool write = true;
        while (!fin1.eof()){
            double run;
            double prediction;
            vector<double> predictionNodes;
            fin1 >> run;
            if (run == runPrevT)
                write = false;
            if (write)
                ofstr << (int)run;
            for (size_t i = 0; i < nodes_length; i ++){
                fin1 >> prediction;
                predictionNodes.push_back(prediction);
                //cout << prediction << endl;
                if (write)
                    ofstr << "  " << prediction;
            }
            if (write)
                ofstr << endl;
            runPrevT = run;
        }
        fin1.close();
    }
    ofstr.close();

    ofstr.open((saveLocation+ "function_prediction/predicted/predicted1_all.txt").c_str());
    for (int i = 0; i < 1; i++){
        ifstream fin1;
        fin1.open((saveLocation+ "function_prediction/predicted/predicted1_" + to_string(i) + ".txt").c_str());
        int runPrevT = 0;
        bool write = true;
        while (!fin1.eof()){
            double run;
            double prediction;
            vector<double> predictionNodes;
            fin1 >> run;
            if (run == runPrevT)
                write = false;
            if (write)
                ofstr << (int)run;
            for (size_t i = 0; i < nodes_length; i ++){
                fin1 >> prediction;
                predictionNodes.push_back(prediction);
                //cout << prediction << endl;
                if (write)
                    ofstr << "  " << prediction;
            }
            if (write)
                ofstr << endl;
            runPrevT = run;
        }
        fin1.close();
    }
    ofstr.close();
}


void drawTargetPredictionComparisonWithAsubplot(TGraphErrors* gr2, TGraphErrors* grPT, TGraphErrors* grPTP, TH2F* hpredictedTarget){
    //h1->SetTitle("After efficiency correction;;d#sigma/dM_{KK} [normalized]");

    gStyle->SetOptStat(0);

    // Create a canvas
    TCanvas *c = new TCanvas("c", "Histogram and Ratio", 1800, 800);

    // Create pads for histograms and ratio
    TPad *pad1 = new TPad("pad1", "Main plot", 0.0, 0.3, 1.0, 1.0);
    TPad *pad2 = new TPad("pad2", "Ratio plot", 0.0, 0.0, 1.0, 0.3);

    pad1->SetBottomMargin(0); // Remove bottom margin for the upper pad
    pad2->SetTopMargin(0);    // Remove top margin for the lower pad
    pad2->SetBottomMargin(0.3);

    pad1->Draw();
    pad2->Draw();

    // Draw histograms on the upper pad
    pad1->cd();
    gr2->Draw("AP");
    grPT->Draw("Psame");
    grPTP->Draw("Psame");

    TLegend *legend = new TLegend(0.6, 0.7, 0.9, 0.9);
    legend->AddEntry(gr2, "Target with errors", "lpe");
    legend->AddEntry(grPT, "Predicted, train dataset", "lpe");
    legend->AddEntry(grPTP, "Predicted, test dataset", "lpe");
    legend->Draw();

    // Calculate ratio and draw on the lower pad
    pad2->cd();
    pad2->SetGridy(1);
    hpredictedTarget->SetTitle(";date [day/month]"); // Remove title for the ratio plot
    hpredictedTarget->SetLineColor(kBlack);
    hpredictedTarget->GetYaxis()->SetTitle("Ratio");
    //h_ratio->GetYaxis()->SetRangeUser(0.5,1.5);
    hpredictedTarget->GetYaxis()->SetTitleSize(0.1);
    hpredictedTarget->GetYaxis()->SetTitleOffset(0.3);
    hpredictedTarget->GetYaxis()->SetLabelSize(0.08);
    hpredictedTarget->GetXaxis()->SetTitleSize(0.1);
    hpredictedTarget->GetXaxis()->SetLabelSize(0.08);
    hpredictedTarget->Draw();

    TF1* f1 = new TF1("f1","[0]+[1]*x",950,1500);
    //h_ratio->Fit("f1","","",985,1500);

    // Update canvas
    c->Update();
}


void downsampleGraphInPlace(TGraphErrors* graph, double fraction) {
    if (!graph) return;

    int n = graph->GetN();
    TRandom3 randGen(0); // Random number generator with seed = 0

    for (int i = n - 1; i >= 0; i--) {  // Iterate backward to avoid shifting indices
        if (randGen.Rndm() > fraction) { // Remove (1 - fraction) of points
            graph->RemovePoint(i);
        }
    }
}


void drawTargetPressureOverpressure(TGraph* graph1, TGraph* graph2){
    TCanvas* c = new TCanvas("c", "Target Pressure and Overpressure", 1800, 800);
    c->SetGrid();

    graph1->GetYaxis()->SetTitle("Normalized fluctuations");
    graph1->GetXaxis()->SetTitle("Date [day/month]");
    graph1->GetXaxis()->SetTimeDisplay(1); graph1->GetXaxis()->SetTimeFormat("%d/%m");
    graph1->Draw("AP");

    graph2->Draw("Psame");

    TLegend* legend = new TLegend(0.6, 0.7, 0.9, 0.9);
    legend->AddEntry(graph1, "ToT - atm. pressure", "lpe");
    legend->AddEntry(graph2, "Overpressure", "lpe");
    legend->Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.04);
    latex.DrawLatex(0.15, 0.85, "Module 1, Sector 1");

    c->Update();

}

/// draw a graph with markers comparing totMp and overp, with latex for module-sector
void drawTargetPressureOverpressure1(vector<double> totMp, vector<double> overp){
    TGraph* graph1 = new TGraph(totMp.size(), &totMp[0], &overp[0]);
    graph1->SetMarkerStyle(20);
    graph1->SetMarkerSize(0.6);
    graph1->SetMarkerColor(4);
    graph1->SetLineColor(4);
    graph1->SetLineWidth(1);
    graph1->GetYaxis()->SetRangeUser(-2,2);


    TCanvas* c = new TCanvas("c", "Target Pressure and Overpressure", 1800, 800);
    c->SetGrid();
    graph1->GetYaxis()->SetTitle("ToT - atm. pressure normalized");
    graph1->GetXaxis()->SetTitle("Overpressure normalized");
    graph1->Draw("AP");

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.04);
    latex.DrawLatex(0.15, 0.55, "Module 1, Sector 1");

    c->Update();
}


void drawTargetPredictionComparison(){
    int nodes_length = 24;
    TFile* fileOut = new TFile("savedPlotsTemp1.root","RECREATE");

    setenv("TZ", "Europe/Berlin", 1);
    tzset();

    const char* timeFormat = "%d/%m %H:%M";

    //const std::string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";
    int chanTo = 5;
    /// Reading mean and std
    ifstream fin1;
    vector<double> meanValues, stdValues;
	fin1.open((saveLocation+"function_prediction/meanValuesT.txt").c_str());
    double meanV = 0;
    while (fin1 >> meanV){
        meanValues.push_back(meanV);
    }
    fin1.close();
    fin1.open((saveLocation+"function_prediction/stdValuesT.txt").c_str());
    double stdV = 0;
    while (fin1 >> stdV){
        stdValues.push_back(stdV);
    }
    fin1.close();

    TH2F* targetVsSource = new TH2F("targetVsSource","",100,-5,5,100,-5,5);

    /// Reading target values and averaging -> map
    ifstream fin2;
    TH1F* hdTarget1D = new TH1F("hdTarget1D","target1-target2;#errors;counts",1000,-100,100);
    //fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCSecModPrecise.dat").c_str());
	//fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCSecModPreciseFit2.dat").c_str());
    fin2.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/targets25beamTFull.dat");
    //fin2.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/targets24beamTFull.dat");
    //fin2.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOut.dat");
    std::map<int, vector< pair<double,double> > > targets;
    std::map<int, vector< pair<double,double> > > targetsNormalized;
    vector<int> runsTargets;
    vector<double> valuesTargets,valuesTargetsNormalized,runsTargetsErr,valuesTargetsErr;
    vector< pair<double, double> >  predTargetNodes;
    while (!fin2.eof()){
        int run, runEnd, sector, mod;
        double target, targetErr;
        vector< pair<double, double> > targetNodes;
        vector< pair<double, double> > targetNodesNormalized;
        double meanTarget, meanTargetErr;
        for (size_t i = 0; i < 24; i++){
            fin2 >> run >> runEnd >> sector >> mod >> target >> targetErr;
            //fin2 >> run >> sector >> mod >> target >> targetErr;
            if (predTargetNodes.size()>0){
                //if (fabs(target - predTargetNodes[i].first) < predTargetNodes[i].second*5)
                //    targetErr = fabs(target - predTargetNodes[i].first)/2.;
                //else
                //    targetErr *= 2;
                if (targetErr == 0){
                    targetErr = predTargetNodes[i].second;
                }
                hdTarget1D->Fill((target - predTargetNodes[i].first)/targetErr);
            }
            meanTarget+=(target-meanValues[meanValues.size()-nodes_length*2+i])/stdValues[meanValues.size()-nodes_length*2+i];
            meanTargetErr += pow(targetErr/stdValues[meanValues.size()-nodes_length*2+i],2);

            double targetNormalized = (target-meanValues[meanValues.size()-24*2+i])/stdValues[meanValues.size()-24*2+i];
            double targetErrNormalized = targetErr/stdValues[meanValues.size()-24*2+i];

            if (i == chanTo && run < 11445503850){
                runsTargets.push_back(run);
                valuesTargetsNormalized.push_back(targetNormalized);
                valuesTargets.push_back(target);
                runsTargetsErr.push_back(0);
                //valuesTargetsErr.push_back(targetErr/stdValues[meanValues.size()-nodes_length*2+i]);
                valuesTargetsErr.push_back(targetErr);
                if (target < 20){
                    cout << run << "    " << target << "    " << targetErr << endl;
                }
            }
            //cout << run << endl;
            //if (valuesTargets.size() >= 1){
                //targetNodes.push_back(make_pair(valuesTargets[valuesTargets.size()-1],valuesTargetsErr[valuesTargetsErr.size()-1]));
            targetNodes.push_back(make_pair(target,targetErr));
            targetNodesNormalized.push_back(make_pair(targetNormalized,targetErrNormalized));
                //targetNodes.push_back(make_pair(target/meanValues[meanValues.size()-24+i],targetErr/meanValues[meanValues.size()-24+i]));
                //targetNodesNormalized.push_back(make_pair(valuesTargetsNormalized[valuesTargetsNormalized.size()-1],valuesTargetsErr[valuesTargetsErr.size()-1]));
                //targetNodesNormalized.push_back(make_pair((target-meanValues[meanValues.size()-24+i])/stdValues[meanValues.size()-24+i],(targetErr)/stdValues[meanValues.size()-24+i]));
            //}
        }
        //meanTarget = meanTarget/24.;
        //meanTargetErr = sqrt(meanTargetErr)/24.;
        //runsTargets.push_back(run);
        //valuesTargets.push_back(meanTarget);
        //runsTargetsErr.push_back(0);
        //valuesTargetsErr.push_back(meanTargetErr);

        targets[run] = targetNodes;
        targetsNormalized[run] = targetNodesNormalized;
        predTargetNodes = targetNodes;
    }
    fin2.close();

    TGraphErrors* gr2 = new TGraphErrors();

    //valuesTargets = runningMeanVector(valuesTargets, 20);
    for (size_t i = 0; i < runsTargets.size(); i++){
        //targets[runsTargets[i]] = valuesTargets[i];
        int n = gr2->GetN();
        gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),targetsNormalized[runsTargets[i]][chanTo].first);
        //gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),targets[runsTargets[i]][chanTo].first);
        //gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),valuesTargets[i]);
        //gr2->SetPoint(n,(double)(runsTargets[i]),valuesTargets[i]);
        gr2->SetPointError(n, 0, targetsNormalized[runsTargets[i]][chanTo].second);
        //gr2->SetPointError(n, 0, targets[runsTargets[i]][chanTo].second);
    }
    //downsampleGraphInPlace(gr2,0.1);
    gr2->SetMarkerStyle(21);
    gr2->SetMarkerSize(0.6);
    gr2->SetMarkerColor(4);
    gr2->SetLineColor(4);
    gr2->SetLineWidth(1);
    cout << gr2->GetN() << endl;
    cout << "finished with target " << endl;


    /// Reading MDC epics data
    //fin2.open((saveLocation + "info_tables/MDCModSecPrecise.dat").c_str());
    //fin2.open((saveLocation + "info_tables/MDCModSecPreciseExtended.dat").c_str());
    //fin2.open((saveLocation + "info_tables/MDCModSecPreciseCosmic25VaryHV.dat").c_str());
    //fin2.open((saveLocation + "info_tables/MDCModSecPreciseFeb22ends9.dat").c_str());
    fin2.open((saveLocation + "info_tables/MDCModSecPreciseApr25ends9.dat").c_str());
    //fin2.open((saveLocation + "info_tables/MDCModSecPreciseFeb24ends9a.dat").c_str());
    std::map<int, vector< double > > mdcChanChamb0;
    cout << "a" << endl;
    const int channelNumber = 9;
    vector<double> runsChans;
    vector<double> valuesChans[channelNumber][24];
    vector<double> runsChans2;
    vector<double> valuesChans1;
    vector<double> valuesChans2;
    TH1F* hHvValues = new TH1F("hHvValues","",100,-10,10);
    string line;
    nodes_length = 24;
    
    while (!fin2.eof()){
        int run;
        fin2 >> run;
        //run += 2*3600;
        vector<double> valuesChan1;
        double fillHist = -100;
        for (size_t i = 0; i < channelNumber; i++){
            double valueBuffChan;
            double valueChan;
            //cout << valueChan << endl;
            for (size_t j = 0; j < 24; j++){
                fin2 >> valueBuffChan;
                //valuesChans[i][j].push_back((valueBuffChan-900)*0.5);
                //valuesChans[i][j].push_back(valueBuffChan-0);
                //valuesChans[i][j].push_back(2-valueBuffChan/meanValues[i*nodes_length+j]);
                //valuesChans[i][j].push_back(-(valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j]);
                //if (j == chanTo){
                    if (i == 0){
                        if (stdValues[i*nodes_length+j] != 0)
                            valuesChans[i][j].push_back(-0.5*(valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j]); 
                        else
                            valuesChans[i][j].push_back(valueBuffChan-meanValues[i*nodes_length+j]); 
                    }
                    else{
                        if (stdValues[i*nodes_length+j] != 0)
                            valuesChans[i][j].push_back((valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j]); 
                        else
                            valuesChans[i][j].push_back(valueBuffChan-meanValues[i*nodes_length+j]); 
                    }
                    //valuesChans.push_back(valueBuffChan);
                    valuesChan1.push_back(valueBuffChan);
                    hHvValues->Fill((valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j]);
                //}

                if (targetsNormalized.find(run) != targetsNormalized.end() && i == 0 && j == chanTo && run < 445503850){
                //if (i == 0 && j == chanTo){
                    //targetVsSource->Fill(valuesChans[i][valuesChans[i].size()-1], targetsNormalized[run][j].first);
                    runsChans2.push_back((double)(runToDateNumber(run)));
                    //valuesChans2.push_back((valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j] + targetsNormalized[run][j].first );
                    valuesChans2.push_back((2-valueBuffChan/meanValues[i*nodes_length+j]) - targets[run][j].first );
                    fillHist = (valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j] + targetsNormalized[run][j].first ;
                }
                else if (targetsNormalized.find(run) != targetsNormalized.end() && i == 3 && j == chanTo && fillHist != -100 && run < 445503850){
                    targetVsSource->Fill(-0.5*(valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j], fillHist );
                    //valuesChans1.push_back(-(valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j]);
                    valuesChans1.push_back(2-(valueBuffChan/meanValues[i*nodes_length+j]));
                }
            }
        }
        runsChans.push_back((double)(runToDateNumber(run)));
        mdcChanChamb0[run] = valuesChan1;
    }

    TDatime da(2022,2,3,15,58,00);
    //gStyle->SetTimeOffset(da.Convert());
    gr2->GetXaxis()->SetTimeOffset(0, "gmt");
    //gStyle->SetTimeOffset(0, "gmt"); 
    
    cout << "b" << endl;
    nodes_length = 12;
    fin2.close();
    TGraph* grMDCchan[channelNumber];
    TGraph* grMDCchan2;
    grMDCchan2 = new TGraph(runsChans2.size(),&runsChans2[0],&valuesChans2[0]);
    grMDCchan2->SetName("grMDCchan2");
    grMDCchan2->SetMarkerStyle(22);    grMDCchan2->SetMarkerSize(0.7);    grMDCchan2->SetLineColor(4);    grMDCchan2->SetMarkerColor(4);    grMDCchan2->SetLineWidth(3);
    grMDCchan2->GetXaxis()->SetTimeDisplay(1); grMDCchan2->GetXaxis()->SetTimeFormat(timeFormat);
    for (size_t i = 0; i < channelNumber; i++){
        grMDCchan[i] = new TGraph(runsChans.size(),&runsChans[0],&valuesChans[i][chanTo][0]);
        grMDCchan[i]->SetMarkerStyle(22);    grMDCchan[i]->SetMarkerSize(0.7);    grMDCchan[i]->SetLineColor(1);    grMDCchan[i]->SetMarkerColor(1);    grMDCchan[i]->SetLineWidth(3);
        if (i == 2){
            grMDCchan[i]->SetMarkerStyle(22);    grMDCchan[i]->SetMarkerSize(0.8);    grMDCchan[i]->SetLineColor(3);    grMDCchan[i]->SetMarkerColor(3);    grMDCchan[i]->SetLineWidth(1);}
        if (i == 3){
            grMDCchan[i]->SetLineColor(2);    grMDCchan[i]->SetMarkerColor(2);    grMDCchan[i]->SetLineWidth(1);}
        grMDCchan[i]->GetXaxis()->SetTimeDisplay(1); grMDCchan[i]->GetXaxis()->SetTimeFormat(timeFormat);
    }


    fin1.open((saveLocation+ "function_prediction/predicted/predicted_0.txt").c_str());
    std::map<int, vector< double > > predictions;
    double runPrevT = 0;
    while (!fin1.eof()){
        double run = 0;
        double prediction = 0;
        vector<double> predictionNodes;
        fin1 >> run;
        for (size_t i = 0; i < nodes_length; i ++){
            fin1 >> prediction;
            predictionNodes.push_back(prediction);
        }
        if (run != runPrevT)
            predictions[(int)run] = predictionNodes;
        runPrevT = run;
    }
    fin1.close();

    fin2.open((saveLocation+ "function_prediction/predicted/predicted1_0.txt").c_str());
    std::map<int, vector< double > > predictionsTest;
    runPrevT = 0;
    while (!fin2.eof()){
        double run = 0;
        double prediction = 0;
        vector<double> predictionNodes;
        fin2 >> run;
        for (size_t i = 0; i < nodes_length; i++){
            fin2 >> prediction;
            predictionNodes.push_back(prediction);
        }
        if (run != runPrevT)
            predictionsTest[(int)run] = predictionNodes;
        runPrevT = run;
    }
    fin2.close();

    cout << "lasjfdksf;" << endl;

    /// _____ making graphErrors for prediction-target
    vector<double> runsPredictedTarget, diffPredictedTarget, runsPredictedTargetErr, diffPredictedTargetErr;
    vector<double> diffPredictedTargetDiff;
    //TH2F* hpredictedTarget = new TH2F("hpredictedTarget","(predicted-target)/targetErr;run;(P-T)/T_{err} [#sigma]",10000,runToDateNumber(runsTargets[1])-100000,runToDateNumber(runsTargets[runsTargets.size()-1])+100000, 1000, 0.8, 1.2);
    TH2F* hpredictedTarget = new TH2F("hpredictedTarget","(predicted-target)/targetErr;run;(P-T)/T_{err} [#sigma]",10000,runToDateNumber(runsTargets[1])-100000,runToDateNumber(445559647), 1000, 0.8, 1.2);
    TProfile* ppredictedTarget = new TProfile("ppredictedTarget","(predicted-target)/targetErr average;run;(P-T)/T_{err} [#sigma]",1000,runToDateNumber(443654673),runToDateNumber(446359647));
    TH1F* hpredictedTarget1D = new TH1F("hpredictedTarget1D","predicted-target;#std;counts",100,-10,10);
    TH2F* h2dPredictionsVsTargetTrain = new TH2F("h2dPredictionsVsTargetTrain","Predictions Vs Target training; prediction; target",1000,0,100,1000,0,100);
    TH2F* h2dPredictionsVsTargetTest = new TH2F("h2dPredictionsVsTargetTest","Predictions Vs Target testing; prediction; target",1000,0,100,1000,0,100);
    for (auto x: predictions){
        for (size_t i = chanTo; i < chanTo+1; i ++){
            runsPredictedTarget.push_back((double)(runToDateNumber((int)(x.first))));
            runsPredictedTargetErr.push_back(0);
            diffPredictedTarget.push_back( x.second[i] );
            diffPredictedTargetErr.push_back(0);
        }
        if (targets.find(x.first) == targets.end())
            continue;
        for (size_t i = chanTo; i < chanTo+1; i ++){
            //diffPredictedTarget.push_back( (x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            diffPredictedTargetDiff.push_back( (x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24+i] );
            //diffPredictedTarget.push_back( (x.second[i] - meanValues[meanValues.size()-24+i])/stdValues[meanValues.size()-24+i] );
            h2dPredictionsVsTargetTrain->Fill(x.second[i],targets[x.first][i].first);
            //hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i]/targets[x.first][i].first));
            if (fabs((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second ) < 10)
                ppredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            //hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24*2+i] );
            //hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
        }
    }
    vector<double> runsPredictedTargetP, diffPredictedTargetP, runsPredictedTargetErrP, diffPredictedTargetErrP;
    vector<double> diffPredictedTargetPDiff;
    for (auto x: predictionsTest){
        for (size_t i = chanTo; i < chanTo+1; i ++){
            runsPredictedTargetP.push_back((double)(runToDateNumber((int)(x.first))));
            runsPredictedTargetErrP.push_back(0);
            diffPredictedTargetP.push_back( x.second[i] );
            diffPredictedTargetErrP.push_back(0);
        }
        if (targets.find(x.first) == targets.end())
            continue;
        for (size_t i = chanTo; i < chanTo+1; i ++){
        //for (size_t i = 0; i < 18; i ++){
            //diffPredictedTargetP.push_back( (x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            diffPredictedTargetPDiff.push_back( (x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24+i] );
            //diffPredictedTargetP.push_back( (x.second[i] - meanValues[meanValues.size()-24+i])/stdValues[meanValues.size()-24+i] );
            h2dPredictionsVsTargetTest->Fill(x.second[i],targets[x.first][i].first);
            //hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i]/targets[x.first][i].first));
            if (fabs((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second ) < 10)
                ppredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            //hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24*2+i] );
            hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
        }
    }

    cout << "xx" << endl;

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

    //downsampleGraphInPlace(grPT,0.1);
    //downsampleGraphInPlace(grPTP,0.1);

    TGraphErrors* grPTd = new TGraphErrors(runsPredictedTarget.size(),&diffPredictedTargetDiff[0], &diffPredictedTarget[0], &runsPredictedTargetErr[0], &diffPredictedTargetErr[0]);
    TGraphErrors* grPTPd = new TGraphErrors(runsPredictedTargetP.size(),&diffPredictedTargetPDiff[0], &diffPredictedTargetP[0], &runsPredictedTargetErrP[0], &diffPredictedTargetErrP[0]);
    grPT->SetMarkerStyle(26);    grPT->SetMarkerSize(1.0);    grPT->SetLineColor(3);    grPT->SetMarkerColor(3);    grPT->SetLineWidth(1);
    grPTP->SetMarkerStyle(24);   grPTP->SetMarkerSize(1.0);   grPTP->SetLineColor(2);   grPTP->SetMarkerColor(2);   grPTP->SetLineWidth(1);
    

    cout << "x1" << endl;

    //TDatime da(2022,2,3,15,58,00);
    //gStyle->SetTimeOffset(da.Convert());

    //grPT->GetXaxis()->SetLimits(grPT->GetX()[0]-100000, grPT->GetX()[runsPredictedTarget.size()-1]*1.3);
    //grPT->GetYaxis()->SetRangeUser(-6,6);
    gr2->GetXaxis()->SetTimeDisplay(1); gr2->GetXaxis()->SetTimeFormat(timeFormat);
    grPT->GetXaxis()->SetTimeDisplay(1); grPT->GetXaxis()->SetTimeFormat(timeFormat);
    grPTP->GetXaxis()->SetTimeDisplay(1); grPTP->GetXaxis()->SetTimeFormat(timeFormat);

    hpredictedTarget->GetXaxis()->SetTimeDisplay(1); hpredictedTarget->GetXaxis()->SetTimeFormat(timeFormat);
    ppredictedTarget->GetXaxis()->SetTimeDisplay(1); ppredictedTarget->GetXaxis()->SetTimeFormat(timeFormat);
    mean_graph->GetXaxis()->SetTimeDisplay(1); mean_graph->GetXaxis()->SetTimeFormat(timeFormat);
    std_dev_graph->GetXaxis()->SetTimeDisplay(1); std_dev_graph->GetXaxis()->SetTimeFormat(timeFormat);

    cout << "x2" << endl;

    TCanvas* canvas = new TCanvas("canvas", "Prediction vs Target calibration comparison", 1920, 950);
    TPad* pad = new TPad("pad", "Pad", 0.01, 0.01, 0.99, 0.99);
    pad->Draw();
    pad->cd();
    pad->SetGridy();

    //gr2->SetTitle("Prediction vs Target calibration comparison;Date [day/month];Average ToT [a.u.]");
    gr2->SetTitle("Prediction vs Target calibration comparison;Date [day/month];Fluctuations / average [%]");
    //gr1->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    //grP->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    grPT->SetTitle("Prediction - Target;date;#sigma");
    grPTP->SetTitle("Prediction - Target;date;#sigma");

    TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(gr2, "Average ToT", "lpe");
    //legend->AddEntry(gr1, "predicted, train dataset", "l");
    //legend->AddEntry(grP, "predicted, test dataset", "l");
    //legend->AddEntry(grPT, "Predicted, train dataset", "lpe");
    //legend->AddEntry(grPTP, "Predicted, test dataset", "lpe");

    //legend->AddEntry(grMDCchan2, "ToT - atm.pressure", "lpe");
    legend->AddEntry(grMDCchan[0], "Atm. pressure", "lpe");
    //legend->AddEntry(grMDCchan[3], "Overpressure", "lpe");





    cout << "x3" << endl;

    //hdTarget1D->SetLineColor(2);
    //hdTarget1D->Draw();
    //hpredictedTarget1D->DrawNormalized("same",hdTarget1D->GetEntries());
    //hpredictedTarget1D->Draw();
    //gr2->GetYaxis()->SetRangeUser(0,100);
    gr2->Draw("AP");
    //grPT->Draw("Psame");
    //grPTP->Draw("Psame");


    //gr2->GetXaxis()->SetLimits(runToDateNumber(runsTargets[1])-100000, runToDateNumber(runsTargets[runsTargets.size()-1])+100000);
    //gr2->GetXaxis()->SetLimits(runToDateNumber(runsTargets[1])-100000, runToDateNumber(445559647));
    //hpredictedTarget->GetXaxis()->SetLimits(gr2->GetX()[0]-100000, gr2->GetX()[runsPredictedTarget.size()-1]*1.3);
    
    //drawTargetPredictionComparisonWithAsubplot(gr2, grPT, grPTP, hpredictedTarget);   //!!!!!!!!!!!!!!!!!!!!!!!!1

    //drawTargetPressureOverpressure(grMDCchan2, grMDCchan[3]);
    //drawTargetPressureOverpressure1(valuesChans1, valuesChans2);

    //TCanvas* canvas2 = new TCanvas("canvas2", "Prediction vs Target calibration comparison", 1920, 950);
    //canvas2->cd();
    //grMDCchan[3]->SetMarkerColor(5);
    //grMDCchan2->Draw("AP");
    grMDCchan[0]->Draw("Psame");
    //grMDCchan[3]->Draw("AP");
    grMDCchan[2]->Draw("Psame");
    grMDCchan[3]->Draw("Psame");
    //grMDCchan[1]->Draw("Psame");
    //grMDCchan[1]->Draw("Psame");
    //grMDCchan[6]->Draw("Psame");
    //grMDCchan[3]->Draw("Psame");

    //h2dPredictionsVsTargetTrain->Draw("colz");
    //h2dPredictionsVsTargetTest->Draw("colz");
    
    //grMDCchan[6]->Draw("AP");
    //targetVsSource->Draw();
    //hHvValues->Draw();
    //gr1->Draw("Psame");
    //grP->Draw("Psame");
    //gr1L->Draw("Psame");
    //grPL->Draw("Psame");
    //grPT->Draw("AP");
    //hpredictedTarget->SetDrawOption("colz");
    //hpredictedTarget->Draw("colz");
    //mean_graph->Draw("Psame");
    //std_dev_graph->GetYaxis()->SetRangeUser(-10,10);
    //std_dev_graph->Draw("APL");
    //std_dev_graph1->Draw("PLsame");
    //ppredictedTarget->Draw("same");
    //legend->Draw();


    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.04);
    int moduleLatex = chanTo/6;
    int sectorLatex = chanTo%6;
    latex.DrawLatex(0.15, 0.85, Form("Module %d, Sector %d",moduleLatex,sectorLatex));
    
    //TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);
    canvas->Update();

    fileOut->cd();
    hpredictedTarget1D->Write();
    //fileOut->Close();
}



vector<TGraph*> compareHVSensitivities() {
    int nChambers = 12;
    std::string pred_path = "serverData/function_prediction/predicted/";
    std::vector<int> hv_values;
    for (int i = 1650; i < 1790; i+=10){
        hv_values.push_back(i);
    }
    
    //TCanvas *canvas = new TCanvas("canvas", "HV Sensitivity Comparison", 1200, 800);

    vector<TGraph*> graphs;

    // BEAM slopes
    for (int ch = 0; ch < nChambers; ++ch) {
        std::vector<double> hvs, tots;
        for (int hv : hv_values) {
            std::ifstream fin(pred_path + "predicted_" + std::to_string(hv) + ".txt");
            double tot_sum = 0.0;
            int count = 0;
            double run_id;
            while (fin >> run_id) {
                double val;
                for (int i = 0; i < 12; ++i) {
                    fin >> val;
                    if (i == ch) {
                        tot_sum += val;
                    }
                }
                count++;
            }
            if (count > 0) {
                hvs.push_back(hv);
                tots.push_back(tot_sum / count);
            }
        }

        if (hvs.size() < 2) continue; // Not enough data to fit
        TGraph* gr = new TGraph(hvs.size(), &hvs[0], &tots[0]);
        gr->SetName(Form("graphExp%d",ch));
        gr->SetMarkerSize(1.5);
        gr->SetMarkerStyle(20);
        //gr->Draw("AP");
        gr->Fit("pol1", "Q");
        TF1* fit = gr->GetFunction("pol1");
        double slope = fit->GetParameter(1);

        graphs.push_back(gr);

        //canvas->Update();
        //cin.get();
    }
    return graphs;
}



template <typename T>
typename std::map<int, T>::const_iterator find_closest(const std::map<int, T>& m, int key) {
    if (m.empty()) return m.end();

    auto lower = m.upper_bound(key);

    if (lower == m.begin()) {
        return lower;
    }
    if (lower == m.end()) {
        return std::prev(lower);
    }

    auto prev = std::prev(lower);
    if (std::abs(prev->first - key) <= std::abs(lower->first - key)) {
        return prev;
    } else {
        return lower;
    }
}

template <typename T>
typename std::map<int, T>::const_iterator find_next(const std::map<int, T>& m, int key) {
    if (m.empty()) return m.end();

    auto lower = m.upper_bound(key);

    if (lower == m.end()) {
        return std::prev(lower);
    }

    return lower;
}
template <typename T>
int find_next_run(const std::map<int, T>& m, int key) {
    if (m.empty()) return 0;

    auto lower = m.upper_bound(key);

    if (lower == m.end()) {
        return std::prev(lower)->first + 600;
    }

    return lower->first-150;
}


double twoLinesFit(double *x, double *par) {
    return x[0] < par[4] ? x[0]*par[0] + par[1] : x[0]*par[2] + par[3];
}


void drawTvsHV(std::map<int, vector< pair<double,double> > > targets, 
               std::map<int,int> targetsRunEnd, 
               std::map<int, vector< double > > mdcChanChamb0,
               std::map<int, vector< double > > predictions,
               std::map<int, vector< double > > predictionsTest){
    TFile* fOut = new TFile("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/fOutFits1new100.root","RECREATE");
    fOut->cd();

    TCanvas* c2 = new TCanvas("c2", "Prediction vs Target calibration comparison", 1820, 1000);
    TCanvas* canvas = new TCanvas("canvas", "Prediction vs Target calibration comparison", 1820, 1000);
    c2->cd();

    vector<double> slopesCosmic;
    vector<double> slopesBeam;

    vector<TGraph*> expGraphsHV = compareHVSensitivities();

    for (size_t chamberI = 0; chamberI < 12; chamberI++){
        ///// Target vs HV + pressure dependencies

        vector<double> hvPoints, hvPointsErrors, targetPoints, targetPointsErrors, pressurePoints, pressurePointsErrors;
        vector<double> phvPoints, phvPointsErrors, phvPointsP, phvPointsErrorsP, predictedPoints, predictedPointsErrors, predictedPointsP, predictedPointsErrorsP;
        for (const auto& n: targets){
            //if (n.second[chamberI].first == 0)
            //    continue;
            
            //int nextTargetRun = find_next_run(targets,n.first);
            int nextTargetRun = targetsRunEnd[n.first];
            //auto closest = find_closest(mdcChanChamb0, n.first);
            double averageHV = 0; double averageP = 0; int counterIn = 0;
            for (const auto& m: mdcChanChamb0 ){
                if (m.first+30 >= n.first && m.first+30 < nextTargetRun-30){
                    //cout << n.first << "    <   " << m.first << "   <   " << nextTargetRun << endl;
                    averageHV += m.second[1*24+chamberI];
                    averageP += m.second[0*24+chamberI];
                    counterIn += 1;
                }
            }
            if (counterIn != 0){
                averageHV = averageHV/(double)(counterIn);
                averageP = averageP/(double)(counterIn);
            }
            else{
                averageHV = find_closest(mdcChanChamb0, n.first)->second[1*24+chamberI];
                averageP = find_closest(mdcChanChamb0, n.first)->second[0*24+chamberI];
            }
            if (averageHV <= 1500 || nextTargetRun-n.first < 250)
                continue;
            //targetPoints.push_back(1./2.*TMath::Erfc(n.second[chamberI].first/2.));
            targetPoints.push_back(n.second[chamberI].first);
            //targetPoints.push_back(n.second[chamberI].second);
            targetPointsErrors.push_back(n.second[chamberI].second);
            //targetPointsErrors.push_back(0);

            if (predictions.find(n.first) != predictions.end()){
                predictedPoints.push_back(predictions[n.first][chamberI]);
                predictedPointsErrors.push_back(0);
                phvPoints.push_back(averageHV);
                phvPointsErrors.push_back(0);
            }
            if (predictionsTest.find(n.first) != predictionsTest.end()){
                predictedPointsP.push_back(predictionsTest[n.first][chamberI]);
                predictedPointsErrorsP.push_back(0);
                phvPointsP.push_back(averageHV);
                phvPointsErrorsP.push_back(0);
            }

            //hvPoints.push_back(closest->second[1*24+chamberI]);
            hvPoints.push_back(averageHV);
            hvPointsErrors.push_back(0);
            //pressurePoints.push_back(closest->second[0*24+chamberI]);
            pressurePoints.push_back(averageP);
            pressurePointsErrors.push_back(0);

            //cout << n.first << "    " <<  averageHV << "  " << n.second[chamberI].first << endl;
        }
        double minPressure = *min_element(pressurePoints.begin(), pressurePoints.end()); double maxPressure = *max_element(pressurePoints.begin(), pressurePoints.end());

        TGraphErrors* grTvsHV = new TGraphErrors(hvPoints.size(),&hvPoints[0], &targetPoints[0], &hvPointsErrors[0], &targetPointsErrors[0]);
        TGraphErrors* grPressurevsHV = new TGraphErrors(hvPoints.size(),&hvPoints[0], &pressurePoints[0], &hvPointsErrors[0], &pressurePointsErrors[0]);
        TGraphErrors* grPredictedvsHV = new TGraphErrors(phvPoints.size(),&phvPoints[0], &predictedPoints[0], &phvPointsErrors[0], &predictedPointsErrors[0]);
        TGraphErrors* grPredictedPvsHV = new TGraphErrors(phvPointsP.size(),&phvPointsP[0], &predictedPointsP[0], &phvPointsErrorsP[0], &predictedPointsErrorsP[0]);
        canvas->cd();
        canvas->SetGridy();
        canvas->SetGridx();
        grTvsHV->SetName("grTvsHV");
        grTvsHV->SetTitle("grTvsHV;High Voltage [V];Mean ToT [ns]");
        //grTvsHV->SetTitle("grTvsHV;High Voltage [V];ToT distribution width [ns]");
        //grTvsHV->SetTitle("grTvsHV;High Voltage [V];Signal-To-Valley ratio");
        //grTvsHV->SetTitle("grTvsHV;High Voltage [V];Erfc((valleyPos-signalPeak)/2#sigma)");
        grPressurevsHV->SetName("grPressurevsHV");
        grTvsHV->SetMarkerStyle(20);    grTvsHV->SetMarkerSize(1.3);    grTvsHV->SetLineColor(1);    grTvsHV->SetMarkerColor(1);    grTvsHV->SetLineWidth(1);
        grPressurevsHV->SetMarkerStyle(21);    grPressurevsHV->SetMarkerSize(1.4);    grPressurevsHV->SetLineColor(kBlue);    grPressurevsHV->SetMarkerColor(kBlue);    grPressurevsHV->SetLineWidth(2);
        
        //grTvsHV->GetYaxis()->SetRangeUser(0,grTvsHV->GetHistogram()->GetMaximum());
        grTvsHV->Draw("AP");
        //TF1* fTvsHV = new TF1("fTvsHV", "[0]*x+[1]",1600,1800);
        //TF1* fTvsHV = new TF1("fTvsHV", "twoLinesFit",1600,1800,5);
        //TF1* fTvsHV = new TF1("fTvsHV", "[0] + [1]*(x-[3]) + [2]*pow(x-[3],2)",1600,1800);
        //TF1* fTvsHV = new TF1("fTvsHV", "[0] + [1]*(x-[3]) + [2]*exp((x-[3])/[4])",1600,1790);
        TF1* fTvsHV = new TF1("fTvsHV", "[0] + [1]*(x-[2])",1600,1790);
        //fTvsHV->SetParameters(0.5,0,0.5,1700,500);
        fTvsHV->SetParameters(0.5,0,1710);
        grTvsHV->Fit("fTvsHV","","",1600,1780);

        gPad->Update(); // Force canvas update before modifying axes
        TGaxis *rightAxis = new TGaxis(gPad->GetUxmax(), gPad->GetUymin(),
                                    gPad->GetUxmax(), gPad->GetUymax(),
                                    minPressure-2, maxPressure+2, 510, "M");
        rightAxis->SetTitle("Atmospheric Pressure [a.u.]");
        rightAxis->SetLineColor(kBlack);
        rightAxis->SetLabelColor(kRed);
        rightAxis->SetTitleColor(kBlack);
        //rightAxis->Draw();

        // Customize and draw the second graph
        double yMin1 = grTvsHV->GetHistogram()->GetMinimum();
        double yMax1 = grTvsHV->GetHistogram()->GetMaximum();
        cout << yMin1 << "      " << yMax1 << endl;
        //double scaleFactor = (grTvsHV->GetYaxis()->GetMaximum() - grTvsHV->GetYaxis()->GetMinimum()) / (maxPressure*1.2 - minPressure/1.2);
        double scaleFactor = (yMax1 - yMin1) / (maxPressure+4 - minPressure);
        for (int i = 0; i < hvPoints.size(); i++) {
            double x, y;
            grPressurevsHV->GetPoint(i, x, y);
            double newY = yMin1 + (y - minPressure + 2) * scaleFactor;
            grPressurevsHV->SetPoint(i, x, newY);
        }
        //grPressurevsHV->Draw("P"); // "SAME" overlays it on the existing plot

        grPredictedvsHV->SetMarkerStyle(26);    grPredictedvsHV->SetMarkerSize(1.0);    grPredictedvsHV->SetLineColor(3);    grPredictedvsHV->SetMarkerColor(3);    grPredictedvsHV->SetLineWidth(1);
        grPredictedPvsHV->SetMarkerStyle(24);   grPredictedPvsHV->SetMarkerSize(1.0);   grPredictedPvsHV->SetLineColor(3);   grPredictedPvsHV->SetMarkerColor(2);   grPredictedPvsHV->SetLineWidth(1);
        grPredictedvsHV->Draw("Psame");
        grPredictedPvsHV->Draw("Psame");


        TF1* fitExpHV = expGraphsHV[chamberI]->GetFunction("pol1");
        double scaleFactorExp = fTvsHV->Eval(1710) / fitExpHV->Eval(1710);
        for (int i = 0; i < expGraphsHV[chamberI]->GetN(); ++i) {
            double x, y;
            expGraphsHV[chamberI]->GetPoint(i, x, y);
            expGraphsHV[chamberI]->SetPoint(i, x, y * scaleFactorExp);
        }
        expGraphsHV[chamberI]->SetMarkerStyle(21);
        expGraphsHV[chamberI]->SetMarkerColor(4);
        expGraphsHV[chamberI]->Draw("Psame");

        slopesCosmic.push_back(fTvsHV->GetParameter(1)/fTvsHV->Eval(1710));
        slopesBeam.push_back(fitExpHV->GetParameter(1)/fitExpHV->Eval(1710));
        

        TLegend *legend1 = new TLegend(0.2, 0.6, 0.5, 0.8);
        legend1->AddEntry(grTvsHV, "ToT mean", "lpe");
        //legend1->AddEntry(grPressurevsHV, "Atm. Pressure scaled", "lpe");
        legend1->AddEntry(fTvsHV, "pol(1)+exp fit", "l");
        legend1->AddEntry(grPredictedPvsHV, "ToT predicted cosmic", "lp");
        legend1->AddEntry(expGraphsHV[chamberI], "ToT beam scaled", "lp");
        legend1->SetBorderSize(0);
        legend1->Draw();
        
        TPaveText *title1 = new TPaveText(0.2, 0.82, 0.5, 0.88, "NDC"); // NDC = Normalized device coordinates
        title1->AddText(Form("module %d, sector %d",chamberI/6+1,chamberI%6+1));
        title1->SetFillColor(0); // Transparent background
        title1->SetTextSize(0.04);
        title1->SetTextAlign(22); // Center alignment
        title1->Draw();

        //TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);
        canvas->Update();


        ///// Difference between fit vs other parameters

        for (int i = 0; i < hvPoints.size(); i++) {
            targetPoints[i] = targetPoints[i] - fTvsHV->Eval(hvPoints[i]);
            pressurePoints[i] = (pressurePoints[i] - 1011)*3;
        }
        TGraphErrors* grTvsHVdifferenceFit = new TGraphErrors(hvPoints.size(),&hvPoints[0], &targetPoints[0], &hvPointsErrors[0], &targetPointsErrors[0]);
        grTvsHVdifferenceFit->SetName("grTvsHVdifferenceFit");
        grTvsHVdifferenceFit->SetTitle("grTvsHVdifferenceFit;High Voltage [V];ToT mean - fit [ns]");
        grTvsHVdifferenceFit->SetMarkerStyle(20);    grTvsHVdifferenceFit->SetMarkerSize(1.3);    grTvsHVdifferenceFit->SetLineColor(1);    grTvsHVdifferenceFit->SetMarkerColor(1);    grTvsHVdifferenceFit->SetLineWidth(1);
        
        TGraphErrors* grPvsHVshifted = new TGraphErrors(hvPoints.size(),&hvPoints[0], &pressurePoints[0], &hvPointsErrors[0], &pressurePointsErrors[0]);
        //TGraphErrors* grPvsHVshifted = new TGraphErrors(targetPoints.size(),&targetPoints[0], &pressurePoints[0], &hvPointsErrors[0], &pressurePointsErrors[0]);
        grPvsHVshifted->SetName("grPvsHVshifted");
        grPvsHVshifted->SetMarkerStyle(21);    grPvsHVshifted->SetMarkerSize(1.3);    grPvsHVshifted->SetLineColor(kBlue);    grPvsHVshifted->SetMarkerColor(kBlue);    grPvsHVshifted->SetLineWidth(1);
        c2->SetGridy();
        c2->cd();
        grTvsHVdifferenceFit->Draw("AP");
        grPvsHVshifted->Draw("Psame");

        TLegend *legend2 = new TLegend(0.6, 0.7, 0.9, 0.9);
        legend2->AddEntry(grTvsHVdifferenceFit, "ToT mean - fit", "lpe");
        legend2->AddEntry(grPvsHVshifted, "Atm. Pressure scaled", "lpe");
        legend2->SetBorderSize(0);
        legend2->Draw();

        title1->Draw();

        canvas->Update();
        c2->Update();
        cin.get();
        grTvsHV->SetName(Form("grTvsHV_%d",chamberI));
        canvas->SetName(Form("canvas_%d",chamberI));
        grTvsHV->Write();
        canvas->Write();
    }
    fOut->Close();

    std::cout << "Chamber | Beam slope [%/10V] | Cosmic slope [%/10V] | Ratio \n";
    for (int i = 0; i < 12; ++i) {
        double ratio = (slopesCosmic[i] != 0) ? (slopesBeam[i] / slopesCosmic[i]) : 0.0;
        std::cout << "   " << i << "    |   " << slopesBeam[i]*1000 << "   |   " << slopesCosmic[i]*1000 << "   |   " << ratio << " %\n";
    }

}


void drawTargetPredictionComparisonCosmics(){
    int nodes_length = 24;
    TFile* fileOut = new TFile("savedPlotsTemp11.root","RECREATE");

    const char* timeFormat = "%d/%m %H:%M";

    //const std::string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";
    int chanTo = 2;
    /// Reading mean and std
    ifstream fin1;
    vector<double> meanValues, stdValues;
	fin1.open((saveLocation+"function_prediction/meanValuesT.txt").c_str());
    double meanV = 0;
    while (fin1 >> meanV){
        meanValues.push_back(meanV);
    }
    fin1.close();
    fin1.open((saveLocation+"function_prediction/stdValuesT.txt").c_str());
    double stdV = 0;
    while (fin1 >> stdV){
        stdValues.push_back(stdV);
    }
    fin1.close();

    TH2F* targetVsSource = new TH2F("targetVsSource","",100,-5,5,100,-5,5);

    /// Reading target values and averaging -> map
    ifstream fin2;
    TH1F* hdTarget1D = new TH1F("hdTarget1D","target1-target2;#errors;counts",1000,-100,100);
	//fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCSecModPreciseFit2.dat").c_str());
    //fin2.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOut.dat");
    //fin2.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOutSeparative2.dat");
    //fin2.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOutSeparative2properPars.dat");
    fin2.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOutSeparative100_2properPars.dat");
    //fin2.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOutSeparative2WidthproperPars.dat");
    std::map<int, vector< pair<double,double> > > targets;
    std::map<int, int > targetsRunEnd;
    std::map<int, vector< pair<double,double> > > targetsNormalized;
    vector<int> runsTargets;
    vector<double> valuesTargets,valuesTargetsNormalized,runsTargetsErr,valuesTargetsErr;
    vector< pair<double, double> >  predTargetNodes;
    while (true){
        int run, runEnd, sector, mod;
        double target, targetErr;
        vector< pair<double, double> > targetNodes;
        vector< pair<double, double> > targetNodesNormalized;
        double meanTarget, meanTargetErr;
        for (size_t i = 0; i < 24; i++){
            //if (!(fin2 >> run >> sector >> mod >> target >> targetErr)) {
            if (!(fin2 >> run >> runEnd >> sector >> mod >> target >> targetErr)) {
                // We failed to read (likely end-of-file or bad format):
                // break out of both loops so we don’t process bad data.
                goto end_of_read;
            }
            if (predTargetNodes.size()>0){
                //if (fabs(target - predTargetNodes[i].first) < predTargetNodes[i].second*5)
                //    targetErr = fabs(target - predTargetNodes[i].first)/2.;
                //else
                //    targetErr *= 2;
                if (targetErr == 0){
                    targetErr = predTargetNodes[i].second;
                }
                hdTarget1D->Fill((target - predTargetNodes[i].first)/targetErr);
            }
            meanTarget+=(target-meanValues[meanValues.size()-nodes_length*2+i])/stdValues[meanValues.size()-nodes_length*2+i];
            meanTargetErr += pow(targetErr/stdValues[meanValues.size()-nodes_length*2+i],2);
            if (i == chanTo && run < 11445503850){
                runsTargets.push_back(run);
                valuesTargetsNormalized.push_back((target-meanValues[meanValues.size()-24+i])/stdValues[meanValues.size()-24+i]);
                //valuesTargetsNormalized.push_back((target-150)/50);
                valuesTargets.push_back(target);
                runsTargetsErr.push_back(0);
                //valuesTargetsErr.push_back(targetErr/stdValues[meanValues.size()-nodes_length*2+i]);
                valuesTargetsErr.push_back(targetErr);
                cout << run << "    " << runEnd-run << endl;
                //if (target < 20){
                    //cout << run << "    " << target << "    " << targetErr << endl;
                //}
            }
            //cout << run << endl;
            //if (valuesTargets.size() >= 1){
                //targetNodes.push_back(make_pair(valuesTargets[valuesTargets.size()-1],valuesTargetsErr[valuesTargetsErr.size()-1]));
                //targetNodesNormalized.push_back(make_pair(valuesTargetsNormalized[valuesTargetsNormalized.size()-1],valuesTargetsErr[valuesTargetsErr.size()-1]));
                targetNodes.push_back(make_pair(target*1,targetErr*1));
                targetNodesNormalized.push_back(make_pair(target,targetErr));
            //}
            
        }
        //meanTarget = meanTarget/24.;
        //meanTargetErr = sqrt(meanTargetErr)/24.;
        //runsTargets.push_back(run);
        //valuesTargets.push_back(meanTarget);
        //runsTargetsErr.push_back(0);
        //valuesTargetsErr.push_back(meanTargetErr);

        //targets[int((run+runEnd)/2)] = targetNodes;
        targets[run] = targetNodes;
        targetsNormalized[run] = targetNodesNormalized;
        targetsRunEnd[run] = runEnd;
        predTargetNodes = targetNodes;
    }
    end_of_read:;
    fin2.close();

    /// Reading MDC epics data
    //fin2.open((saveLocation + "info_tables/MDCModSecPrecise.dat").c_str());
    //fin2.open((saveLocation + "info_tables/MDCModSecPreciseExtended.dat").c_str());
    //fin2.open((saveLocation + "info_tables/MDCModSecPreciseFeb22ends.dat").c_str());
    //fin2.open((saveLocation + "info_tables/MDCModSecPreciseCosmic25VaryHV1.dat").c_str());
    //fin2.open((saveLocation + "info_tables/MDCModSecPreciseCosmic25_2VaryHV1.dat").c_str());
    fin2.open((saveLocation + "info_tables/MDCModSecPreciseCosmic25_106VaryHVends9a.dat").c_str());
    std::map<int, vector< double > > mdcChanChamb0;
    cout << "a" << endl;
    const int channelSize = 9;
    vector<double> runsChans;
    vector<double> valuesChans[channelSize][24];
    vector<double> runsChans2;
    vector<double> valuesChans2;
    TH1F* hHvValues = new TH1F("hHvValues","",100,-10,10);
    string line;
    nodes_length = 24;
	
    while (!fin2.eof()){
        int run;
        fin2 >> run;
        vector<double> valuesChan1;
        double fillHist = -100;
        for (size_t i = 0; i < channelSize; i++){
            double valueBuffChan;
            double valueChan;
            //cout << valueChan << endl;
            for (size_t j = 0; j < 24; j++){
                fin2 >> valueBuffChan;
                //if (j == chanTo){
                    if (i == 0)
                        //valuesChans[i][j].push_back(-(valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j]);
                        valuesChans[i][j].push_back(valueBuffChan/13.); 
                    else{
                        if (stdValues[i*nodes_length+j] != 0)
                            //valuesChans[i][j].push_back(-0.5*(valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j]); 
                            valuesChans[i][j].push_back(valueBuffChan/13.); 
                        else
                            valuesChans[i][j].push_back(valueBuffChan-meanValues[i*nodes_length+j]); 
                    }
                    //valuesChans.push_back(valueBuffChan);
                    valuesChan1.push_back(valueBuffChan);
                    hHvValues->Fill((valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j]);
                //}

                if (targetsNormalized.find(run) != targetsNormalized.end() && i == 0 && j == chanTo && run < 445503850){
                //if (i == 0 && j == chanTo){
                    //targetVsSource->Fill(valuesChans[i][valuesChans[i].size()-1], targetsNormalized[run][j].first);
                    runsChans2.push_back((double)(runToDateNumber(run)));
                    valuesChans2.push_back((valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j] + targetsNormalized[run][j].first );
                    fillHist = (valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j] + targetsNormalized[run][j].first ;
                }
                else if (targetsNormalized.find(run) != targetsNormalized.end() && i == 3 && j == chanTo && fillHist != -100 && run < 445503850){
                    targetVsSource->Fill(-0.5*(valueBuffChan-meanValues[i*nodes_length+j])/stdValues[i*nodes_length+j], fillHist );
                }
            }
        }
        runsChans.push_back((double)(runToDateNumber(run)));
        mdcChanChamb0[run] = valuesChan1;
    }
    
    cout << "b" << endl;
    nodes_length = 12;
    fin2.close();
    TGraph* grMDCchan[channelSize];
    TGraph* grMDCchan2;
    grMDCchan2 = new TGraph(runsChans2.size(),&runsChans2[0],&valuesChans2[0]);
    grMDCchan2->SetName("grMDCchan2");
    grMDCchan2->SetMarkerStyle(22);    grMDCchan2->SetMarkerSize(0.7);    grMDCchan2->SetLineColor(4);    grMDCchan2->SetMarkerColor(4);    grMDCchan2->SetLineWidth(3);
    grMDCchan2->GetXaxis()->SetTimeDisplay(1); grMDCchan2->GetXaxis()->SetTimeFormat(timeFormat);
    for (size_t i = 0; i < channelSize; i++){
        grMDCchan[i] = new TGraph(runsChans.size(),&runsChans[0],&valuesChans[i][chanTo][0]);
        grMDCchan[i]->SetMarkerStyle(22);    grMDCchan[i]->SetMarkerSize(0.7);    grMDCchan[i]->SetLineColor(1);    grMDCchan[i]->SetMarkerColor(1);    grMDCchan[i]->SetLineWidth(3);
        if (i == 2){
            grMDCchan[i]->SetMarkerStyle(22);    grMDCchan[i]->SetMarkerSize(1.0);    grMDCchan[i]->SetLineColor(3);    grMDCchan[i]->SetMarkerColor(3);    grMDCchan[i]->SetLineWidth(1);}
        grMDCchan[i]->GetXaxis()->SetTimeDisplay(1); grMDCchan[i]->GetXaxis()->SetTimeFormat(timeFormat);
    }



    TGraphErrors* gr2 = new TGraphErrors();

    //valuesTargets = runningMeanVector(valuesTargets, 20);
    for (size_t i = 0; i < runsTargets.size(); i++){
        //targets[runsTargets[i]] = valuesTargets[i];
        //int n = gr2->GetN();
        //gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),targetsNormalized[runsTargets[i]][chanTo].first);
        //gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),targets[runsTargets[i]][chanTo].first);
        //gr2->SetPoint(n,(double)(runToDateNumber(runsTargets[i])),valuesTargets[i]);
        //gr2->SetPoint(n,(double)(runsTargets[i]),valuesTargets[i]);
        //gr2->SetPointError(n, runsTargetsErr[i], valuesTargetsErr[i]);
    }
    for (auto &a: targets){
        int n = gr2->GetN();
        gr2->SetPoint(n,(double)(runToDateNumber(a.first)),a.second[chanTo].first);
        gr2->SetPointError(n,0.0,a.second[chanTo].second);
    }
    //downsampleGraphInPlace(gr2,0.1);
    gr2->SetMarkerStyle(22);
    gr2->SetMarkerSize(1.5);
    gr2->SetMarkerColor(4);
    gr2->SetLineColor(4);
    gr2->SetLineWidth(2);
    cout << gr2->GetN() << endl;
    cout << "finished with target " << endl;

    fin1.open((saveLocation+ "function_prediction/predicted/predictedCosmics_.txt").c_str());
    //fin1.open((saveLocation+ "function_prediction/predicted/predicted_0.txt").c_str());
    std::map<int, vector< double > > predictions;
    double runPrevT = 0;
    while (!fin1.eof()){
        double run = 0;
        double prediction = 0;
        vector<double> predictionNodes;
        fin1 >> run;
        for (size_t i = 0; i < nodes_length; i ++){
            fin1 >> prediction;
            predictionNodes.push_back(prediction);
            //cout << prediction << endl;
        }
        if (run != runPrevT)
            predictions[(int)run] = predictionNodes;
        runPrevT = run;
    }
    fin1.close();

    fin2.open((saveLocation+ "function_prediction/predicted/predictedCosmics1_.txt").c_str());
    //fin2.open((saveLocation+ "function_prediction/predicted/predicted1_0.txt").c_str());
    std::map<int, vector< double > > predictionsTest;
    runPrevT = 0;
    while (!fin2.eof()){
        double run = 0;
        double prediction = 0;
        vector<double> predictionNodes;
        fin2 >> run;
        for (size_t i = 0; i < nodes_length; i++){
            fin2 >> prediction;
            predictionNodes.push_back(prediction);
        }
        if (run != runPrevT)
            predictionsTest[(int)run] = predictionNodes;
        runPrevT = run;
    }
    fin2.close();

    cout << "lasjfdksf;" << endl;

    /// _____ making graphErrors for prediction-target
    vector<double> runsPredictedTarget, diffPredictedTarget, runsPredictedTargetErr, diffPredictedTargetErr;
    vector<double> diffPredictedTargetDiff;
    //TH2F* hpredictedTarget = new TH2F("hpredictedTarget","(predicted-target)/targetErr;run;(P-T)/T_{err} [#sigma]",10000,runToDateNumber(runsTargets[1])-100000,runToDateNumber(runsTargets[runsTargets.size()-1])+100000, 1000, 0.8, 1.2);
    TH2F* hpredictedTarget = new TH2F("hpredictedTarget","(predicted-target)/targetErr;run;(P-T)/T_{err} [#sigma]",10000,runToDateNumber(runsTargets[1])-100000,runToDateNumber(445559647), 1000, 0.8, 1.2);
    TProfile* ppredictedTarget = new TProfile("ppredictedTarget","(predicted-target)/targetErr average;run;(P-T)/T_{err} [#sigma]",1000,runToDateNumber(443654673),runToDateNumber(446359647));
    TH1F* hpredictedTarget1D = new TH1F("hpredictedTarget1D","predicted-target;#std;counts",100,-10,10);
    TH2F* h2dPredictionsVsTargetTrain = new TH2F("h2dPredictionsVsTargetTrain","Predictions Vs Target training; prediction; target",1000,0,100,1000,0,100);
    TH2F* h2dPredictionsVsTargetTest = new TH2F("h2dPredictionsVsTargetTest","Predictions Vs Target testing; prediction; target",1000,0,100,1000,0,100);
    for (auto x: predictions){
        for (size_t i = chanTo; i < chanTo+1; i ++){
            runsPredictedTarget.push_back((double)(runToDateNumber((int)(x.first))));
            runsPredictedTargetErr.push_back(0);
            diffPredictedTarget.push_back( x.second[i] );
            diffPredictedTargetErr.push_back(0);
        }
        if (targets.find(x.first) == targets.end())
            continue;
        for (size_t i = chanTo; i < chanTo+1; i ++){
            //diffPredictedTarget.push_back( (x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            diffPredictedTargetDiff.push_back( (x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24+i] );
            //diffPredictedTarget.push_back( (x.second[i] - meanValues[meanValues.size()-24+i])/stdValues[meanValues.size()-24+i] );
            h2dPredictionsVsTargetTrain->Fill(x.second[i],targets[x.first][i].first);
            //hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i]/targets[x.first][i].first));
            if (fabs((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second ) < 10)
                ppredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            //hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24*2+i] );
            //hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
        }
    }
    vector<double> runsPredictedTargetP, diffPredictedTargetP, runsPredictedTargetErrP, diffPredictedTargetErrP;
    vector<double> diffPredictedTargetPDiff;
    for (auto x: predictionsTest){
        for (size_t i = chanTo; i < chanTo+1; i ++){
            runsPredictedTargetP.push_back((double)(runToDateNumber((int)(x.first))));
            runsPredictedTargetErrP.push_back(0);
            diffPredictedTargetP.push_back( x.second[i] );
            diffPredictedTargetErrP.push_back(0);
        }
        if (targets.find(x.first) == targets.end())
            continue;
        for (size_t i = chanTo; i < chanTo+1; i ++){
        //for (size_t i = 0; i < 18; i ++){
            //diffPredictedTargetP.push_back( (x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            diffPredictedTargetPDiff.push_back( (x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24+i] );
            //diffPredictedTargetP.push_back( (x.second[i] - meanValues[meanValues.size()-24+i])/stdValues[meanValues.size()-24+i] );
            h2dPredictionsVsTargetTest->Fill(x.second[i],targets[x.first][i].first);
            //hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            hpredictedTarget->Fill(runToDateNumber(x.first),(x.second[i]/targets[x.first][i].first));
            if (fabs((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second ) < 10)
                ppredictedTarget->Fill(runToDateNumber(x.first),(x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
            //hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/stdValues[meanValues.size()-24*2+i] );
            hpredictedTarget1D->Fill((x.second[i] - targets[x.first][i].first)/targets[x.first][i].second );
        }
    }

    cout << "xx" << endl;

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

    //downsampleGraphInPlace(grPT,0.1);
    //downsampleGraphInPlace(grPTP,0.1);

    TGraphErrors* grPTd = new TGraphErrors(runsPredictedTarget.size(),&diffPredictedTargetDiff[0], &diffPredictedTarget[0], &runsPredictedTargetErr[0], &diffPredictedTargetErr[0]);
    TGraphErrors* grPTPd = new TGraphErrors(runsPredictedTargetP.size(),&diffPredictedTargetPDiff[0], &diffPredictedTargetP[0], &runsPredictedTargetErrP[0], &diffPredictedTargetErrP[0]);
    grPT->SetMarkerStyle(26);    grPT->SetMarkerSize(1.5);    grPT->SetLineColor(3);    grPT->SetMarkerColor(3);    grPT->SetLineWidth(1);
    grPTP->SetMarkerStyle(24);   grPTP->SetMarkerSize(1.5);   grPTP->SetLineColor(3);   grPTP->SetMarkerColor(2);   grPTP->SetLineWidth(1);
    

    cout << "x1" << endl;

    TDatime da(2022,2,3,15,58,00);
    gStyle->SetTimeOffset(da.Convert());

    //grPT->GetXaxis()->SetLimits(grPT->GetX()[0]-100000, grPT->GetX()[runsPredictedTarget.size()-1]*1.3);
    //grPT->GetYaxis()->SetRangeUser(-6,6);
    //gr2->GetXaxis()->SetTimeDisplay(1); gr2->GetXaxis()->SetTimeFormat("%d/%m");
    gr2->GetXaxis()->SetTimeDisplay(1); gr2->GetXaxis()->SetTimeFormat("%H:%M");
    grPT->GetXaxis()->SetTimeDisplay(1); grPT->GetXaxis()->SetTimeFormat("%d/%m");
    grPTP->GetXaxis()->SetTimeDisplay(1); grPTP->GetXaxis()->SetTimeFormat("%d/%m");

    hpredictedTarget->GetXaxis()->SetTimeDisplay(1); hpredictedTarget->GetXaxis()->SetTimeFormat("%d/%m");
    ppredictedTarget->GetXaxis()->SetTimeDisplay(1); ppredictedTarget->GetXaxis()->SetTimeFormat("%d/%m");
    mean_graph->GetXaxis()->SetTimeDisplay(1); mean_graph->GetXaxis()->SetTimeFormat("%d/%m");
    std_dev_graph->GetXaxis()->SetTimeDisplay(1); std_dev_graph->GetXaxis()->SetTimeFormat("%d/%m");

    cout << "x2" << endl;

    TCanvas* canvas = new TCanvas("canvas", "Prediction vs Target calibration comparison", 1920, 950);
    TPad* pad = new TPad("pad", "Pad", 0.01, 0.01, 0.99, 0.99);
    pad->Draw();
    pad->cd();
    pad->SetGridy();

    gr2->SetTitle("Prediction vs Target calibration comparison;Time [hour:minute];ToT distribution mean [ns]");
    //gr1->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    //grP->SetTitle("Prediction vs Target calibration comparison;date;-dE/dx [MeV g^{-1} cm^{2}]");
    grPT->SetTitle("Prediction - Target;date;#sigma");
    grPTP->SetTitle("Prediction - Target;date;#sigma");

    TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(gr2, "Mean signal ToT from fit, [a.u.]", "lpe");
    //legend->AddEntry(gr1, "predicted, train dataset", "l");
    //legend->AddEntry(grP, "predicted, test dataset", "l");
    //legend->AddEntry(grPT, "Predicted, train dataset", "lpe");
    //legend->AddEntry(grPTP, "Predicted, test dataset", "lpe");
    legend->AddEntry(grPTP, "ToT predicted, fine-tuned cosmic", "lp");

    //legend->AddEntry(grMDCchan2, "ToT - atm.pressure", "lpe");
    //legend->AddEntry(grMDCchan[0], "Atm pressure", "lpe");
    //legend->AddEntry(grMDCchan[3], "Overpressure", "lpe");
    //legend->AddEntry(grMDCchan[1], "High Voltage, [V]", "lpe");


    cout << "x3" << endl;

    //hdTarget1D->SetLineColor(2);
    //hdTarget1D->Draw();
    //hpredictedTarget1D->DrawNormalized("same",hdTarget1D->GetEntries());
    //hpredictedTarget1D->Draw();
    //gr2->GetYaxis()->SetRangeUser(0,100);
    
    gr2->Draw("AP");
    //grPT->Draw("Psame");
    //grPTP->Draw("Psame");


    //gr2->GetXaxis()->SetLimits(runToDateNumber(runsTargets[1])-100000, runToDateNumber(runsTargets[runsTargets.size()-1])+100000);
    //gr2->GetXaxis()->SetLimits(runToDateNumber(runsTargets[1])-100000, runToDateNumber(445559647));
    //hpredictedTarget->GetXaxis()->SetLimits(gr2->GetX()[0]-100000, gr2->GetX()[runsPredictedTarget.size()-1]*1.3);
    
    //drawTargetPredictionComparisonWithAsubplot(gr2, grPT, grPTP, hpredictedTarget);   //!!!!!!!!!!!!!!!!!!!!!!!!1

    //TCanvas* canvas2 = new TCanvas("canvas2", "Prediction vs Target calibration comparison", 1920, 950);
    //canvas2->cd();
    //grMDCchan[3]->SetMarkerColor(5);
    //grMDCchan2->Draw("AP");
    //grMDCchan[0]->Draw("Psame");
    //grMDCchan[0]->Draw("Psame");
    //grMDCchan[0]->Draw("Psame");
    //grMDCchan[1]->Draw("AP");
    grMDCchan[1]->Draw("Psame");
    //grMDCchan[6]->Draw("Psame");
    //grMDCchan[3]->Draw("Psame");

    //h2dPredictionsVsTargetTrain->Draw("colz");
    //h2dPredictionsVsTargetTest->Draw("colz");
    
    //grMDCchan[6]->Draw("AP");
    //targetVsSource->Draw();
    //hHvValues->Draw();
    //gr1->Draw("Psame");
    //grP->Draw("Psame");
    //gr1L->Draw("Psame");
    //grPL->Draw("Psame");
    //grPT->Draw("AP");
    //hpredictedTarget->SetDrawOption("colz");
    //hpredictedTarget->Draw("colz");
    //mean_graph->Draw("Psame");
    //std_dev_graph->GetYaxis()->SetRangeUser(-10,10);
    //std_dev_graph->Draw("APL");
    //std_dev_graph1->Draw("PLsame");
    //ppredictedTarget->Draw("same");
    legend->Draw();

    TPaveText *title1 = new TPaveText(0.2, 0.82, 0.5, 0.88, "NDC"); // NDC = Normalized device coordinates
    title1->AddText(Form("module %d, sector %d",chanTo/6+1,chanTo%6+1));
    title1->SetFillColor(0); // Transparent background
    title1->SetTextSize(0.04);
    title1->SetTextAlign(22); // Center alignment
    title1->Draw();


    
    //drawTvsHV(targets,targetsRunEnd,mdcChanChamb0,predictions,predictionsTest);




    fileOut->cd();
    hpredictedTarget1D->Write();
    //fileOut->Close();
}



void drawRunLengthDistribution(){
    //vector<int> runBorders = dateRunF::loadrunlist(0, 1e10); //444140006
    vector< pair<int,int> > runBorders = dateRunF::loadrunlistWithEnds(0, 1e10); 
    TH1F* hRunLength = new TH1F("hRunLength","Run length distribution;run length [s];counts",1000,0,1000);
    TH1F* hRunDiff = new TH1F("hRunDiff","Run distance distribution;run distance [s];counts",10000,0,10000);
    for (size_t i = 0; i < runBorders.size()-2; i++){
        //hRunLength->Fill(runBorders[i+1]-runBorders[i]);
        hRunLength->Fill(runBorders[i].second-runBorders[i].first);

        int run1 = (int)((runBorders[i].second+runBorders[i].first)/2);
        int run2 = (int)((runBorders[i+1].second+runBorders[i+1].first)/2);
        hRunDiff->Fill(run2-run1);
    }
    TCanvas* canvas = new TCanvas("canvas", "Run length distribution", 1720, 1250);
    gStyle->SetOptStat(0);
    hRunLength->SetLineColor(1);
    hRunLength->SetLineWidth(2);
    //hRunLength->Draw();
    hRunDiff->SetLineColor(1);
    hRunDiff->SetLineWidth(2);
    hRunDiff->Draw();
    //TLine *l1 = new TLine(30,0,30,hRunLength->GetMaximum()*1.1);
    //TLine *l2 = new TLine(200,0,200,hRunLength->GetMaximum()*1.1);
    TLine *l1 = new TLine(30,0,30,hRunDiff->GetMaximum()*1.1);
    TLine *l2 = new TLine(450,0,450,hRunDiff->GetMaximum()*1.1);
    l1->SetLineColor(2);
    l2->SetLineColor(2);
    l1->SetLineWidth(2);
    l2->SetLineWidth(2);
    hRunLength->GetYaxis()->SetRangeUser(0.1, hRunLength->GetMaximum()*1.1);
    hRunDiff->GetYaxis()->SetRangeUser(0.1, hRunDiff->GetMaximum()*1.1);
    //l1->Draw("same");
    l2->Draw("same");
    canvas->Update();

    //canvas->SaveAs("/home/localadmin_jmesschendorp/Pictures/runLengthDistribution.png");
}