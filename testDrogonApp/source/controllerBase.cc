#include "../include/controllerBase.h"


#include "../include/functions.h"
#include "../include/constants.h"

#include <chrono>

using namespace std;
using namespace preTrainFunctions;
using namespace vectorFunctions;
using namespace dateRunF; 

ControllerBase::ControllerBase(TriggerDataManager* itriggerManager, EpicsDBManager* iepicsManager, NeuralNetwork* ineuralNetwork, ServerData* iserverData){
    triggerManager = itriggerManager;
    epicsManager = iepicsManager;
    neuralNetwork = ineuralNetwork;
    serverData = iserverData;
    
    sentenceLength = 15;

    out = new TFile("outHome.root","RECREATE");
    out->cd();

    hTriggerDataTime = new TH1F("hTriggerDataTime", "trigger data reading time; t, ms; counts", 100, 0, 500);
    hEpicsDataTime = new TH1F("hEpicsDataTime", "epics data reading time; t, ms; counts", 100, 0, 500);
    hNetworkDataTime = new TH1F("hNetworkDataTime", "NN prediction calculation time; t, ms; counts", 100, 0, 10);

}

ControllerBase::~ControllerBase(){
    out->Close();
}


void ControllerBase::compareTargetPredictionFromTraining(){
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
    cin.get();
    cin.get();
}


vector<float> ControllerBase::makeNNInputTensor(int run){
    vector< vector<float> > tempInpInversed;
    vector< vector<double> > tempDBInversed;
    vector<int> runsDBind;
    vector<float> nnInpTens;

    vector<int> runBorders = loadrunlist(0, 1e10);
    auto p = std::find(runBorders.begin(), runBorders.end(), run);
    if (p == runBorders.end()){
        cout << "bad run?" << endl;
        return nnInpTens;
    }
    int index = std::distance(runBorders.begin(), p);

    int nBack = 0;
    while (tempInpInversed.size() < sentenceLength){
        size_t j = (index-nBack) < 0 ? 0 : (size_t)(index-nBack);
        nBack+=1;
        //vector <double> trPars = triggerManager->getTriggerData(runBorders[j])[0];
        //if (trPars.size()<1)
        //    continue;

        //vector<float> nnInpPars = neuralNetwork->formNNInput(epicsManager->getDBdata(runBorders[j], runBorders[j+1]), trPars);
        //vector<float> nnInpPars = neuralNetwork->formNNInput(epicsManager->getDBdata(runBorders[j], runBorders[j+1]));
        //tempInpInversed.push_back(nnInpPars);
        tempDBInversed.push_back(epicsManager->getDBdata(runBorders[j], runBorders[j+1]));
        runsDBind.push_back(j);
    }

    bool firstRunDataWritten = false;
    for (int i = sentenceLength-1; i >= 0; i--){
        vector<float> nnInpPars = neuralNetwork->formNNInput(tempDBInversed[i]);
        //for (float x: tempInpInversed[i]){
        for (float x: nnInpPars){
            nnInpTens.push_back(x);
        }

        if (runsDBind[i] != 0)
            epicsManager->appendDBTable("app", runBorders[runsDBind[i]], tempDBInversed[i]);
        else{
            if (!firstRunDataWritten)
                epicsManager->appendDBTable("app", runBorders[runsDBind[i]], tempDBInversed[i]);
            firstRunDataWritten = true;
        }
    }
    return nnInpTens;
}

float ControllerBase::moveForwardCurrentNNInput(){
    
    auto start = std::chrono::high_resolution_clock::now();

    vector<int> runlist = serverData->getRunList();
    int currentRunIndex = serverData->getCurrentRunIndex();
    int currentRun = runlist[currentRunIndex];
    vector<float> currentNNInput = serverData->getCurrentNNInput();
    if (currentRunIndex+2 >= runlist.size()){
        return -1;
    }
    int nextRun = runlist[currentRunIndex+1];
    int nextNextRun = runlist[currentRunIndex+2];

    long long elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

    long long elapsed3 = 0;
    long long elapsed4 = 0;
    long long elapsed5 = 0;
    if (currentNNInput.size() < sentenceLength){
        currentNNInput = makeNNInputTensor(currentRun);
        if (currentNNInput.size() < sentenceLength)
            return -3;
    }
    else{

        elapsed3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

        currentRunIndex = currentRunIndex+1;
        currentRun = nextRun;
        nextRun = nextNextRun;
        //vector <double> trPars = triggerManager->getTriggerData(currentRun)[0];
        //if (trPars.size() <= 1){
        //    serverData->setCurrentRunIndex(currentRunIndex);
        //    return -2;
        //}
        
        elapsed4 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

        //vector<float> nnInpPars = neuralNetwork->formNNInput(epicsManager->getDBdata(currentRun, nextRun), trPars);
        vector<double> dbData = epicsManager->getDBdata(currentRun, nextRun);
        epicsManager->appendDBTable("app", currentRun, dbData);
        vector<float> nnInpPars = neuralNetwork->formNNInput(dbData);
        currentNNInput.erase(currentNNInput.begin(), currentNNInput.begin() + nnInpPars.size());
        for (size_t i = 0; i < nnInpPars.size(); i++){
            currentNNInput.push_back(nnInpPars[i]);
        }

    }
    
    elapsed5 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

    serverData->setCurrentNNInput(currentNNInput);
    serverData->setCurrentRunIndex(currentRunIndex);
    float output = neuralNetwork->getPrediction(currentNNInput);
    
    auto elapsed6 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

    cout << elapsed2 << "   " << elapsed3 << "   " << elapsed4 << "   " << elapsed5 << "   " << elapsed6 << endl;
    if (elapsed3!=0){
        hTriggerDataTime->Fill((elapsed4-elapsed3)/1000.);
        hEpicsDataTime->Fill((elapsed5-elapsed4)/1000.);
        hNetworkDataTime->Fill((elapsed6-elapsed5)/1000.);
    }

    return output;
}


// Old, not used
void ControllerBase::drawManyPredictions(){
    TGraph* gr = new TGraph();

    gr->SetTitle("prediction vs run; run; prediction");
    gr->SetMarkerStyle(22);
    gr->SetMarkerSize(1.5);
    gr->SetMarkerColor(4);
    gr->SetLineColor(4);

    /// Reading target values and averaging -> map    
    ifstream fin2;
	fin2.open((saveLocation + "info_tables/run-mean_dEdxMDC1.dat").c_str());
    std::map<int,double> targets;
    vector<int> runsTargets;
    vector<double> valuesTargets;
    while (!fin2.eof()){
        int run;
        double target, targetErr;
        fin2 >> run >> target >> targetErr; 
        targets[run] = target;
        runsTargets.push_back(run);
        valuesTargets.push_back(target);
    }
    //valuesTargets = runningMeanVector(valuesTargets, 20);
    for (size_t i = 0; i < runsTargets.size(); i++){
        targets[runsTargets[i]] = valuesTargets[i];
    }
    fin2.close();
    TGraph* gr2 = new TGraph();
    gr2->SetMarkerStyle(21);
    gr2->SetMarkerSize(1.0);
    gr2->SetMarkerColor(2);
    gr2->SetLineColor(2);

    //for (auto x: targets){
        //gr2->SetPoint(gr2->GetN(), runToDateNumber(x.first), x.second);
    //}


    ifstream fin1;
	fin1.open((saveLocation + "runlist.dat").c_str());
    TCanvas* canvas = new TCanvas("myCanvas", "My Canvas", 800, 600);
    TPad* pad = new TPad("myPad", "My Pad", 0, 0, 1, 1);
    pad->Draw();
    pad->cd();

    int run = 0;
    int count = 0;
    vector<int> runsPred;
    vector<double> valuesPred; 
    vector<float> nnInpTens;
    while (!fin1.eof() && count < 1100){

        int runNext;
        fin1 >> runNext;

        count +=1;
        if (count < 1000){
            //run = runNext;
            continue;
        }
        //count = 0;
        
        if (run != 0){
            vector <double> trPars = triggerManager->getTriggerData(run)[0];
            //cout << "a " << trPars.size() << endl;
            vector<double> dbPars = epicsManager->getDBdata(run, runNext);
            //cout << "b " << dbPars.size() << endl;
            if (trPars.size() > 1){
                vector<float> nnInpPars = neuralNetwork->formNNInput(epicsManager->getDBdata(run, runNext), triggerManager->getTriggerData(run)[0]);
                
                while (nnInpTens.size()<15*nnInpPars.size()){
                    for (size_t i = 0; i < nnInpPars.size(); i++){
                        nnInpTens.push_back(nnInpPars[i]);
                    }
                }

                nnInpTens.erase(nnInpTens.begin(), nnInpTens.begin() + nnInpPars.size());
                for (size_t i = 0; i < nnInpPars.size(); i++){
                    nnInpTens.push_back(nnInpPars[i]);
                }
                cout << nnInpTens[3] << " " << nnInpTens[15] << " " << nnInpTens[16] << endl;
                float predict = neuralNetwork->getPrediction(nnInpTens);
                
                runsPred.push_back(run);
                valuesPred.push_back((double)predict);


                //gr->SetPoint(gr->GetN(),runToDateNumber(run),(double)predict);
                //gr->GetXaxis()->SetRangeUser(gr->GetX()[0]-1,run+1);
                //gr->Draw("AP");

                gr2->SetPoint(gr2->GetN(),runToDateNumber(run),targets[run]);
                //gr2->Draw("Psame");

                //pad->Update();
            }
        }
        run = runNext;
    }
    fin1.close();


    //valuesPred = runningMeanVector(valuesPred, 20);
    for (size_t i = 0; i < runsPred.size(); i++){
        gr->SetPoint(gr->GetN(), runToDateNumber(runsPred[i]), valuesPred[i]);
    }


    TDatime da(2022,2,3,15,58,00);
    gStyle->SetTimeOffset(da.Convert());
    //gr->GetXaxis()->SetNdivisions(510, kTRUE);
    gr->GetXaxis()->SetTimeDisplay(1);
    gr->GetXaxis()->SetTimeFormat("%d %H");

    gr->Draw("AP");
    gr2->Draw("Psame");
}


void ControllerBase::changeRunList(){
    saveRunNumbers(serverData->getTrHistsLoc());
}


void ControllerBase::writeData(){
    out->cd();
    hTriggerDataTime->Write();
    hEpicsDataTime->Write();
    hNetworkDataTime->Write();
    out->Save();
    out->Close();
}

