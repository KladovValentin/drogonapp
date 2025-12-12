#include "../include/histsUpdater.h"

#include "../include/functions.h"
#include "../include/constants.h"

using namespace std;


//HistsUpdater::HistsUpdater(HistsUpdater* serv, HistsUpdater* controllerBase){
HistsUpdater::HistsUpdater(int argc, char** argv, std::shared_ptr<ControllerBase>& icontrollerBase){

    /// Add a path where the hists are stored
    /// Add hists and graphs here, no need to register - will be located automatically
    /// Set server settings, like base updating rate
    /// Add elements (buttons) and edit them
    /// Get html to edit as well

    controllerBase = icontrollerBase;
    _serv = new THttpServer("http:8100");
    _serv->AddLocation("mydir/", (saveLocation+"qaHists").c_str());
    _serv->SetReadOnly(kFALSE);
    _serv->SetItemField("/", "_monitoring", "1000");   // monitoring interval in ms
    _serv->SetItemField("/graphs", "_drawopt", "AP");
    _serv->SetTimer(0, kTRUE);
    _serv->SetJSROOT("https://root.cern.ch/js/latest/");
    _serv->SetJSROOT("https://jsroot.gsi.de/latest/");
    gSystem->Setenv("HTTP_NO_GZIP", "1");

    //trainingCheckGraph->SetName("trainingCheckGraph");
    trainingTargetsGraph = new TGraph();
    trainingPredictionsGraph = new TGraph();
    trainingTargetsGraph->SetName("trainingTargetsGraph");
    trainingPredictionsGraph->SetName("trainingPredictionsGraph");

    predictionGraph = new TGraph();
    predictionGraph->SetName("predictionGraph");
    predictionVsTargetGraph = new TGraph();
    predictionVsTargetGraph->SetName("predictionVsTargetGraph");
    targetsPredictions = new TGraph();
    targetsPredictions->SetName("targetsPredictions");
    for (size_t i = 0; i < controllerBase->getNodesLength(); i++){
        predictionGraphNodes[i] = new TGraph();
        predictionGraphNodes[i]->SetName(Form("predictionGraphNodes_%d",(int)i));
        targetsGtaphNodes[i] = new TGraphErrors();
        targetsGtaphNodes[i]->SetName(Form("targetsGtaphNodes%d",(int)i)); 
        targetsPredictionsNodes[i] = new TGraph();
        targetsPredictionsNodes[i]->SetName(Form("targetsPredictionsNodes%d",(int)i));
    }

    _serv->Register("graphs/subfolder", trainingTargetsGraph);
    _serv->Register("graphs/subfolder", trainingPredictionsGraph);


    cout << "setted up main cicle" << endl;
    //TApplication approot("myapp", &argc, argv);
    //httpThread->Run();
    //approot.Run();
    cout << "additional ouput for approot check" << endl;

    //int a = updateMainCicle();
}

HistsUpdater::~HistsUpdater(){

}

void HistsUpdater::updateTrainingCheckGraph(){
    int nodes = controllerBase->getNodesLength();
    ifstream fin2;
	fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCSecModNew.dat").c_str());
    std::map<int, vector<double> > targetsMap;
    vector<int> runsTargets;
    vector<double> valuesTargets;
    while (!fin2.eof()){
        int run;
        double target, targetErr;
        fin2 >> run >> target >> target >> target >> targetErr; 
        targetsMap[run].push_back(target);
        runsTargets.push_back(run);
        valuesTargets.push_back(target);
    }
    fin2.close();
    //valuesTargets = vectorFunctions::runningMeanVector(valuesTargets, 20);

    ifstream fin1;
	fin1.open((saveLocation+ "function_prediction/predicted.txt").c_str());
    vector<int> runsPredicted;
    vector<double> valuesPredicted;
    while (!fin1.eof()){
        double run;
        double prediction;
        vector<double> predictions;
        fin1 >> run;
        for (size_t i = 0; i < nodes; i ++){
            fin1 >> prediction;
            predictions.push_back(prediction);
            runsPredicted.push_back((run));
            valuesPredicted.push_back(prediction);
        }
    }
    fin1.close();

    trainingTargetsGraph->Set(0);
    for (size_t i =0; i < runsTargets.size(); i++)
        trainingTargetsGraph->SetPoint(i,(double)runsTargets[i],valuesTargets[i]);
    trainingPredictionsGraph->Set(0);
    for (size_t i =0; i < runsPredicted.size(); i++)
        trainingTargetsGraph->SetPoint(i,(double)runsPredicted[i],valuesPredicted[i]);

    vector<int> runsCompared;
    vector<double> valuesCompared;
    for (size_t ri = 0; ri < runsPredicted.size(); ri++){
        if (targetsMap[runsPredicted[ri]].size()!=0){
            runsCompared.push_back(runsPredicted[ri]);
            valuesCompared.push_back(valuesPredicted[ri]);
        }
    }

}

/// Add functions (controllerBase) that get (update) histograms
void HistsUpdater::updateHists(vector< pair<int, vector<float> > > newPredictions){
    TFile* qaHists = new TFile((saveLocation+"qaHists/qahists.root").c_str(),"UPDATE");
    if (newPredictions.size()>0){
        // update contrinuous prediction graph
        for (pair< int, vector<float> > entry: newPredictions){
            for (float pred: entry.second)
                predictionGraph->SetPoint(predictionGraph->GetN(),(double)(entry.first),(double)(pred));
        }

        TGraph *existingGraph = dynamic_cast<TGraph*>(qaHists->Get(predictionGraph->GetName()));
        if (existingGraph)
            predictionGraph->Write(predictionGraph->GetName(),TObject::kOverwrite);
        else
            predictionGraph->Write(predictionGraph->GetName());
    }

    TDirectory* dirNodesPred = dynamic_cast<TDirectory*>(qaHists->Get("nodesPred"));
    if (!dirNodesPred)
        dirNodesPred = qaHists->mkdir("nodesPred");
    dirNodesPred->cd();
    if (newPredictions.size()>0){
        // update contrinuous prediction graph
        for (pair< int, vector<float> > entry: newPredictions){
            for (size_t i = 0; i < entry.second.size(); i++)
                predictionGraphNodes[i]->SetPoint(predictionGraphNodes[i]->GetN(),(double)(entry.first),(double)(entry.second[i]));
        }
        for (size_t i = 0; i < controllerBase->getNodesLength(); i++){

            TGraph *existingGraph = dynamic_cast<TGraph*>(dirNodesPred->Get(predictionGraphNodes[i]->GetName()));
            if (predictionGraphNodes[i]->GetN() > 0){
                if (existingGraph)
                    predictionGraphNodes[i]->Write(predictionGraphNodes[i]->GetName(),TObject::kOverwrite);
                else
                    predictionGraphNodes[i]->Write(predictionGraphNodes[i]->GetName());
            }
        }
    }
    qaHists->cd();

    // update all other graphs
    //updateTrainingCheckGraph();

//    trainingTargetsGraph->Write();
//    trainingPredictionsGraph->Write();

    qaHists->Save();
    qaHists->Close();


    /// Update prediction file with newPredictions here
    ofstream fout1;
	fout1.open((saveLocation + "runsPredicted.dat").c_str(),std::ios_base::app);
    for (size_t i = 0; i < newPredictions.size(); i++){
        fout1 << newPredictions[i].first;
        for (size_t j = 0; j < newPredictions[i].second.size(); j++){
            fout1 << "  " << newPredictions[i].second[j];
        }
        fout1 << endl;
    }
    fout1.close();
}

void HistsUpdater::updateTargetGraphs(){
    TFile* qaHists = new TFile((saveLocation+"qaHists/qahists.root").c_str(),"UPDATE");
    size_t nodesSize = controllerBase->getNodesLength();
    /// Read run-mean file with targets vs run
    ifstream fin2;
	fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCSecModzxc.dat").c_str());
    std::map<int, vector< pair<double,double> > > targets;
    string line; 
    while (std::getline(fin2, line)){
        int run, sec, mod;
        double target, targetErr;
        std::istringstream iss(line);
        iss >> run >> sec >> mod >> target >> targetErr;
        targets[run].push_back(make_pair(target,targetErr));
    }
    fin2.close();


    /// Read graph with predictions? Better - file with table predicted-run
    ifstream finPred((saveLocation+"runsPredicted.dat").c_str());
    if (!finPred.good()){
        std::ofstream fileCreate((saveLocation+"runsPredicted.dat").c_str());
        fileCreate.close();
    }
    int run = 0;
    std::map<int, vector<double>> predictionMap;
    while (std::getline(finPred, line)){
        std::istringstream iss(line);
        iss >> run;
        double predValue = 0;
        vector<double> predictions;
        while (iss >> predValue) {
            predictions.push_back(predValue);
        }
        predictionMap[run] = predictions;
    }
    finPred.close();


    /// Create graphs for targetVsRun
    for (size_t i = 0; i < nodesSize; i++){
        targetsGtaphNodes[i]->Set(0);
        targetsPredictionsNodes[i]->Set(0);
    }
    targetsPredictions->Set(0);

    for (auto entry: targets){
        for (size_t i = 0; i < entry.second.size(); i++){
            targetsGtaphNodes[i]->SetPoint(targetsGtaphNodes[i]->GetN(),(double)(entry.first),(double)(entry.second[i].first));
            targetsGtaphNodes[i]->SetPointError(targetsGtaphNodes[i]->GetN()-1,0.0,(double)(entry.second[i].second));
        }
    }

    /// Target-prediction graph
    for (auto entryTarget: targets){
        int run = entryTarget.first;
        auto it  = predictionMap.find(run);
        //cout << "run for target is " << run << endl;
        if (it == predictionMap.end())
            continue;
        //cout << "xbc" << endl;
        vector<double> predictionValues = predictionMap[run];
        vector< pair<double,double> > targetValues = entryTarget.second;
        //cout << predictionValues.size() << "    " << targetValues.size() << endl;
        if (predictionValues.size() == targetValues.size()){
            for (size_t i = 0; i < targetValues.size(); i++){
                //cout << "gsdasdsf " << targetsPredictionsNodes[i]->GetN() << endl;
                targetsPredictionsNodes[i]->SetPoint(targetsPredictionsNodes[i]->GetN(),(double)run,(predictionValues[i]-targetValues[i].first)/targetValues[i].second);
                targetsPredictions->SetPoint(targetsPredictions->GetN(),(double)run,(predictionValues[i]-targetValues[i].first)/targetValues[i].second);
            }
        }
    }

    TDirectory* dirNodesPred = dynamic_cast<TDirectory*>(qaHists->Get("nodesTarget"));
    if (!dirNodesPred)
        dirNodesPred = qaHists->mkdir("nodesTarget");
    dirNodesPred->cd();
    for (size_t i = 0; i < nodesSize; i++){
        //cout << targetsGtaphNodes[i]->GetN() << "   " << targetsPredictionsNodes[i]->GetN() << endl;
        TGraph *existingGraph = dynamic_cast<TGraph*>(dirNodesPred->Get(targetsGtaphNodes[i]->GetName()));
        if (existingGraph){
            if (targetsGtaphNodes[i]->GetN() > 0)
                targetsGtaphNodes[i]->Write(targetsGtaphNodes[i]->GetName(),TObject::kOverwrite);
        }
        else{
            targetsGtaphNodes[i]->Write(targetsGtaphNodes[i]->GetName());
        }

        TGraph *existingGraph1 = dynamic_cast<TGraph*>(dirNodesPred->Get(targetsPredictionsNodes[i]->GetName()));
        if (existingGraph1){
            if (targetsPredictionsNodes[i]->GetN() > 0){
                targetsPredictionsNodes[i]->Write(targetsPredictionsNodes[i]->GetName(),TObject::kOverwrite);
            }
        }
        else{
            if (targetsPredictionsNodes[i]->GetN() > 0){
                targetsPredictionsNodes[i]->Write(targetsPredictionsNodes[i]->GetName());
            }
        }
    }
    qaHists->cd();
    TGraph *existingGraph = dynamic_cast<TGraph*>(qaHists->Get(targetsPredictions->GetName()));
    if (targetsPredictions->GetN() > 0){
        if (existingGraph)
            targetsPredictions->Write(targetsPredictions->GetName(),TObject::kOverwrite);
        else
            targetsPredictions->Write(targetsPredictions->GetName());
    }

    qaHists->Save();
    qaHists->Close();
}

vector<int> HistsUpdater::loadProcessedrunlist(int run1, int run2){
    vector<int> result;
    ifstream fin1((saveLocation+"runlistProcessed.dat").c_str());
    if (!fin1.good()){
        std::ofstream fileCreate((saveLocation+"runlistProcessed.dat").c_str());
        fileCreate.close();
        return result;
    }
    int run = 0;
    while (fin1 >> run){
        if (run <= run1 || run >= run2)
            continue;
        result.push_back(run);
    }
    fin1.close();
    return result;
}

void HistsUpdater::updateProcessedRunList(vector<int> newRunNumbers){
    vector<int> oldProcessedRuns = loadProcessedrunlist(0,1e9);
    ofstream fout1;
	fout1.open((saveLocation + "runlistProcessed.dat").c_str(),std::ios_base::app);
    for (size_t i = 0; i < newRunNumbers.size(); i++){
        auto it = std::find(oldProcessedRuns.begin(), oldProcessedRuns.end(), newRunNumbers[i]);
        if (it == oldProcessedRuns.end())
            fout1 << newRunNumbers[i] << endl;
    }
    fout1.close();
}

int HistsUpdater::updateMainCicle(){


    /// Load here last used runlist, not current!!! So that for the runs which were taken, but not used here (programm off), in the main loop will be done stuff
    vector<int> processedRunList = loadProcessedrunlist(0,1e9);
    int maxRun = 0; int maxRunIndex = 0;
    int lastPredRun = 0; int lastPredRunIndex = -1;
    if (processedRunList.size() > 0){
        maxRun = processedRunList[processedRunList.size()-1];
        maxRunIndex = processedRunList.size()-1;
        if (processedRunList.size() > 1){
            lastPredRun = processedRunList[processedRunList.size()-2];
            lastPredRunIndex = processedRunList.size()-2;
        }
    }
    cout << "max run for continuous prediction is set to " << maxRun << endl;

    int retrainCounter = 0;
    int start_counter = 0;

    while(1){
        _serv->ProcessRequests();

        /// Check if new settings appear in a config file to change controllerBase
        controllerBase->checkNewSettingsConfig();
        //cout << "new settings checked" << endl;


        /// Check low amount of new runs
        vector<int> currentRunList = dateRunF::loadrunlist(0,1e9);
        if (currentRunList.size() == 0){
            maxRun = 0;
            maxRunIndex = 0;
            lastPredRunIndex = -1;
            continue;
        }
        if (currentRunList.size() == 1){
            //cout << "setting maxRun to " << currentRunList[0] << endl;
            maxRun = currentRunList[0];
            maxRunIndex = 0;
            lastPredRunIndex = -1;
            continue;
        }
        if (currentRunList.size() >= 2 && maxRun == 0){
            //cout << "setting maxRun to " << currentRunList[0] << endl;
            maxRun = currentRunList[0];
            maxRunIndex = 0;
            lastPredRunIndex = -1;
        }
        if (start_counter == 0 && currentRunList.size() >= 2 && processedRunList.size() > 0){
            /// IF we only started, there were something before in globals and processed runs, then
            /// As processed run save only processed (except maximum) -> we need to get 2 equal runlist if no new global run apear
            /// Thus, we need to append processed run list (manifested in maxRun) with the next entry from the new list (that could change or not)
            auto it1 = std::find(currentRunList.begin(), currentRunList.end(), maxRun);
            maxRun = *(it1+1);
            start_counter = 1;
        }
        //maxRun = currentRunList[currentRunList.size()-4];
        //maxRun = currentRunList[0];
        //maxRunIndex = 0;
        //lastPredRunIndex = -1;
        //lastPredRunIndex = currentRunList.size()-5;


        /// Finding new runs based on the maxRun from lastPredicted or from the last loop entry
        int maxRunNew = currentRunList[currentRunList.size()-1];
        auto it1 = std::find(currentRunList.begin(), currentRunList.end(), maxRun);
        auto it2 = std::find(currentRunList.begin(), currentRunList.end(), maxRunNew);
        vector<int> newRunNumbers = {};
        //cout << maxRun << " " << maxRunIndex << "   " << lastPredRunIndex << "  " << std::distance(currentRunList.begin(), it1) << "   " << std::distance(currentRunList.begin(), it2) << endl;
        if (it1 != currentRunList.end() && it2 != currentRunList.end()) {
            if (it1 != it2){
                std::vector<int> subVector(it1, it2);  // including previous maxRun, but not including a new one
                newRunNumbers = subVector;   
            }
        }
        cout << "new runs number = " << newRunNumbers.size() << endl;

        
        /// Process new runs
        vector< pair<int, vector<float> > > newPredictions;
        if (controllerBase->serverData->getContinuousPredictionRunning()){
            newPredictions.clear();
            controllerBase->serverData->setCurrentRunIndex(lastPredRunIndex+1);
            for (int newRun: newRunNumbers){
                /// For each new file get a prediction and update hists with these new predictions (for continuous predictions)
                /// Current run will be automatically moved through runlist by moveForward function
                vector<float> newPrediction = controllerBase->moveForwardCurrentNNInput(); //number of nodes (cells) for the current run
                cout << newPrediction.size() << endl;
                newPredictions.push_back(make_pair(newRun,newPrediction));
                cout << "prediction is made for run " << newRun << endl;
                //cin.get();

                retrainCounter += 1;
                if (retrainCounter >= 10){
                    controllerBase->writeData();
                    updateTargetGraphs();
                    retrainCounter = 0;
                    //int trainRunsAmount = controllerBase->neuralNetwork->remakeInputDataset(false);
                    //cout << "trainRunsAmount = " << trainRunsAmount << endl;
                    //if (trainRunsAmount >= 50){
                    //    std::system("xterm -e python3 /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/trainClassificationTemplate.py > ~/outputTrain.log 2>&1 &");
                    //}
                }

            }

            //std::this_thread::sleep_for(std::chrono::seconds(10));
        }

        else {
            //std::this_thread::sleep_for(std::chrono::seconds(1));
            /// append epics db table, as for these scipped runs, targets are anyway retrieved, useful for training
            for (size_t i = 0; i < newRunNumbers.size(); i++){
                int startRuntt = newRunNumbers[i];
                cout << "getting epics data with startrun = " << startRuntt << endl;
                vector<double> dbData = controllerBase->epicsManager->getDBdata(startRuntt, newRunNumbers[i]);
                cout << "filling epics table with startrun = " << startRuntt << endl;
                controllerBase->epicsManager->appendDBTable("app", startRuntt, dbData);
                retrainCounter += 1;
            }
        }

        updateProcessedRunList(newRunNumbers);
        updateHists(newPredictions);

        //lastPredRun = currentRunList[currentRunList.size()-2];
        lastPredRunIndex = currentRunList.size()-2;
        maxRun = currentRunList[currentRunList.size()-1];
        maxRunIndex = currentRunList.size()-1;

        //retrainCounter += newRunNumbers.size();
        //cout << retrainCounter << endl;
        if (retrainCounter >= 500){
            /// Update a graph/s with targets? To be able to compare to previously made predictions
            updateTargetGraphs();

            /// Launch retraining in xterm
            retrainCounter = 0;
            int trainRunsAmount = controllerBase->neuralNetwork->remakeInputDataset(false);
            cout << "trainRunsAmount = " << trainRunsAmount << endl;
            if (trainRunsAmount >= 50){
                std::system("xterm -e python3 /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/trainClassificationTemplate.py > ~/outputTrain.log 2>&1 &");
            }
            //scin.get();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return 1;
}