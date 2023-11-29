#include "MainController.h"


#include "../include/functions.h"
#include "../include/constants.h"

using namespace std;
//using namespace drogon;



/// @brief  SETUP MAIN CONTROLLER - LAUNCH MAIN SYSTEMS
MainController::MainController() {
    std::cout << "a" << std::endl;

    serverData = new ServerData();
    triggerManager = new TriggerDataManager();
    epicsManager = new EpicsDBManager(serverData->getDbListeningPort());
    neuralNetwork = new NeuralNetwork();

    cout << "b" << endl;


    triggerManager->changeHistsLocation(serverData->getTrHistsLoc());
    triggerManager->changeChannels(serverData->getTrChannels());
    epicsManager->changeChannelNames(serverData->getDbShape(), serverData->getDbChannels());
    
    controllerBase = new ControllerBase(triggerManager, epicsManager, neuralNetwork, serverData); 

    //epicsManager->makeTableWithEpicsData("new", 444140006, 1e9);
    //epicsManager->makeTableWithEpicsData("new", 444472254, 444533980);
    //neuralNetwork->remakeInputDataset();
    //neuralNetwork->retrainModel();
    //controllerBase->compareTargetPredictionFromTraining();

    //serverData->readSettingsJSON();
    //serverData->writeSettingsJSON();

}


/*void MainController::getTrHistsLoc(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback){
    string histsLocation = serverData->getTrHistsLoc();
    Json::Value ret;
    ret["histsLocation"].append(histsLocation);
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}

void MainController::setTrHistsLoc(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, string newTrHistsLoc){
    serverData->setTrHistsLoc(newTrHistsLoc);
    Json::Value ret;
    ret["result"]="ok";
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}

void MainController::getEpicsChannels(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback){
    vector<string> channelsdb = serverData->getDbChannels();
    Json::Value ret;
    for (const auto& value : channelsdb)
        ret["channels"].append(value);
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}

void MainController::setEpicsChannels(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, vector<string> newDBChannels){
    serverData->setDbChannels(newDBChannels);
    Json::Value ret;
    ret["result"]="ok";
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}

void MainController::getTriggerChannels(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback){
    vector<int> channelstr = serverData->getTrChannels();
    Json::Value ret;
    for (const auto& value : channelstr)
        ret["channels"].append(value);
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}

void MainController::setTriggerChannels(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, vector<int> newTrChannels){
    serverData->setTrChannels(newTrChannels);
    Json::Value ret;
    ret["result"]="ok";
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}*/


void MainController::setSettings(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, string newTrHistsLoc, vector<int> newTrChannels, vector<string> newDBChannels){
    serverData->setTrHistsLoc(newTrHistsLoc);
    //serverData->setDbChannels(newDBChannels);
    serverData->setTrChannels(newTrChannels);
    Json::Value ret;
    ret["result"]="ok";
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}
void MainController::getSettings(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback){
    string histsLocation = serverData->getTrHistsLoc();
    vector< pair <vector<int>, string> > channelsdb = serverData->getDbChannels();
    vector<int> channelstr = serverData->getTrChannels();
    Json::Value ret;
    ret["histsLocation"] = histsLocation;
    for (const auto& value : channelstr)
        ret["channelsTr"].append(value);
    for (const auto& value : channelsdb)
        ret["channelsDb"].append(value.second);
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}


void MainController::readEpicsTable(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, int startRun, int endRun){
    epicsManager->makeTableWithEpicsData("new", startRun, endRun);
    Json::Value ret;
    ret["result"]="ok";
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}
void MainController::readTriggTable(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, int startRun, int endRun, bool formLists){
    if (formLists)
        triggerManager->make_lists();
    triggerManager->makeTableWithTriggerData();
    Json::Value ret;
    ret["result"]="ok";
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}
void MainController::retrainModel(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, bool remakeInput){
    if (remakeInput)
        neuralNetwork->remakeInputDataset();
    neuralNetwork->retrainModel();
    Json::Value ret;
    ret["result"]="ok";
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}


/// @brief GET SINGLE PREDICTION CONSTRUCTING THE WHOLE LSTM INPUT USING RUNLIST.
void MainController::getPrediction(const HttpRequestPtr &req,
               std::function<void (const HttpResponsePtr &)> &&callback,
               int run1){

    //float prediction = controllerBase->getPrediction(run1, run2);
    float prediction = neuralNetwork->getPrediction(controllerBase->makeNNInputTensor(run1));


    /*ifstream fin2;
	fin2.open((saveLocation + "info_tables/run-mean_dEdxMDC1.dat").c_str());
    std::map<int,double> targets;
    while (!fin2.eof()){
        int runt;
        double target, targetErr;
        fin2 >> runt >> target >> targetErr; 
        targets[runt] = target;
    }
    fin2.close();

    cout << "target is " << targets[run] << endl;*/

    Json::Value ret;
    ret["result"]="ok";
    ret["prediction"]=prediction;
    //ret["token"]=drogon::utils::getUuid();
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}

void MainController::getContinuousPrediction(const HttpRequestPtr &req,
               std::function<void (const HttpResponsePtr &)> &&callback){
    
    
    float prediction = controllerBase->moveForwardCurrentNNInput();
    int currentRun = serverData->getCurrentRun();
    

    ifstream fin2;
	fin2.open((saveLocation + "info_tables/run-run-mean_dEdxMDCAllNew.dat").c_str());
    std::map<int,double> targets;
    while (!fin2.eof()){
        int run;
        double target, targetErr;
        fin2 >> run >> target >> targetErr; 
        targets[run] = target;
    }
    fin2.close();

    Json::Value ret;
    ret["result"]="ok";
    ret["run"]=currentRun;
    ret["prediction"]=prediction;
    ret["target"]=targets[currentRun];
    cout << targets[currentRun] << endl;
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);

}


void MainController::getVectorsForTrainingCheck(const HttpRequestPtr &req,
               std::function<void (const HttpResponsePtr &)> &&callback){
    
    ifstream fin2;
	fin2.open((saveLocation + "info_tables/run-mean_dEdxMDCAllNew.dat").c_str());
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
    fin2.close();
    valuesTargets = vectorFunctions::runningMeanVector(valuesTargets, 20);

    ifstream fin1;
	fin1.open((saveLocation+ "function_prediction/predicted.txt").c_str());
    vector<int> runsPredicted;
    vector<double> valuesPredicted;
    while (!fin1.eof()){
        double run;
        double prediction;
        fin1 >> run >> prediction;
        runsPredicted.push_back((run));
        valuesPredicted.push_back(prediction);
    }
    fin1.close();


    Json::Value ret;
    ret["result"]="ok";
    for (const auto& value : runsTargets) {
        ret["runsTargets"].append(value);
    }
    for (const auto& value : valuesTargets) {
        ret["valuesTargets"].append(value);
    }
    for (const auto& value : runsPredicted) {
        ret["runsPredicted"].append(value);
    }
    for (const auto& value : valuesPredicted) {
        ret["valuesPredicted"].append(value);
    }
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}


void MainController::getPredictedList(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback){

    map<int, float> predictedList = serverData->getPredictedValues();
    Json::Value ret;
    for (const auto& value : predictedList)
        ret[to_string(value.first)] = value.second;
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
}


void MainController::stopServer(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback){
    Json::Value ret;
    ret["result"] = "ok";
    auto resp=HttpResponse::newHttpJsonResponse(ret);
    callback(resp);
    controllerBase->writeData();
}


