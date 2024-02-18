#include "include/functions.h"
#include "include/workWithTrigger.h"
#include "include/workWithDB.h"
#include "include/neuralNetwork.h"

#include "include/controllerBase.h"
#include "include/histsUpdater.h"
#include "include/continuousPredictor.h"
#include "include/serverData.h"
#include "include/constants.h"

#include "controllers/TestCtrl.h"
#include "controllers/MainController.h"

#include "include/globals.h"
#include "include/serviceLocator.h"

#include <Python.h>

#include <drogon/drogon.h>
#include <drogon/HttpAppFramework.h>
#include <drogon/plugins/Plugin.h>
//#include <drogon/plugins/CorsPlugin.h>

#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>
#include <thread>

#include "TStyle.h"
#include "TDatime.h"
#include "TApplication.h"
#include <TSystem.h>
#include <TROOT.h>
#include "THttpServer.h"
#include "TThread.h"

using namespace std;
using namespace dateRunF;
using namespace vectorFunctions;

using namespace drogon;


//void startMainServer(const std::shared_ptr<MainController>& icontrollerBase){
void startMainServer(){
    app().loadConfigFile("./config.json");
    //app().addListener("0.0.0.0",8080);
    //app().registerController<MainController>(icontrollerBase);
    app().registerPostHandlingAdvice(
        [](const drogon::HttpRequestPtr &req, const drogon::HttpResponsePtr &resp) {
            resp->addHeader("Access-Control-Allow-Origin", "*");
        });
    app().run();
}


int main(int argc, char** argv) {


    //ControllerBase* controllerBase = new ControllerBase();
    //ControllerBase* controllerBase = new ControllerBase();
    //auto mainController = std::make_shared<MainController>(controllerBase);

    /// Ideally, change to normal class object. Initialize normally once, and forward a copy to the child drogon process.
    /// It only needs to save the data via serverdata copy. And the main copy of the controllerBase in the parent process will read the data from file regulary
    std::shared_ptr<ControllerBase> controllerBase = std::make_shared<ControllerBase>();
    ServiceLocator::initialize(controllerBase);

    //controllerBase->compareTargetPredictionFromTraining();
    //epicsManager->makeTableWithEpicsData("new", 443670000, 1e9);
    //neuralNetwork->retrainModel();
    //float a = controllerBase->getPrediction(446737374, 446737470);
    //controllerBase->drawManyPredictions();
    //controllerBase->epicsManager->makeTableWithEpicsData("app", 446077834, 1e9);
    //controllerBase->neuralNetwork->remakeInputDataset(false);

    vector<float> inp = {-0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, 1.15, 1.19213, 1.28335, 1.69697, 1.50619, 1.43025, 0.114225, 1.2106, 0.557263, 0, 0, -0.577436, 1.20882, 1.2026, 1.11423, 0.922802, 1.2895, 1.31063, 1.86656, 1.04046, 1.25881, -0.913003, 1.24629, 1.23368, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, 1.15, 1.19213, 1.28335, 1.69697, 1.50619, 1.43025, 0.114225, 1.2106, 0.557263, 0, 0, -0.577436, 1.20882, 1.2026, 1.11423, 0.922802, 1.2895, 1.31063, 1.86656, 1.04046, 1.25881, -0.913003, 1.24629, 1.23368, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, 1.15, 1.19213, 1.28335, 1.69697, 1.50619, 1.43025, 0.114225, 1.2106, 0.557263, 0, 0, -0.577436, 1.20882, 1.2026, 1.11423, 0.922802, 1.2895, 1.31063, 1.86656, 1.04046, 1.25881, -0.913003, 1.24629, 1.23368, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, 1.01647, 1.22912, 1.08716, 1.45236, 1.41305, 1.36161, 0.124045, 1.14602, 0.576243, 0, 0, -0.649008, 1.21491, 1.2118, 1.10974, 0.907692, 1.3059, 1.28224, 1.83982, 1.02719, 1.25147, -0.889066, 1.2379, 1.2288, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, 1.11889, 1.24665, 1.2449, 1.48192, 1.58487, 1.47091, 0.0309097, 1.2089, 0.552984, 0, 0, -0.623491, 1.21062, 1.21796, 1.11977, 0.892438, 1.29838, 1.28405, 1.823, 1.0153, 1.25138, -0.945185, 1.2379, 1.2335, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924};
    //auto start = std::chrono::high_resolution_clock::now();
    Py_Initialize();
    //vector<float> result = controllerBase->neuralNetwork->getRawPredictionPython(inp);
    //std::this_thread::sleep_for(std::chrono::seconds(5));
    //vector<float> result1 = controllerBase->neuralNetwork->getRawPredictionPython(inp);
    //long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    //cout << "TIME PASSED    " << elapsed << endl;
    //gSystem->Exec("python3 /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/makePrediction.py");
    //gSystem->Exec("python3 /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/makePrediction.py");

    /// Start a drogon control loop to update settings etc
    pid_t pid = fork();
    if (pid == 0) {
        // Child process: start Drogon server
        startMainServer();
        exit(0);
    }


    /// Start a process to get new offline calibration and append an offline calibration table for training
    //std::system("g++ -o fetchNewCalibrations fetchNewCalibrations.cc");
    //std::system("xterm -e ~/gsiWorkFiles/fetchMDCcalibrations/fetchNewCalibrations > ~/output.log 2>&1 &");


    /// Start a process to get new files and thus new runs, take directory from config, take start run from existing runlist, check config regularly
    //std::system("g++ -o fetchNewRuns fetchNewRuns.cc");
    //std::system("xterm -e ~/gsiWorkFiles/fetchMDCcalibrations/fetchNewRuns > ~/outputRun.log 2>&1 &");   // Or use xterm -e to launch in a new terminal


    /// Start a main loop, where periodically new runs are checked from a runlist and calibrations are predicted
    /// Online predictions: Periodically start composition of a training table and launch training in a separate temporary process (xterm, no waiting)
    /// Process httpserver requests, periodically update control graphs and histograms (save them to the file?)
    /// New settings: Check new settings from config periodically
    /// Compose input from Epics: When cont pred - append when getting input for prediction; when no - just get new for new runs
    /// Updating canvases: Canvases can be stored to a .root file, visible from thttpserver. File should be defined in constants (changable?).
    HistsUpdater* histUpdater = new HistsUpdater(argc, argv, controllerBase);
    int hUpResult = histUpdater->updateMainCicle();




    cout << "a" << endl;
    //approot.Run();
    //app().run();

    waitpid(pid, nullptr, 0);
    Py_Finalize();



    //drogon::app().run();


    //serverData->writeSettings();


    return EXIT_SUCCESS;
}
