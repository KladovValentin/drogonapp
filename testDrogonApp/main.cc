#include "include/functions.h"
#include "include/workWithTrigger.h"
#include "include/workWithDB.h"
#include "include/neuralNetwork.h"

#include "include/controllerBase.h"
#include "include/serverData.h"

#include "controllers/TestCtrl.h"
#include "controllers/MainController.h"

#include <drogon/drogon.h>
#include <drogon/HttpAppFramework.h>
#include <drogon/plugins/Plugin.h>
//#include <drogon/plugins/CorsPlugin.h>

#include <sstream>
#include <cstdlib>

#include "TStyle.h"
#include "TDatime.h"
#include <TApplication.h>
#include <TSystem.h>
#include <TROOT.h>

using namespace std;
using namespace dateRunF;
using namespace vectorFunctions;

using namespace drogon;


int main(int argc, char** argv) {


    /*ServerData* serverData = new ServerData();
    TriggerDataManager* triggerManager = new TriggerDataManager();
    EpicsDBManager* epicsManager = new EpicsDBManager(serverData->getDbListeningPort());
    NeuralNetwork* neuralNetwork = new NeuralNetwork();


    triggerManager->changeHistsLocation(serverData->getTrHistsLoc());
    triggerManager->changeChannels(serverData->getTrChannels());
    epicsManager->changeChannelNames(serverData->getDbChannels());

    ControllerBase* controllerBase = new ControllerBase(triggerManager, epicsManager, neuralNetwork, serverData); */


    //controllerBase->compareTargetPredictionFromTraining();

    //epicsManager->makeTableWithEpicsData("new", 443670000, 1e9);

    //neuralNetwork->retrainModel();

    //float a = controllerBase->getPrediction(446737374, 446737470);
    
    //controllerBase->drawManyPredictions();

    app().loadConfigFile("./config.json");

    // Enable the CorsMiddleware
    //drogon::app().registerMiddleware<drogon::CorsMiddleware>();

    TApplication approot("myapp", &argc, argv);


    //auto ctrlPtr = std::make_shared<CustomCtrl>("Hi");
    //app().registerController(ctrlPtr);

    app().registerPostHandlingAdvice(
        [](const drogon::HttpRequestPtr &req, const drogon::HttpResponsePtr &resp) {
            resp->addHeader("Access-Control-Allow-Origin", "*");
        });

    cout << "a" << endl;
    //approot.Run();
    app().run();



    //drogon::app().run();


    //serverData->writeSettings();


    return EXIT_SUCCESS;
}
