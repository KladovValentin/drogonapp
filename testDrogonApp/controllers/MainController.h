#include <drogon/HttpController.h>

#include "../include/functions.h"
#include "../include/workWithTrigger.h"
#include "../include/workWithDB.h"
#include "../include/neuralNetwork.h"
#include "../include/controllerBase.h"

#include <stdio.h>
#include <string>

using namespace drogon;

class MainController : public drogon::HttpController<MainController>
{
private:
    ServerData* serverData;
    TriggerDataManager* triggerManager;
    EpicsDBManager* epicsManager;
    NeuralNetwork* neuralNetwork;
    ControllerBase* controllerBase;

public:

    METHOD_LIST_BEGIN
        /*METHOD_ADD(MainController::getTrHistsLoc, "/getTrHistsLoc", Get);
        METHOD_ADD(MainController::setTrHistsLoc, "/setTrHistsLoc?newTrHistsLoc={1}", Get);
        METHOD_ADD(MainController::getEpicsChannels, "/getEpicsChannels", Get);
        METHOD_ADD(MainController::setEpicsChannels, "/setEpicsChannels?newDBChannels={1}", Get);
        METHOD_ADD(MainController::getTriggerChannels, "/getTriggerChannels", Get);
        METHOD_ADD(MainController::setTriggerChannels, "/setTriggerChannels?newTrChannels={1}", Get);*/

        METHOD_ADD(MainController::getSettings, "/getSettings", Get);
        METHOD_ADD(MainController::setSettings, "/setSettings?newTrHistsLoc={1},newTrChannels={2},newDBChannels={3}", Get);

        METHOD_ADD(MainController::readEpicsTable, "/readEpicsTable?startRun={1},endRun={2}", Get);
        METHOD_ADD(MainController::readTriggTable, "/readTriggTable?startRun={1},endRun={2},formLists={3}", Get);
        METHOD_ADD(MainController::retrainModel, "/retrainModel?remakeInput={1}", Get);

        METHOD_ADD(MainController::getPrediction, "/getPrediction?run1={1}", Get);
        METHOD_ADD(MainController::getContinuousPrediction, "/getNextContinuousPrediction", Get);
        METHOD_ADD(MainController::getVectorsForTrainingCheck, "/getVectorsForTrainingCheck", Post);
        METHOD_ADD(MainController::getPredictedList, "/getPredictedList", Get);

        METHOD_ADD(MainController::stopServer, "/stopServer", Get);

    METHOD_LIST_END


    MainController();

    /*void MainController::getTrHistsLoc(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback);
    void MainController::setTrHistsLoc(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, string newTrHistsLoc);
    void MainController::getEpicsChannels(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback);
    void MainController::setEpicsChannels(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, vector<string> newDBChannels);
    void MainController::getTriggerChannels(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback);
    void MainController::setTriggerChannels(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, vector<int> newTrChannels);
    */

    void setSettings(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, string newTrHistsLoc, vector<int> newTrChannels, vector<string> newDBChannels);
    void getSettings(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback);

    void readEpicsTable(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, int startRun, int endRun);
    void readTriggTable(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, int startRun, int endRun, bool formLists);
    void retrainModel(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, bool remakeInput);


    void getPrediction(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback, int run1);   

    void getContinuousPrediction(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback);

    void getVectorsForTrainingCheck(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback);

    void getPredictedList(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback);

    void stopServer(const HttpRequestPtr &req, std::function<void (const HttpResponsePtr &)> &&callback);

};

