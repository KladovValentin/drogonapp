#include "../include/continuousPredictor.h"

#include "../include/functions.h"
#include "../include/constants.h"

using namespace std;

ContinuousPredictor::ContinuousPredictor(std::shared_ptr<ControllerBase>& icontrollerBase){
    cout << "starting a prediction process.." << endl;
    controllerBase = icontrollerBase;
}

ContinuousPredictor::~ContinuousPredictor(){}

void ContinuousPredictor::start(){

    cout << controllerBase->serverData << endl;

    int portt = controllerBase->serverData->getDbListeningPort();
    cout << "port predictor " << portt << endl;
    controllerBase->serverData->setDbListeningPort(portt+1);
    cout << controllerBase->serverData->getDbListeningPort() << endl;
    while(0){
        //cout << "starting a prediction process.." << endl;
        //cout << controllerBase->serverData->getContinuousPredictionRunning() << endl;
        while (controllerBase->serverData->getContinuousPredictionRunning()){
            cout << "aaa" << endl;
            ofstream fout;
            fout.open((saveLocation + "testOut.txt").c_str(),std::ios::app);
            fout << "predicted" << endl;
            fout.close();
            cout << "predicting..." << endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
}