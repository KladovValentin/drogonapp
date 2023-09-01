#include "../include/serverData.h"

#include "../include/constants.h"


#include "../include/functions.h"
#include "../include/constants.h"

using namespace std;

ServerData::ServerData(){
    readSettings();
    readPredictedValues();
    runBorders = dateRunF::loadrunlist(0, 1e10);
    setCurrentRunIndex(2509);
}

ServerData::~ServerData(){
    writeSettings();
    writePredictedValues();
}

void ServerData::readSettings(){
    ifstream fin0;
	fin0.open((saveLocation + "globalSettings.txt").c_str());
    string line;
    std::getline(fin0, triggerHistsPath);
    std::getline(fin0, line);
    std::stringstream ss(line);
    int num;
    trigChannels.clear();
    while (ss >> num) {
        trigChannels.push_back(num);
    }

    std::getline(fin0, line);
    if (line == "epics channels"){
        dbNames.clear();
        while (1){
            std::getline(fin0, line);
            if (line == "listening port")
                break;
            std::stringstream ss1(line);
            string dbname;
            while (ss1 >> dbname) {
                dbNames.push_back(dbname);
                //cout << dbname << endl;
            }
        }
    }
    for(string dbname: dbNames){
        cout << dbname << " ";
    }
    cout << endl;

    fin0 >> epicsDBPort;
    fin0.close();
}

void ServerData::writeSettings(){
    ofstream fout;
    fout.open((saveLocation + "globalSettings.txt").c_str());
    fout << triggerHistsPath << endl;
    
    for (int x: trigChannels)
        fout << x << "  ";
    fout << endl;

    for (string x: dbNames)
        fout << x << "  ";
    fout << endl;

    fout << epicsDBPort << endl;
    fout.close();
}

void ServerData::readPredictedValues(){
    ifstream fin1;
	fin1.open((saveLocation+"predictedValues.dat").c_str());
    while (fin1.get() != EOF){
        int run = 0;
        float predictedValue = 0;
        fin1 >> run >> predictedValue;
        predictedValues[run] = predictedValue;
    }
    fin1.close();
}

void ServerData::writePredictedValues(){
    ofstream fout;
    fout.open((saveLocation + "predictedValues.txt").c_str());
    for (auto x: predictedValues)
        fout << x.first << "    " << x.second << endl;
    fout.close();
}