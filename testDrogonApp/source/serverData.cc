#include "../include/serverData.h"

#include "../include/constants.h"


#include "../include/functions.h"
#include "../include/constants.h"

using namespace std;

ServerData::ServerData(){
    readSettingsJSON();
    readPredictedValues();
    runBorders = dateRunF::loadrunlist(0, 1e10);
    setCurrentRunIndex(2509);
}

ServerData::~ServerData(){
    //readSettingsJSON();
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
        
        std::getline(fin0, line);
        std::stringstream ssShape(line);
        size_t shapeEl;
        dbShape.clear();
        while (ssShape >> shapeEl)
            dbShape.push_back(shapeEl);
        
        int n_dim = dbShape.size()-1; // dimensions of spacial parametrization (excluding channels dim)
        
        dbNames.clear();
        while (1){
            std::getline(fin0, line);
            if (line == "listening port")
                break;
            std::stringstream ss1(line);
            string multi, dbname;
            int multiI = 0;
            vector<int> multiVect;
            for (size_t i = 0; i < n_dim; i++){
                ss1 >> multiI;
                multiVect.push_back(multiI);
            }
            ss1 >> dbname; 
            //ss1 >> multi >> dbname; 
            //if (multi != "0" && multi != "1"){
            //    continue;
            //}
            //if (((TString)dbname).Contains("%")){
            //    dbname = (string)(Form(dbname.c_str(),1));
            //}
            dbNames.push_back(make_pair(multiVect, dbname));
            cout << dbname << endl;
        }
    }
    for(pair <vector<int>, string> dbname: dbNames){
        cout << dbname.second << " ";
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

    for (pair <vector<int>,string> x: dbNames)
        fout << x.second << "  ";
    fout << endl;

    fout << epicsDBPort << endl;
    fout.close();
}

void ServerData::writeSettingsJSON(){
    Json::Value jsonObj;
    jsonObj["triggerHistsFile"] = triggerHistsPath;

    for (int& trChan: trigChannels)
        jsonObj["triggerChannels"].append(trChan);

    for (size_t& i: dbShape)
        jsonObj["shape"].append(i);

    for(pair <vector<int>, string> dbname: dbNames){
        Json::Value a;
        for (int& i: dbname.first)
            a["parameters"].append(i);
        a["name"] = dbname.second;
        jsonObj["epics_channels"].append(a);
    }

    jsonObj["port"] = epicsDBPort;

    std::ofstream outFile((saveLocation + "globalSettings.json").c_str());
    outFile << std::setw(4) << jsonObj << std::endl;
    outFile.close();
}

void ServerData::readSettingsJSON(){
    Json::Value jsonObj;
    std::ifstream inFile((saveLocation + "globalSettings.json").c_str());
    inFile >> jsonObj;
    inFile.close();

    trigChannels.clear();
    dbNames.clear();
    dbShape.clear();

    triggerHistsPath = jsonObj["triggerHistsFile"].asString();
    for (const auto& entry : jsonObj["triggerChannels"])
        trigChannels.emplace_back(entry.asInt());

    for (const auto& entry : jsonObj["shape"])
        dbShape.emplace_back(entry.asInt());
    
    epicsDBPort = jsonObj["port"].asInt();

    for (const auto& entry : jsonObj["epics_channels"]) {
        vector<int> a;
        for (const auto& entry1 : entry["parameters"])
            a.push_back(entry1.asInt());
        dbNames.emplace_back(make_pair(a, entry["name"].asString()));
    }
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