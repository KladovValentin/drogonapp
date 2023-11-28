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
    //writeSettings();
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