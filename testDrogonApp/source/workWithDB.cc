#include "../include/workWithDB.h"

#include "../include/functions.h"
#include "../include/constants.h"

using namespace std;

using namespace dateRunF;

EpicsDBManager::EpicsDBManager(int port){
    channel_names = {""};
    //channel_ids = {"8564"};
    listeningPort = port; // 3335

    // check if port is listening already. If not - make it listen 
    TString setupListening = (gSystem->GetFromPipe(("netstat -tulpn | grep '"+to_string(listeningPort)+"' | grep 'LISTEN'").c_str()));
    //cout << setupListening << endl;
    if (!setupListening.Contains("LISTEN"))
    {
        gSystem->Exec(("ssh -i ~/.ssh/id_rsa -J vkladov@lxlogin.gsi.de -L "+to_string(listeningPort)+":lxhaddcs10p:5432 vkladov@lxhaddcs11 -fN").c_str());
    }
    
    // Connect to the database
    pqxxConnection = new pqxx::connection("user=report password=$report host=127.0.0.1 port="+to_string(listeningPort)+" dbname=archive target_session_attrs=any");
    if (!pqxxConnection->is_open()) 
    {
        cerr << "Failed to establish connection" << endl;
    }

    // Convert names to ids
    //remakeChannelIds();
    
}
EpicsDBManager::~EpicsDBManager(){}


void EpicsDBManager::remakeChannelIds(){
    channel_ids.clear();
    for (string name: channel_names){
        //cout << name << endl;
        pqxx::work w(*pqxxConnection);
        pqxx::result rows = w.exec("SELECT channel_id FROM channel WHERE name = '"+name+"'");
        cout << "channel " << name << " found! " << rows.begin()["channel_id"].as<int>() << endl;
        if (rows.size() < 1){
            cout << "channel " << name << " not found!" << endl;
            channel_ids.push_back(0);
            continue;
        }
        int id = rows.begin()["channel_id"].as<int>();
        //cout << id << endl;
        channel_ids.push_back(to_string(id));
    }
    base_string = "SELECT channel_id,smpl_time,float_val FROM sample WHERE ";
    chan_string = "(channel_id=";
    chan_string += channel_ids[0];
    for (size_t i = 1; i < channel_ids.size(); i++){
        chan_string += " OR channel_id=" + channel_ids[i];
    }
    chan_string = chan_string + ")";
}

void EpicsDBManager::changeChannelNames(vector<size_t> shapeNew, vector< pair <string, string> > dbnamesNew){
    // transform channel names (with %d and multi) and shape into channelnames vector
    //shape = {7,6};
    shape = shapeNew;
    dbnames = dbnamesNew;
    channel_names.clear();
    for (pair<string, string> dbname: dbnames){
        if (dbname.first == "0")
            channel_names.push_back(dbname.second);
        else if (dbname.first == "1")
            for (size_t i = 0; i < shape[1]; i++)
                channel_names.push_back((string)(Form(dbname.second.c_str(),i+1)));
    }

    //channel_names = dbnames;
    remakeChannelIds();
}

void EpicsDBManager::changeListeningPort(int newPort){
    listeningPort = newPort;
}

vector<double> EpicsDBManager::getDBdataBase(string command, string dateLeft){

    /// With verification on null or 0

    vector<double> result;
    vector< vector<double> > resultT;
    for (size_t i = 0; i < channel_ids.size(); i++){
        vector<double> vi = {};
        resultT.push_back(vi);
    }
    string prevDate = "";
    try {
        pqxx::work w(*pqxxConnection);
        pqxx::result rows = w.exec(command);
        for (auto row = rows.begin(); row != rows.end(); ++row) {
            string dateValue = row["smpl_time"].as<std::string>();
            if (row["float_val"].is_null() || row["channel_id"].is_null() || row["smpl_time"].is_null()){
                cout << "found null" << endl;
                continue;
            }
            prevDate = dateValue;
            double dbValue = row["float_val"].as<double>(); 
            int dbId = row["channel_id"].as<int>(); 

            int index = -1;
            auto p = std::find(channel_ids.begin(), channel_ids.end(), to_string(dbId));
            if (p == channel_ids.end()){
                cout << "bad id?" << endl;
                continue;
            }
            index = std::distance(channel_ids.begin(), p);
            resultT[index].push_back(dbValue);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return result;
    }
    //w.commit();
    if (prevDate==""){
        prevDate = dateLeft;
    }

    for (size_t i = 0; i < channel_ids.size(); i++){
        double meani = 0;

        /// fix situation if there are no measurements in the requested t range by getting the last existing entry
        if (resultT[i].size()==0){
            pqxx::work w1(*pqxxConnection);
            pqxx::result rows = w1.exec(base_string + "channel_id=" + channel_ids[i] + " AND float_val!=0 AND smpl_time<='" + prevDate + "' ORDER BY smpl_time DESC LIMIT "+to_string(1));
            double dbValue = rows.begin()["float_val"].as<double>(); 
            meani = dbValue;
        }
        /// else we can have several measurements which are needed to be averaged
        else {
            for (size_t j = 0; j < resultT[i].size(); j++){
                meani+=resultT[i][j]/resultT[i].size();
            }
        }
        result.push_back(meani);
    }
    cout << result.size() << endl;

    vector<double> resultExpanded;
    size_t resultIndex = 0;
    for (size_t i = 0; i< shape[0]; i++){
        if (dbnames[i].first == "0"){
            for (size_t j = 0; j < shape[1]; j++){
                resultExpanded.push_back(result[resultIndex]);
            }
            resultIndex += shape[1];
        }
        else {
            for (size_t j = 0; j < shape[1]; j++){
                resultExpanded.push_back(result[resultIndex]);
                resultIndex += 1;
            }
        }
    }

    cout << resultExpanded.size() << endl;
    
    return resultExpanded;
}

vector<double> EpicsDBManager::getDBdata(int n, string channel){
    string command = base_string + "channel_id=" + channel + " ORDER BY smpl_time ASC LIMIT "+to_string(n);
    return getDBdataBase(command, "2022");
}

vector<double> EpicsDBManager::getDBdata(string date1, string date2){
    string command = base_string + chan_string + " AND smpl_time >= '" + date1 + "' AND smpl_time <= '" + date2 + "' ORDER BY smpl_time ASC";
    return getDBdataBase(command, date1);
}

vector<double> EpicsDBManager::getDBdata(int run, int runnext){
    vector<double> result;
    result = getDBdata(runToDate(run), runToDate(runnext));
    //for (size_t i = 0; i < result.size(); i++)
    //    cout << result[i] << endl;
    return result;
}


void EpicsDBManager::makeTableWithEpicsData(string mode, int runl, int runr){
    vector<int> runBorders = loadrunlist(runl, runr); //444140006
    cout << runBorders.size() << "  " << runBorders[0] << endl;
    //vector<int> runBorders = loadrunlist(443670000, 1e9);

    std::ofstream fout;
    if (mode == "new") 
        fout.open((saveLocation+"info_tables/MDCALLSec2.dat").c_str());
    else if (mode == "app") 
        fout.open((saveLocation+"info_tables/MDCALLSec2.dat").c_str(), std::ios_base::app);
    


    size_t i = 0;
    while (i < runBorders.size()-1){
        vector<double> dbPars = getDBdata(runBorders[i],runBorders[i+1]);
        fout << runBorders[i] << "  ";
        for (size_t j = 0; j < dbPars.size(); j++){
            fout << dbPars[j] << "  ";
        } 
        fout << endl;

        /// save intermediate resulting table
        if ((i + 1) % 100 == 0) {
            fout.close();
            fout.open((saveLocation+"info_tables/MDCALLSec2.dat").c_str(), std::ios_base::app);
            std::cout << "Saved data up to entry " << (i + 1) << std::endl;
        }
        i++;
    }
    fout.close();
}
