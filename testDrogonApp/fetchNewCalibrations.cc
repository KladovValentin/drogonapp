#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <sstream>
#include <vector>
#include <filesystem>
#include <regex>
#include <chrono>
#include <thread>

#include <jsoncpp/json/json.h>

/// g++ -o fetchNewCalibrations fetchNewCalibrations.cc
/// ./fetchNewCalibrations > output.log 2>&1 &

namespace fs = std::filesystem;

using namespace std;

string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";

vector<int> loadrunlist(){
    vector<int> result;
    ifstream fin1;
	fin1.open((saveLocation+"runlistzxc.dat").c_str());
    int run = 0;
    while (fin1 >> run){
        result.push_back(run);
    }
    fin1.close();
    return result;
}

string runToFileName(int run){
    /// Run number to file name (usually, names at HADES mark the start of a run).

    int day = (((int)(run-443670000)/86400)+31);
    int year = 22 + (int)day/365;
    int visoyears = (int)(year-21)/4;
    day = day - 365*(year-22) - visoyears;
    if (day<0){
        year = year-1;
        if ((int)(year-21)/4 == visoyears){
            /// we are currently on visoyear
            day = day+366;
        }
        else{
            day = day+365;
        }
    }
    day = day+1; /// we say "the first of april, not 0th".

    int runMDay = (run-443670000)%86400;
    int hour = (int)(runMDay/3600);
    int minute = (runMDay%3600)/60;
    int second = (runMDay%3600)%60;

    string daystr = day < 100 ? "0"+to_string(day) : to_string(day);
           daystr = day < 10 ? "0"+daystr : daystr;
    string hourstr = hour<10 ? "0"+to_string(hour) : to_string(hour);
    string minutestr = minute<10 ? "0"+to_string(minute) : to_string(minute);
    string secondstr = second<10 ? "0"+to_string(second) : to_string(second);
    return to_string(year) + daystr+hourstr+minutestr+secondstr;
}

vector<string> getNewFNames(string startRun, const char* dirName){
    vector<string> fileNames;
    try {
        for (const auto& entry : fs::directory_iterator(dirName)) {
            if (fs::is_regular_file(entry)) {
                std::string filename = entry.path().filename().string();

                //cout << filename << endl;

                // Use a regular expression to match "be" + number + ".hld" pattern
                //std::regex pattern("be22045(\\d+)0(\\d+).hld");
                std::regex pattern("be(\\d+)\\01.hld");
                std::smatch match;

                if (std::regex_match(filename, match, pattern)) {
                    //cout << match[1].str() << " ";
                    //int runNumber = std::stoi(match[1].str());
                    string runDate = match[1].str();
                    //int splitNumber = std::stoi(match[2].str());
                    //cout << runNumber << "  " << splitNumber << endl;

                    // Check if the number is greater than N
                    if (runDate > startRun) {
                        fileNames.push_back((string)dirName+"/"+filename);
                        std::cout << filename << std::endl;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return fileNames;
}

void writeListFile(vector<string> fileNames, string day){
    ofstream fout(("/home/localadmin_jmesschendorp/lustre/hades/user/vkladov/sub/testBatchFarm/lists/online_day"+day+".list").c_str());
    for (string x: fileNames){
        // Find the position of the substring "/lustre"
        size_t startPos = x.find("/lustre");

        if (startPos != std::string::npos) {
            // Extract the substring starting from "/lustre"
            std::string desiredPart = x.substr(startPos);
            fout << desiredPart << endl;
            std::cout << "Desired Part: " << desiredPart << std::endl;
        } else {
            // Substring not found
            std::cout << "Substring not found." << std::endl;
        }
    }
    fout.close();
}

std::string executeCommand(const char* command) {
    FILE* pipe = popen(command, "r");
    if (!pipe) {
        return "Error executing command";
    }

    char buffer[128];
    std::string result = "";

    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL) {
            result += buffer;
        }
    }

    pclose(pipe);
    return result;
}

void appendTargetFileWithNewEntries(std::pair<int, int> newEntriesIndices){
    //std::string newTable = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/info_tables/run-mean_dEdxMDCSecModNew.dat";
    //std::string oldTable = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/info_tables/run-mean_dEdxMDCSecModzxc.dat";
    std::string newTable = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/info_tables/run-mean_dEdxMDCSecModAdd.dat";
    std::string oldTable = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/info_tables/run-mean_dEdxMDCSecModzxc.dat";

    // Read old file
    std::ifstream file(oldTable);
    if (!file.good()){
        std::ofstream fileCreate(oldTable);
        if (fileCreate.is_open()) {
            std::cout << "File created." << std::endl;
        }
        fileCreate.close();
    }
    std::map<int,int> existingEntries;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double value;
        std::vector<double> row;
        while (iss >> value) {
            row.push_back(value);
        }
        existingEntries[(int)row[0]] = (int)row[0];
    }
    file.close();

    cout << existingEntries.size() << endl;


    // Read new file
    std::ifstream newfile(newTable);
    std::vector< std::vector<double> > newEntries;
    while (std::getline(newfile, line)) {
        std::istringstream iss(line);
        double value;
        std::vector<double> row;
        while (iss >> value) {
            row.push_back(value);
        }
        newEntries.push_back(row);
    }
    newfile.close();


    // Append old file
    std::ofstream outfile(oldTable, std::ios::app);
    int index = 0;
    for (std::vector<double> newEntry: newEntries){
        if (existingEntries.find((int)newEntry[0]) != existingEntries.end()) {
            continue;
        }
        //else if (newEntry.size() == 5 && index >= newEntriesIndices.first && index < newEntriesIndices.second) {
        else if (newEntry.size() == 5) {
            outfile << (int)newEntry[0] << "    " << (int)newEntry[1] << "  " << (int)newEntry[2] << "  " << newEntry[3] << "   " << newEntry[4] << std::endl;
        }
        index += 1;
    }
    outfile.close();
    
}

int main(int argc, char* argv[]) {
    // Check if the required number of command-line arguments is provided
    if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << "<folder_path>\n";
        return 1;
    }
    int nextDay = 0;
    int tempCount = 41;

    while(1){

        vector<int> runlist = loadrunlist();

        Json::Value jsonObj;
        std::ifstream inFile((saveLocation + "fetchSettings.json").c_str());
        inFile >> jsonObj;
        inFile.close();
        string day = jsonObj["day"].asString();
        //string day = "045";

        int num = std::stoi(day) + nextDay;
        day = num < 100 ? "0"+to_string(num) : to_string(num);
        day = num < 10 ? "0"+day : day;

        string baseBaseDir = "/home/localadmin_jmesschendorp/lustre/hades/raw/feb22/22/"; 
        string baseRawDir = baseBaseDir + day;
        string startRun = "22045213722";
        //string startRun = runToFileName(runlist[runlist.size()-1]);


        // Write the list of new exp files
        //vector<string> newFileNames = getNewFNames(startRun, baseRawDir.c_str());
        //std::sort(newFileNames.begin(), newFileNames.end());

        cout << "x" << endl;

        // If the set day is old and newfiles are depleted -> switch to new day (only here because want some freedom with this setting in the file)
        auto currentTime = std::chrono::system_clock::now();
        std::time_t currentTime_t = std::chrono::system_clock::to_time_t(currentTime);
        std::tm* currentTime_tm = std::localtime(&currentTime_t);
        int dayOfYear = currentTime_tm->tm_yday + 1; // Adding 1 since day of the year starts from 0
        //if (newFileNames.size()==0 && dayOfYear > std::stoi(day)){
        //    nextDay += 1;
        //    continue;
        //}
        nextDay = 0;

        //writeListFile(newFileNames, day);


        std::string launchScript = (string)("ssh -J vkladov@lxpool.gsi.de vkladov@vae23.hpc.gsi.de") +
                                    (string)(" 'cd /lustre/hades/user/vkladov/sub/testBatchFarm") +
                                    (string)(" . ./sendScript_raw_SL.sh " + day + "'");
        //std::system(launchScript.c_str());


        std::string checkSqueue = (string)("ssh -J vkladov@lxpool.gsi.de vkladov@vae23.hpc.gsi.de") +
                                    (string)(" 'squeue -u vkladov'");

        int index1 = 0; int index2 = 10*24;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        tempCount+=1;
        day = tempCount < 100 ? "0"+to_string(tempCount) : to_string(tempCount);
        day = tempCount < 10 ? "0"+day : day;
        while (1){
            cout << day << "    " << tempCount << endl;
            std::this_thread::sleep_for(std::chrono::seconds(0));
            /*string squeueOut = executeCommand(checkSqueue.c_str());
            if (squeueOut.find("vkladov") != std::string::npos){
                cout << "jobs are still running" << endl;
                std::this_thread::sleep_for(std::chrono::minutes(15));
            }
            else{*/
                cout << "jobs finished" << endl;
                std::string hadd = (string)("ssh -J vkladov@lxpool.gsi.de vkladov@vae23.hpc.gsi.de") +
                                    (string)(" 'cd testBatchFarm;") +
                                    (string)(" hadd -f /u/vkladov/testBatchFarm/rawHists45.root /lustre/hades/user/vkladov/rawCalStuff/day" + day + "/*.root;") +
                                    (string)(" root -l -q analyseHists.cc'");
                std::system(hadd.c_str());
                std::system("cp /home/localadmin_jmesschendorp/u/testBatchFarm/mDCCals45.dat /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/info_tables/run-mean_dEdxMDCSecModAdd.dat");
                cout << "copied" << endl;
                break;
            //}

            
        }
        /// append existing table with new data, check if it is new?
        appendTargetFileWithNewEntries(std::make_pair(index1,index2));
        cout << "appended file" << endl;
        index1+=10*24;
        index2+=10*24;
        //break;
    }



    // fill raw_day_x.list on lustre testBatchFarm
    // launch sendScrip... with argument being this new list
    // ssh lustre and periodically check squeue -u vkladov output
    // hadd files, can make from here?
    // launch analyseHists script via root (root -x ...?) to make a table
    // append target table file

    return 0;
}

