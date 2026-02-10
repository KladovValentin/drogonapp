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

/// g++ -std=c++17 -o fetchNewRuns fetchNewRuns.cc $(pkg-config --cflags --libs jsoncpp)
/// ./fetchNewRuns > outputRun.log 2>&1 &

namespace fs = std::filesystem;
using namespace std;

string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/";

vector<int> loadrunlist(int run1, int run2, string fName){
    vector<int> result;
    ifstream fin1((saveLocation+fName).c_str());
    if (!fin1.good()){
        std::ofstream fileCreate((saveLocation+fName).c_str());
        if (fileCreate.is_open()) {
            std::cout << "File created." << std::endl;
        }
        fileCreate.close();
        return result;
    }
    int run = 0;
    while (fin1 >> run){
        result.push_back(run);
    }
    fin1.close();
    return result;
}

int getRunIdFromFileName(string fName){
    string inputString = fName;

    size_t startPos = inputString.find("be");
    size_t endPos = inputString.find(".hld");
    size_t prefixLength = ((string)("be")).size();
    size_t length = endPos - prefixLength - 2;
    inputString = inputString.substr(startPos + prefixLength, length);

    int year = stoi(inputString.substr(0, 2));
    int dayYear = stoi(inputString.substr(2, 3));
    int hour = stoi(inputString.substr(5, 2));
    int minute = stoi(inputString.substr(7, 2));
    int second = stoi(inputString.substr(9, 2));

    int result = 443670000 - 31*86400 + (year-22)*365*86400 + (dayYear-1)*86400 + hour*3600 + minute*60 + second;
    return result;
}

int main(int argc, char* argv[]) {
    vector<int> sourceRunListzxc = loadrunlist(0,1e9,"runlist.dat");
    int index = 0;
    int nextDay = 0;
    while(true){
        Json::Value jsonObj;
        std::ifstream inFile((saveLocation + "fetchSettings.json").c_str());
        inFile >> jsonObj;
        inFile.close();
        string day = jsonObj["day"].asString();
        int num = std::stoi(day) + nextDay;
        day = num < 100 ? "0"+to_string(num) : to_string(num);
        day = num < 10 ? "0"+day : day;

        const char* dirName = ("/home/localadmin_jmesschendorp/lustre/hades/raw/feb22/22/"+day).c_str();
        /// Get startRun from runlist
        vector<int> currentRunList = loadrunlist(0,1e9,"runlistzxc.dat");
        int startRunN = 0;
        if (currentRunList.size()>0)
            startRunN = *max_element(currentRunList.begin(), currentRunList.end());
        vector<int> newRuns;
        
        if (index < sourceRunListzxc.size())
            newRuns.push_back(sourceRunListzxc[index]);
        else
            break;
        if (sourceRunListzxc[index]>448533980 || sourceRunListzxc[index] < 443756673){
            index+=1;  
            continue;
        }
        index+=1;
        /*
        try {
            for (const auto& entry : fs::directory_iterator(dirName)) {
                if (fs::is_regular_file(entry)) {
                    std::string filename = entry.path().filename().string();

                    //cout << filename << endl;
                    std::regex pattern("be(\\d+)\\01.hld");
                    std::smatch match;

                    if (std::regex_match(filename, match, pattern)) {
                        string runDate = match[1].str();
                        int runNumber = getRunIdFromFileName(filename);

                        // Check if the number is greater than N
                        if (runNumber > startRunN) {
                            newRuns.push_back(runNumber);
                            std::cout << filename << std::endl;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }

        // Get the current time point
        auto currentTime = std::chrono::system_clock::now();
        // Convert the current time point to a time_t object
        std::time_t currentTime_t = std::chrono::system_clock::to_time_t(currentTime);
        // Convert the time_t object to a tm structure
        std::tm* currentTime_tm = std::localtime(&currentTime_t);
        // Get the day of the year from the tm structure
        int dayOfYear = currentTime_tm->tm_yday + 1; // Adding 1 since day of the year starts from 0
        if (newRuns.size()==0 && dayOfYear > std::stoi(day)){
            nextDay += 1;
            continue;
        }
        nextDay = 0;

        */

        ofstream fout1;
        fout1.open((saveLocation + "runlistzxc.dat").c_str(), std::ios_base::app);
        for (size_t i = 0; i < newRuns.size(); i++){
            fout1 << newRuns[i] << endl;
        }
        fout1.close();
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

}
