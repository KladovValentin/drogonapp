#include "../include/functions.h"

//#include "../include/constants.h"

const std::string saveLocation = "/home/localadmin_jmesschendorp/gsiWorkFiles/realTimeCalibrations/backend/serverData/";

using namespace std;

vector<string> stringFunctions::tokenize(string str, const char *delim){
    vector<string> result;
    TString line = str;
    TString tok;
    Ssiz_t from = 0;
    while (line.Tokenize(tok, from, delim))
        result.push_back((string)tok);
    return result;
}



vector<double> vectorFunctions::runningMeanVector(vector<double> vect, int avRange){
    /// Calculates running mean of a vector, centered at the entry, iteratively skips too off values

    double maxdiff = *max_element(vect.begin(), vect.end()) - *min_element(vect.begin(), vect.end());
    TH1 *hStep = new TH1F("hStep",";diff;counts",30,-maxdiff,maxdiff);
    for (size_t i = 0; i < vect.size()-1; i++){
        if (vect[i] > 0 && vect[i+1] > 0){
            hStep->Fill(vect[i+1]-vect[i]);
        }
    }
    double stepCut = 1.5*hStep->GetRMS();
    double stepMean = hStep->GetMean();
    vector<bool> goodForAveraging;
    goodForAveraging.push_back(false);
    for (size_t i = 0; i < vect.size()-1; i++){
        if (fabs(vect[i+1]-vect[i] - stepMean) > stepCut){
            goodForAveraging.push_back(false);            
        }
        else {
            goodForAveraging.push_back(true);
        }
    }
    vector<double> result;
    for (size_t i = 0; i < vect.size(); i++){
        if (!goodForAveraging[i]){
            result.push_back(vect[i]);
            continue;
        }

        double mean2 = 0;
        int count1 = 0;
        for (size_t j = 0; j < avRange+1; j++){
            if ((i+j-avRange/2<0) || (i+j-avRange/2>=vect.size()) || (fabs(vect[i+j-avRange/2]-vect[i] - stepMean) > stepCut))
                continue;
            count1 +=1;
            mean2+=vect[i+j-avRange/2];
        }
        mean2 = mean2/count1;
        if (count1 == 0)
            mean2 = vect[i];
        result.push_back(mean2);
    }
    hStep->Delete();


    /*vector<double> result;
    TH1 *h19 = new TH1F("h19",";ratio;counts",30,0.5,1.5);
    for (size_t i = 0; i < vect.size(); i++){
        int j = 0;

        double mean1 = 0, std1 = 0;
        int count1 = 0;
        for (size_t j = 0; j < 101; j++){
            if ((i+j-50<0) || (i+j-50>=vect.size()))
                continue;
            count1 +=1;
            mean1+=vect[i+j-50];
            std1+= vect[i+j-50]*vect[i+j-50];
        }
        mean1 = mean1/count1;
        std1 = sqrt(std1/count1 - mean1*mean1);

        double mean = 0, std = 0;
        count1 = 0;
        for (size_t j = 0; j < avRange+1; j++){
            if ((i+j-avRange/2<0) || (i+j-avRange/2>=vect.size()) || (fabs(vect[i+j-avRange/2]-mean1) > 4*std1))
                continue;
            count1 +=1;
            mean+=vect[i+j-avRange/2];
            std+= vect[i+j-avRange/2]*vect[i+j-avRange/2];
        }
        mean = mean/count1;
        if (count1 == 0)
            mean = vect[i];
        std = sqrt(std/count1 - mean*mean);

        double mean2 = 0;
        count1 = 0;
        for (size_t j = 0; j < avRange+1; j++){
            if ((i+j-avRange/2<0) || (i+j-avRange/2>=vect.size()) || (fabs(vect[i+j-avRange/2]-mean) > 3*std))
                continue;
            count1 +=1;
            mean2+=vect[i+j-avRange/2];
        }
        mean2 = mean2/count1;
        if (count1 == 0)
            mean2 = vect[i];
        //if (fabs(vect[i]-mean) > 4*std)
        //    mean2 = vect[i];//mean2;//vect[i];
        
        //vect[i] = mean;
        h19->Fill(vect[i]/mean);
        result.push_back(mean2);
    }
    //h19->Draw();
    //TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);
	//c->Update();
    h19->Delete();*/
    return result;
}

vector<double> vectorFunctions::normalizeVectorNN(vector<double> vect){
    /// Normalizes to "normal" distribution, with 0 average and 1 std

    double mean0 = 0, std0 = 0;
    for (double x : vect) {
        mean0 += x;
        std0 += x * x;
    }
    mean0 = mean0/vect.size();
    std0 = sqrt(std0/vect.size() - mean0*mean0);
    //remove too off values from std calculation
    int length = 0;
    double mean = 0, std = 0;
    for (double x : vect) {
        if (fabs(x-mean0)/std0>5)
            continue;
        length += 1;
        mean += x;
        std += x * x;
    }
    mean = mean/length;
    std = sqrt(std/length - mean*mean);

    for (double& x : vect) {
        if (std != 0)
            x = (x-mean)/std;
        else
            x = x - mean;
    }
    return vect;
}
double vectorFunctions::normalizeVector(vector<double>& vect){
    /// Normalizes to average 1 
    double mean = 0;
    for (size_t i = 0; i < vect.size(); i++)
        mean += vect[i];
    mean = mean/vect.size();
    for (size_t i = 0; i < vect.size(); i++)
        vect[i] = vect[i]/mean;
    return mean;
}

vector<float> vectorFunctions::transposeTensorDimensions(vector<float> inputTensor, vector<long int> dims, vector<size_t> permutation){
    size_t numElements = 1;
    for (size_t dim : dims) {
        numElements *= dim;
    }
    std::vector<float> transposedTensor(numElements);

    // Perform the transpose operation
    for (size_t i = 0; i < numElements; ++i) {
        // Convert the flat index to multi-dimensional indices
        size_t flatIndex = i;
        std::vector<size_t> indices(dims.size());
        for (size_t j = 0; j < dims.size(); ++j) {
            indices[j] = flatIndex % (size_t)dims[j];
            flatIndex /= (size_t)dims[j];
        }

        // Permute the indices according to the given permutation vector
        std::vector<size_t> transposedIndices(dims.size());
        for (size_t j = 0; j < dims.size(); ++j) {
            transposedIndices[j] = indices[permutation[j]];
        }

        // Convert the transposed indices back to a flat index for the transposed tensor
        size_t transposedFlatIndex = 0;
        size_t factor = 1;
        for (size_t j = 0; j < dims.size(); ++j) {
            transposedFlatIndex += transposedIndices[j] * factor;
            factor *= dims[j];
        }

        // Copy the value from the original tensor to the transposed tensor
        transposedTensor[i] = inputTensor[transposedFlatIndex];
    }

    // Copy the transposed tensor back to the original tensor
    return transposedTensor;
}


static constexpr long long kRunOffsetSec = 1200000000LL;  // your found shift

double dateRunF::dateToTimeStamp(int year, int month, int day, int hour, int min, int sec){
    std::tm t = {};
    t.tm_year = year - 1900;  // years since 1900
    t.tm_mon  = month - 1;    // months since January
    t.tm_mday = day;
    t.tm_hour = hour;
    t.tm_min  = min;
    t.tm_sec  = sec;
    t.tm_isdst = -1;          // let system decide about DST
    return static_cast<double>(mktime(&t));  // seconds since 1970-01-01
}


double dateRunF::dateToNumber(string date){
    /// Date format with "-" to Root's time axis format. Simplifications are from ROOT itself

    vector<string> tokens = stringFunctions::tokenize(date, "[-]");
    vector<double> segments;
    for (size_t i = 0; i < tokens.size(); i++)
        segments.push_back(atof(tokens[i].c_str()));
    double baseTime = 22*30*12 + 30 + 2 + 15./24. + 58./24./60.;
    return ((segments[0]-2000)*30*12 +(segments[1]-1)*30.+ (segments[2]-1) + segments[3]/24. + segments[4]/24./60. + segments[5]/24./60./60. - baseTime)*24*60*60;
}

string dateRunF::runToDate(int run){
    using namespace std::chrono;

    // Convert to time_t
    std::time_t t = static_cast<std::time_t>(run + kRunOffsetSec);

    // Break down either in UTC or local time
    std::tm tm{};
    //if (useLocalTime) {
    //    tm = *std::localtime(&t);
    //} else {
        tm = *std::localtime(&t);
        //tm = *std::gmtime(&t);
    //}

    // Format
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return os.str();
    
}

double dateRunF::runToDateNumber(int run){
    
    /// Transforms run number to the date. Corrects for 29 days in Feb, for diff in months length

    using namespace std::chrono;
    std::time_t t = static_cast<std::time_t>(run + kRunOffsetSec);
    std::tm tm{};
    tm = *std::localtime(&t);

    //if (run > 495508228)
    //cout << run << "    " << date << endl;
    //return dateRunF::dateToTimeStamp(year, month, day, hour, minute, second);
    return static_cast<double>(mktime(&tm));
    //return dateRunF::dateToNumber(date);
}

TString dateRunF::runToFileName(int run){
    /// Run number to file name (usually, names at HADES mark the start of a run).

    using namespace std;

    // 1) Apply fixed offset to map your "run" seconds to Unix epoch seconds (UTC)
    const time_t corrected = static_cast<time_t>( static_cast<long long>(run) + kRunOffsetSec );

    // 2) Break down in UTC
    std::tm tmUtc = *std::localtime(&corrected);

    // 3) Build components
    const int yearFull = tmUtc.tm_year + 1900;      // e.g. 2022
    const int year2    = yearFull % 100;            // e.g. 22
    const int yday1    = tmUtc.tm_yday + 1;         // 1..366 (Julian day of year)

    // 4) Format YY, DDD, HH, MM, SS with zero padding
    std::ostringstream yy, ddd, hh, mm, ss;
    yy  << std::setw(2) << std::setfill('0') << year2;
    ddd << std::setw(3) << std::setfill('0') << yday1;
    hh  << std::setw(2) << std::setfill('0') << tmUtc.tm_hour;
    mm  << std::setw(2) << std::setfill('0') << tmUtc.tm_min;
    ss  << std::setw(2) << std::setfill('0') << tmUtc.tm_sec;

    // 5) Assemble "beYYDDDHHMMSS"
    const std::string name = std::string("be")
                           + yy.str() + ddd.str() + hh.str() + mm.str() + ss.str();

    return TString(name.c_str());
}

int dateRunF::getRunIdFromAnyFileName(string fName, size_t leftCut, size_t rightCut){
    /// Old function, shouldn't be used
    string erased = fName.erase(0,leftCut);
    erased = erased.substr(0, erased.size()-rightCut);
    //cout << erased << endl;
    int date = stoi(erased);
    int month = (date/1000000-31)/29;
    int day = (date/1000000-31)-28*((date/1000000-31)/29);
    int hour= (date%1000000)/10000;
    int min = ((date%1000000)%10000)/100;
    int sec = ((date%1000000)%10000)%100;
    //std::cout << 443670000 + (day-1)*86400 + hour*3600 + min*60 + sec << std::endl;
    return 443670000 + month*(28*86400) + (day-1)*86400 + hour*3600 + min*60 + sec;
}

int dateRunF::getRunIdFromFileName(string fName){
    /// Maybe add viso years, need to check
    string inputString = fName;

    size_t startPos = inputString.find("be");
    size_t endPos = inputString.find(".hld");
    size_t prefixLength = ((string)("be")).size();
    size_t length = endPos - prefixLength - 2;
    inputString = inputString.substr(startPos + prefixLength, length);

    int YY   = std::stoi(inputString.substr(0,2));
    int DDD  = std::stoi(inputString.substr(2,3));
    int HH   = std::stoi(inputString.substr(5,2));
    int MM   = std::stoi(inputString.substr(7,2));
    int SS   = std::stoi(inputString.substr(9,2));

    int year = 2000 + YY;   // “22” → 2022 etc.

    // Build tm from year, day-of-year and h:m:s
    std::tm tmUtc{};
    tmUtc.tm_year = year - 1900;
    tmUtc.tm_mon  = 0;       // Jan
    tmUtc.tm_mday = DDD;     // day-of-year counting from Jan 1
    tmUtc.tm_hour = HH;
    tmUtc.tm_min  = MM;
    tmUtc.tm_sec  = SS;
    tmUtc.tm_isdst = -1;

    // Convert to seconds since 1970-01-01 UTC
    // timegm() interprets tm as UTC (on Linux/glibc). 
    // If only mktime() is available, it assumes localtime.
    std::time_t unixUTC = mktime(&tmUtc); 

    // Undo the run offset to get the original run integer
    long long run = static_cast<long long>(unixUTC) - kRunOffsetSec;
    return static_cast<int>(run);
}


int dateRunF::getRunIdFromDQFileName(string fName){
  return getRunIdFromAnyFileName(fName,5,26);
}

int dateRunF::getRunFromFullDQFile(TString inFile){
    TString tok;
    Ssiz_t from1 = 0;
    vector<TString> tokens;
    while (inFile.Tokenize(tok, from1, "[/]")) {
		tokens.push_back(tok);
	}
    return getRunIdFromDQFileName(string(tokens[tokens.size()-1]));
}

vector<double> dateRunF::timeVectToDateNumbers(vector<double> vect){
    for (double& x : vect)
        x = dateRunF::runToDateNumber(x);
    return vect;
}

vector<TString> dateRunF::getlistOfFileNames(TString inDir){
    string filesStr = (string)(gSystem->GetFromPipe("ls " + inDir));
    cout << filesStr[0] << endl;
    vector<string> files = stringFunctions::tokenize(filesStr, "\n");
    vector<TString> fileNames;
    for (size_t i = 0; i < files.size(); i++) {
        vector<string> tokens = stringFunctions::tokenize(files[i], "[/]");
        fileNames.push_back((TString)(tokens[tokens.size()-1]));
    }
    return fileNames;
}

void dateRunF::saveRunNumbers(string expFilesLocation){
    vector<TString> filesNames = dateRunF::getlistOfFileNames( (expFilesLocation + "1*/be**01.hld").c_str());
    ofstream fout1;
	fout1.open((saveLocation + "runlist25.dat").c_str());
    for (size_t i = 0; i < filesNames.size(); i++){
        fout1 << dateRunF::getRunIdFromFileName((string)(filesNames[i])) << endl;
        //fout1 << filesNames[i] << "  " << dateRunF::getRunIdFromFileName((string)(filesNames[i])) << endl;
    }
    fout1.close();
}

vector<int> dateRunF::loadrunlist(int run1, int run2){
    vector<int> result;
    ifstream fin1;
	//fin1.open((saveLocation+"runlistzxc.dat").c_str());
    fin1.open((saveLocation+"runlist.dat").c_str());
    int run = 0;
    while (fin1 >> run){
        if (run <= run1 || run >= run2)
            continue;
        result.push_back(run);
    }
    fin1.close();
    return result;
}


vector< pair<int,int> > dateRunF::loadrunlistWithEnds(int run1, int run2){
    vector< pair<int,int> > result;
    ifstream fin1;
	//fin1.open((saveLocation+"runlistzxc.dat").c_str());
    //fin1.open((saveLocation+"runlist.dat").c_str());
    fin1.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/runList.dat");
    //fin1.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/runListCosmics106.dat");
    //fin1.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/runListBeam25.dat");
    //fin1.open("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/runListBeam24.dat");
    int run = 0; int runEnd = 0;
    while (fin1 >> run >> runEnd){
        if (run <= run1 || runEnd >= run2)
            continue;
        result.push_back(std::make_pair(run,runEnd));
    }
    fin1.close();
    return result;
}




vector< pair< int, vector<double> > > preTrainFunctions::readClbTableFromFile(string fName){
    vector< pair< int, vector<double> > > result;
    ifstream fin1;
	fin1.open(fName.c_str());
    string line;
	while (std::getline(fin1, line)){
        double run = 0;
        vector<double> vars;
        std::stringstream ss(line);
        double value;
        ss >> run;
        while (ss >> value) {
            vars.push_back(value);
        }
        result.push_back(make_pair(run,vars));
    }
    fin1.close();
    return result;
}

vector< pair< int, vector<double> > > preTrainFunctions::readClbTableFromFileExtended(string fName){
    vector< pair< int, vector<double> > > result;
    ifstream fin1;
	fin1.open(fName.c_str());
    string line;
	while (std::getline(fin1, line)){
        double run = 0;
        vector<double> vars;
        std::stringstream ss(line);
        double value;
        ss >> run;
        for (size_t i = 0; i < 10; i++){
            std::getline(fin1, line);
            std::stringstream ss1(line);
            while (ss1 >> value) {
                vars.push_back(value);
            }
        }
        result.push_back(make_pair(run,vars));
    }
    fin1.close();
    return result;
}

vector< pair< string, vector<double> > > preTrainFunctions::readDBTableFromFile(string fName){
    ifstream fin1;
	fin1.open(fName.c_str());
    string line0 = "";
    TString line = "";
    while(!line.Contains("# Time")){
        getline(fin1, line0);
        line = line0;
    }
    vector< pair< string, vector<double> > > result;
	while (std::getline(fin1, line0)){
        string date = "";
        string time = "";
        vector<double> values;
        
        std::stringstream ss(line0);

        ss >> date >> time;
        vector<string> segments = stringFunctions::tokenize(time, "[:.]");
        string date_time = date + "-" + segments[0] + "-" + segments[1] + "-" + segments[2];
        
        bool goodFlag = true;
        string tmp;
        while (ss >> tmp) {
            line = tmp;
            if (line.Contains("Error") || line.Contains("Archive") || line.Contains("Data") || line.Contains("N/A")){
                goodFlag = false;
                continue;    
            }
            tmp.erase(std::remove(tmp.begin(), tmp.end(), ','), tmp.end());
            values.push_back(atof(tmp.c_str()));
        }
        if (goodFlag){
            result.push_back(make_pair(date_time,values));
            //cout << result[result.size()-1].first << endl;
        }
    }
	fin1.close();
    return result;
}


vector< vector<double> > preTrainFunctions::clbTableToVectorsTarget(vector< pair< int, vector<double> > > intable){
    //vector< vector<double> > arr = new vector<double>[4];
    vector< vector<double> > arr;
    arr.resize(4,vector<double>(0));
    
    for (size_t i = 0; i < intable.size(); i++){
        arr[0].push_back((intable[i].first));
        arr[1].push_back(intable[i].second[0]);
        arr[2].push_back(0);
        arr[3].push_back(intable[i].second[1]);
    }
    return arr;
}

std::map< int, vector<double> > preTrainFunctions::clbTableToVectorsTarget(vector< pair< int, vector<double> > > intable, vector<int> shape){
    //vector< vector<double> > arr = new vector<double>[4];
    std::map< int, vector<double> > arr;
    int shapeSize = shape.size();
    vector<int> singularVector{1};
    if (shape == singularVector)
        shapeSize = 0;
    int outShapeLength = 1; for (int x: shape){ outShapeLength*=x; }

    for (size_t i = 0; i < intable.size(); i=i+outShapeLength){     //as there are lines for each node, we have to go through them and combine in one line
        if (i+outShapeLength > intable.size()){
            break;
        }
        map<int,int> runsRun;
        for (size_t j = 0; j < outShapeLength; j++){
            runsRun[intable[i+j].first] = intable[i+j].first;
        }
        if (runsRun.size() > 1){   
            break;    // check if there are 2 runs in a nondes_length segment (should be one line for node, related to each run)
        }

        vector<double> meanValues;
        //meanValues.push_back(intable[i].second[0]);                    // end run case
        for (size_t j = 0; j < outShapeLength; j++){
            meanValues.push_back(intable[i+j].second[shapeSize+1]);    // +1 only for end run (shapeSize for m+s)
        }
        for (size_t j = 0; j < outShapeLength; j++){
            meanValues.push_back(intable[i+j].second[shapeSize+1+1]);  // extra +1 from error and another from end run
        }
        arr[intable[i].first] = meanValues;
    }
    return arr;
}

vector< vector<double> > preTrainFunctions::clbTableToVectorsInput(vector< pair< int, vector<double> > > intable){
    
    int size = intable[0].second.size();
    vector< vector<double> > arr;
    arr.resize(size+1,vector<double>(0));
    
    for (size_t i = 0; i < intable.size(); i++){
        arr[0].push_back((intable[i].first));
        
        for (size_t j = 0; j < size; j++) {
            arr[j+1].push_back(intable[i].second[j]);
        }
    }
    return arr;
}

vector< vector<double> > preTrainFunctions::dbTableToVectorsAveraged(vector< pair< string, vector<double> > > table){
    vector<int> runs = dateRunF::loadrunlist(443670000,10000000000); 

    map< int, vector< double> > dataBaseTable;
    int size = table[0].second.size();
    size_t firstRun = 0, secondRun = 1;
    vector<double> xs, ys;
    for (size_t i = 0; i < table.size()-1; i++){

        double x = dateRunF::dateToNumber(table[i].first);
        double x1 = dateRunF::dateToNumber(table[i+1].first);
        
        if (x < dateRunF::runToDateNumber(runs[firstRun]))       // if db starts earlier
            continue;
        if (firstRun+1 > runs.size()-1)                          // if ends later
            break;
        while (x > dateRunF::runToDateNumber(runs[firstRun+1])){ // jumped through runs, skip till coincidence
            if (firstRun+1 > runs.size()-2)
                break;
            firstRun += 1;
        }
        secondRun = firstRun + 1;
        xs.push_back(x);
        if (x1 > dateRunF::runToDateNumber(runs[secondRun])){
            vector<double> tVect;
            for (size_t k = 0; k < size; k++){
                double ymean = 0; for(size_t j=0; j<xs.size(); j++) { ymean+=table[i].second[k]/xs.size(); }
                tVect.push_back(ymean);
            }
            dataBaseTable[(int)runs[firstRun]] = tVect;

            xs.clear();
            ys.clear();
            firstRun = secondRun;
        }
    }
    vector< vector<double> > arr;
    arr.resize(size+1,vector<double>(dataBaseTable.size()));
    //vector<double>* arr = new vector<double>[size+1];

    int counter = 0;
    for (const auto& n : dataBaseTable){
        arr[0][counter] = (n.first);
        for (size_t j = 0; j < size; j++)
            arr[j+1][counter]  = (n.second[j]);
        counter += 1;
    }
    
    return arr;
}
