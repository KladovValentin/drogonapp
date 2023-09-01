
#include "../include/neuralNetwork.h"

#include "../include/functions.h"
#include "../include/constants.h"

using namespace std;
using namespace preTrainFunctions;
using namespace vectorFunctions;
using namespace dateRunF;


NeuralNetwork::NeuralNetwork(){
    additionalFilter = {};
    setupNNPredictions();
}
NeuralNetwork::~NeuralNetwork(){
}

void NeuralNetwork::setupNNPredictions(){

    ifstream fin1;
    meanValues.clear();
	fin1.open((saveLocation+"function_prediction/meanValues.txt").c_str());
    while (!fin1.eof()){
        double a;
        fin1 >> a;
        meanValues.push_back(a);
    }
    fin1.close();
    stdValues.clear();
    fin1.open((saveLocation+"function_prediction/stdValues.txt").c_str());
    while (!fin1.eof()){
        double a;
        fin1 >> a;
        stdValues.push_back(a);
    }
    fin1.close();

    Ort::Env env;
    std::string model_path = saveLocation+ "function_prediction/tempModel.onnx";
    mSession = new Ort::Session(env, model_path.c_str(), Ort::SessionOptions{ nullptr });
    
    Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    mInputDims = inputTensorInfo.GetShape();

    Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    mOutputDims = outputTensorInfo.GetShape();


    inputTensorSize = mInputDims[1]*mInputDims[2];
    //memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

    size_t os = 1;
    for (size_t i = 0; i < mOutputDims.size(); i++)
        os = os * mOutputDims[i];
    outputTensorSize = os;
}

float NeuralNetwork::getPrediction(vector<float> inputTensorValues){
    for (size_t i = 0; i < 15; i++){
        for (size_t j = 0; j < 12; j++){
            inputTensorValues[i*12+j] = (float)((inputTensorValues[i*12+j]-meanValues[j])/stdValues[j]);
        }
    }
    const char* mInputName[] = {"input"};
    const char* mOutputName[] = {"output"};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),inputTensorSize,mInputDims.data(),mInputDims.size());
    vector<float> outputTensorValues(outputTensorSize);
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize,mOutputDims.data(), mOutputDims.size());
    mSession->Run(Ort::RunOptions{nullptr}, mInputName,&inputTensor, 1, mOutputName,&outputTensor, 1);
    
    //vector<float> outputProbs;
    //for (size_t i = 0; i < outputTensorSize; i++)
    //    outputProbs.push_back(outputTensorValues[i]);
    cout << "output size is " << outputTensorValues.size() << " " << inputTensorSize << endl;
    cout << "prediction is " << outputTensorValues[outputTensorValues.size()-1]*stdValues[stdValues.size()-1] + meanValues[meanValues.size()-1] << endl;
    return outputTensorValues[outputTensorValues.size()-1]*stdValues[stdValues.size()-1] + meanValues[meanValues.size()-1];
}

void NeuralNetwork::drawTargetStability(TGraphErrors* targetsAll, TGraphErrors* targetsPP, TGraphErrors* targetsHH){
    targetsAll->SetMarkerStyle(23);
    targetsAll->SetMarkerSize(0.5);
    targetsAll->SetMarkerColor(2);
    targetsAll->GetYaxis()->SetRangeUser(-2,2);
    targetsAll->GetXaxis()->SetTimeDisplay(1);
    targetsAll->GetXaxis()->SetTimeFormat("%d/%m %H");
    targetsAll->SetTitle(";time;relative change All hadrons");

    targetsHH->SetMarkerStyle(24);
    targetsHH->SetMarkerSize(0.5);
    targetsHH->SetMarkerColor(3);
    targetsHH->GetYaxis()->SetRangeUser(-2,2);
    targetsHH->GetXaxis()->SetTimeDisplay(1);
    targetsHH->GetXaxis()->SetTimeFormat("%d/%m %H");
    targetsHH->SetTitle(";time;relative change H-H");

    targetsPP->SetMarkerStyle(22);
    targetsPP->SetMarkerSize(0.5);
    targetsPP->SetMarkerColor(4);
    targetsPP->SetLineColor(4);
    targetsPP->SetLineWidth(1);
    targetsPP->SetName("grClb");
    targetsPP->GetYaxis()->SetRangeUser(-2,2);
    targetsPP->GetXaxis()->SetTimeDisplay(1);
    targetsPP->GetXaxis()->SetTimeFormat("%d/%m %H");
    targetsPP->SetTitle(";time;relative change H-F");

    TCanvas* canvas = new TCanvas("canvas", "Canvas Title", 1220, 980);
    canvas->Divide(1, 3); // Divide canvas into 3 rows and 1 column
    canvas->cd(1);
    targetsAll->Draw("AP");
    canvas->cd(2);
    targetsPP->Draw("AP");
    canvas->cd(3);
    targetsHH->Draw("AP");
    canvas->Update();
    // Keep the application running until the user closes the canvas
    //canvas->WaitPrimitive();
    cin.get();
    delete canvas;
}

void NeuralNetwork::drawInputTargetCorrelations(TGraphErrors* target, vector< vector<double> > inputs){

    target->SetMarkerStyle(22);
    target->SetMarkerSize(0.5);
    target->SetMarkerColor(4);
    target->SetLineColor(4);
    target->SetLineWidth(1);
    target->SetName("grClb");
    target->GetYaxis()->SetRangeUser(-2,2);
    target->GetXaxis()->SetTimeDisplay(1);
    target->GetXaxis()->SetTimeFormat("%d/%m %H");
    target->SetTitle(";time;relative change H-F");

    target->Draw("AP");

    for (size_t i = 1; i < inputs.size(); i++){
        target->Draw("AP");
        TGraph* grInput  = new TGraph( inputs[0].size(), &inputs[0][0], &inputs[i][0]);
        grInput->GetYaxis()->SetRangeUser(-5,5);
        grInput->SetMarkerStyle(22);grInput->SetMarkerSize(0.5);
        grInput->SetMarkerColor(2);grInput->SetLineColor(2);
        grInput->Draw("Psame");
        TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);
        c->Update();
        cin.get();
    }
}

void NeuralNetwork::drawTargetSectorComparison(){
    vector< pair< int, vector<double> > > tableClbAll = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCSecAllNew.dat");     

    vector< vector<double> > arr[6];
    for (size_t s = 0; s < 6; s++){
        arr[s].resize(4,vector<double>(0));
        for (size_t i = 0; i < tableClbAll.size(); i++){
            if (tableClbAll[i].second[0] != s)
                continue;
            arr[s][0].push_back((tableClbAll[i].first));
            arr[s][1].push_back(tableClbAll[i].second[1]);
            arr[s][2].push_back(0);
            arr[s][3].push_back(tableClbAll[i].second[2]);
        }
    }

    TGraphErrors* grClbAll[6];
    double mean0 = 0;
    for (size_t i = 0; i < 6; i++){
        arr[i][0] = timeVectToDateNumbers(arr[i][0]);
        arr[i][1] = runningMeanVector(arr[i][1],20);
        if (i == 0){
            for (size_t j = 0; j < arr[0][0].size(); j++){
                mean0+=arr[0][1][j];
            }
            mean0 = mean0/arr[0][0].size();
        }

        if (i>0){
            for (size_t j = 0; j < arr[i][0].size(); j++){
                arr[i][1][j] = arr[i][1][j]/arr[0][1][j];
            }
        }
    }
    for (size_t i = 0; i < 1; i++){
        //arr[i][1] = normalizeVectorNN(arr[i][1]);
        for (size_t j = 0; j < arr[i][0].size(); j++){
            arr[i][1][j] = arr[i][1][j]/mean0;
        }
    }

    for (size_t i = 0; i < 6; i++){
        grClbAll[i]  = new TGraphErrors( arr[i][0].size(), &arr[i][0][0], &arr[i][1][0], &arr[i][2][0], &arr[i][2][0]);
        grClbAll[i]->SetMarkerStyle(22);
        grClbAll[i]->SetMarkerSize(0.5);
        grClbAll[i]->SetMarkerColor(1);
        grClbAll[i]->GetYaxis()->SetRangeUser(0.75,1.25);
        //if (i == 0)
        //    grClbAll[i]->GetYaxis()->SetRangeUser(-2,2);
        grClbAll[i]->GetXaxis()->SetTimeDisplay(1);
        grClbAll[i]->GetXaxis()->SetTimeFormat("%d/%m");
        grClbAll[i]->SetTitle(Form("sector%d;time;ratio %d/1",i+1,i+1));
        grClbAll[i]->GetXaxis()->SetLabelSize(0.05);
        grClbAll[i]->GetYaxis()->SetLabelSize(0.05);
        grClbAll[i]->GetXaxis()->SetTitleSize(0.08);
        grClbAll[i]->GetYaxis()->SetTitleSize(0.08);
        grClbAll[i]->GetXaxis()->SetTitleOffset(0.65);
        grClbAll[i]->GetYaxis()->SetTitleOffset(0.5);
        if (i == 0)
            grClbAll[i]->SetTitle(Form("sector%d;time;relative change",i+1));
    }

    gStyle->SetTitleFontSize(0.1); // Adjust the value as needed
    gStyle->SetLabelSize(0.45, "xy"); // Adjust the value as needed

    TCanvas* canvas = new TCanvas("canvas", "Canvas Title", 1920, 950);
    canvas->Divide(2, 3); // Divide canvas into 3 rows and 1 column
    for (size_t i = 0; i < 6; i++){
        canvas->cd(i+1);
        TPad* pad = new TPad(Form("pad%d", i + 1), Form("Pad %d", i + 1), 0.0, 0.0, 1.0, 1.0);
        pad->SetBottomMargin(0.15);
        pad->Draw();
        pad->cd();
        pad->SetGridy();
        grClbAll[i]->Draw("AP");
        canvas->Update();
        cin.get();
    }
    delete canvas;

}

void NeuralNetwork::remakeInputDataset(){
    //vector< pair< string, vector<double> > > table = readDBTableFromFile(saveLocation+"info_tables/MDCALL1.dat");
    vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCALL1.dat");
    vector< pair< int, vector<double> > > tableTrig = readClbTableFromFile(saveLocation+"info_tables/trigger_data2.dat");
    vector< pair< int, vector<double> > > tableClb = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDC1New.dat");
    vector< pair< int, vector<double> > > tableClbAll = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCAllNew.dat"); 
    vector< pair< int, vector<double> > > tableClbHH = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCHHNew.dat"); 
    cout << "a" << endl;
    cout << table[15].first << "    " << table[15].second[1] << endl;
    cout << "tableas are read" << endl;


    //vector< vector<double> > dbPars = dbTableToVectorsAveraged(table);
    vector< vector<double> > dbPars = clbTableToVectorsInput(table);
    vector< vector<double> > triggPars = clbTableToVectorsInput(tableTrig);
    vector< vector<double> > meanToT = clbTableToVectorsTarget(tableClb);
    vector< vector<double> > meanToTAll = clbTableToVectorsTarget(tableClbAll);
    vector< vector<double> > meanToTHH = clbTableToVectorsTarget(tableClbHH);

    vector< pair< int, vector<double> > > tableClbAllSec = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCSecAllNew.dat");     

    vector< vector<double> > arr[6];
    for (size_t s = 0; s < 6; s++){
        arr[s].resize(4,vector<double>(0));
    }
    vector< pair<double, double> > sectorsTargets;
    int currentRun = 0;
    for (size_t i = 0; i < tableClbAllSec.size(); i++){
        if (currentRun != tableClbAllSec[i].first){
            sectorsTargets.clear();
        }
        currentRun = tableClbAllSec[i].first;

        if (tableClbAllSec[i].second[0] == sectorsTargets.size()){
            sectorsTargets.push_back(make_pair(tableClbAllSec[i].second[1], tableClbAllSec[i].second[2]));
        }

        if (sectorsTargets.size() == 6){
            for (size_t s = 0; s < 6; s++){
                arr[s][0].push_back(currentRun);
                arr[s][1].push_back(sectorsTargets[s].first);
                arr[s][2].push_back(0);
                arr[s][3].push_back(sectorsTargets[s].second);
            }
        }
    }
    
    cout << dbPars[0][15] << endl;

    //// ____ OUTPUT

    /// _______Running mean calculations
    ///_________________________________
    //meanToT[1] = runningMeanVector(meanToT[1],20);

    for (size_t i = 1; i < dbPars.size(); i++){
        //dbPars[i] = runningMeanVector(dbPars[i],10);
    }
    for (size_t i = 1; i < triggPars.size(); i++){
        //triggPars[i] = runningMeanVector(triggPars[i],100);
    }

    ///_____ Writing to a file
    ofstream ofNN;
    ofNN.open(saveLocation+"nn_input/outNNTest1.dat");
    for (size_t i = 0; i<tableClbAll.size(); i++){
        int run = tableClbAll[i].first;
        double yTarget = meanToTAll[1][i];
        auto p = std::find(dbPars[0].begin(), dbPars[0].end(), run);
        auto pt = std::find(triggPars[0].begin(), triggPars[0].end(), run);
        auto ps = std::find(arr[0][0].begin(), arr[0][0].end(), run);
        //cout << std::distance(dbPars[0].begin(), p) << endl;
        //cout << std::distance(triggPars[0].begin(), pt) << endl;
        if (p != dbPars[0].end() && pt != triggPars[0].end() && ps != arr[0][0].end()){
            //ofNN << dateRunF::runToDateNumber(run);
            ofNN << (run);
            int index = std::distance(dbPars[0].begin(), p);
            for (size_t j = 0; j<dbPars.size()-1; j++){
                if (std::find(additionalFilter.begin(), additionalFilter.end(), j) == additionalFilter.end())
                    ofNN << " " << dbPars[j+1][index];
            }
            int indext = std::distance(triggPars[0].begin(), pt);
            for (size_t j = 0; j<triggPars.size()-1; j++){
                //ofNN << " " << triggPars[j+1][indext];
            }
            int indexs = std::distance(arr[0][0].begin(), ps);
            for (size_t s = 0; s < 6; s++){
                if ((int)arr[s][0][indexs] != run)
                    cout << "run is not the same    "  << (int)arr[s][0][indexs] << " " << run << endl;
                ofNN << " " << arr[s][1][indexs];
            }
            ofNN << endl;
            //ofNN << " " << yTarget << endl;
        }
    }
    ofNN.close();


    /// _____   DRAWING
    dbPars[0] = timeVectToDateNumbers(dbPars[0] );
    meanToT[0]= timeVectToDateNumbers(meanToT[0]);
    meanToTAll[0]= timeVectToDateNumbers(meanToTAll[0]);
    meanToTHH[0]= timeVectToDateNumbers(meanToTHH[0]);
    triggPars[0]= timeVectToDateNumbers(triggPars[0]);

    // scaling
    meanToTAll[1] = normalizeVectorNN(meanToTAll[1]);
    meanToT[1] = normalizeVectorNN(meanToT[1]);
    meanToTHH[1] = normalizeVectorNN(meanToTHH[1]);
    for (size_t i = 1; i < dbPars.size(); i++)
        dbPars[i] =  normalizeVectorNN(dbPars[i]);
    for (size_t i = 1; i < triggPars.size(); i++)
        triggPars[i] =  normalizeVectorNN(triggPars[i]);

    //for (size_t i = 0; i < meanToTAll[0].size(); i++){
    //    meanToT[1][i] = meanToTAll[1][i] !=0 ? meanToT[1][i]/meanToTAll[1][i]-1 : 0. ;
    //    meanToTHH[1][i] = meanToTAll[1][i] !=0 ? meanToTHH[1][i]/meanToTAll[1][i]-1 : 0. ;
    //}



    TDatime da(2022,2,3,15,58,00);
    gStyle->SetTimeOffset(da.Convert());

    TGraphErrors* grClb  = new TGraphErrors( 8000, &meanToT[0][0], &meanToT[1][0], &meanToT[2][0], &meanToT[2][0]);
    TGraphErrors* grClbAll  = new TGraphErrors( 8000, &meanToT[0][0], &meanToTAll[1][0], &meanToT[2][0], &meanToT[2][0]);
    TGraphErrors* grClbHH  = new TGraphErrors( 8000, &meanToT[0][0], &meanToTHH[1][0], &meanToT[2][0], &meanToT[2][0]);

    //grClb->Draw("AP");

    //drawTrainPredictionComparison(meanToT[0], grClb);

    /// Check for stability of the calibration depending on selection criteria
    drawTargetSectorComparison();

    //drawTargetStability(grClbAll,grClb,grClbHH);

    /// check for correlations between input and target for ml
    //drawInputTargetCorrelations(grClbAll,triggPars);
    //drawInputTargetCorrelations(grClbAll,dbPars);

    //auto legend = new TLegend(0.1,0.7,0.48,0.9);
    //legend->AddEntry("grClb","ToT from pp HF","lp");
    //legend->Draw();
}

vector<float> NeuralNetwork::formNNInput(vector<double> db, vector<double> tr){
    tr.erase(tr.begin()+0);

    vector<float> nnInpPars;
    for (size_t i = 0; i < db.size(); i++){
        double x = db[i];
        if (std::find(additionalFilter.begin(), additionalFilter.end(), i) == additionalFilter.end())
            nnInpPars.push_back((float)x);
    }
    for (size_t i = 0; i < tr.size(); i++){
        double x = tr[i];
        if (std::find(additionalFilter.begin(), additionalFilter.end(), i+db.size()) == additionalFilter.end())
            nnInpPars.push_back((float)x);
    }
    return nnInpPars;
}

void NeuralNetwork::retrainModel(){
    gSystem->Exec("python3 /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/trainClassificationTemplate.py");
    gSystem->Exec("python3 /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/predict.py");
}