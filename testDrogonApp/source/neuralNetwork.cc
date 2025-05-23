
#include "../include/neuralNetwork.h"

#include "../include/functions.h"
#include "../include/constants.h"

#include <iomanip>

using namespace std;
using namespace preTrainFunctions;
using namespace vectorFunctions;
using namespace dateRunF;


NeuralNetwork::NeuralNetwork(){
    additionalFilter = {};
    //setupNNPredictions();
    //drawTargetSectorComparison();
}
NeuralNetwork::~NeuralNetwork(){
}

void NeuralNetwork::setupNNPredictions(){
    ifstream fin1;
    meanValues.clear();
	fin1.open((saveLocation+"function_prediction/meanValues.txt").c_str());
    double meanV = 0;
    while (fin1 >> meanV){
        meanValues.push_back(meanV);
    }
    fin1.close();
    stdValues.clear();
    fin1.open((saveLocation+"function_prediction/stdValues.txt").c_str());
    double stdV = 0;
    while (fin1 >> stdV){
        stdValues.push_back(stdV);
    }
    fin1.close();

    env = new Ort::Env();
    std::string model_path = saveLocation+ "function_prediction/tempModel.onnx";
    mSession = new Ort::Session(*env, model_path.c_str(), Ort::SessionOptions{ nullptr });
    
    Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    mInputDims = inputTensorInfo.GetShape();

    Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    mOutputDims = outputTensorInfo.GetShape();

    size_t is = 1;
    for (size_t i = 0; i < mInputDims.size(); i++){
        is = is * mInputDims[i];
        cout << "input DIMS:: ";
        cout << mInputDims[i] << endl;
    }
    inputTensorSize = is;

    size_t os = 1;
    for (size_t i = 0; i < mOutputDims.size(); i++){
        os = os * mOutputDims[i];
        cout << "output DIMS:: ";
        cout << mOutputDims[i] << endl;
    }
    outputTensorSize = os;
}


PyObject* vectorToPyList(vector<float> vec) {
    vector<double> vecDouble;
    for (size_t i = 0; i < vec.size(); i++){
        vecDouble.push_back((double)vec[i]);
    }
    for (size_t i = 0; i < vecDouble.size(); i++){
        int digAbove = 0;
        if (vecDouble[i] != 0) {
            digAbove = static_cast<int>(std::log10(std::abs((int)vecDouble[i]))) + 1;
        }
        int powerCheck = 6-digAbove;
        vecDouble[i] = (int)(vecDouble[i]*pow(10,5))/pow(10,5);
    }

    //cout << "Double Input ";
    //for (size_t i = 0; i < vecDouble.size(); i++){
    //    cout << std::fixed << std::setprecision(5) << vecDouble[i] << ", ";
    //}
    cout << endl;

    PyObject* pyList = PyList_New(vecDouble.size());
    if (!pyList) {
        PyErr_Print();
        return NULL;
    }
    for (size_t i = 0; i < vec.size(); ++i) {
        PyObject* item = PyFloat_FromDouble(vecDouble[i]);
        if (!item) {
            Py_DECREF(pyList);
            PyErr_Print();
            return NULL;
        }
        PyList_SET_ITEM(pyList, i, item);
    }
    return pyList;
}

std::vector<float> pyListToVector(PyObject* pyList) {
    std::vector<float> vec;
    if (!PyList_Check(pyList)) {
        std::cerr << "Input is not a list\n";
        return vec;
    }
    Py_ssize_t size = PyList_Size(pyList);
    vec.reserve(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PyList_GetItem(pyList, i);
        if (!item || !PyFloat_Check(item)) {
            std::cerr << "Invalid list item at index " << i << "\n";
            continue;
        }
        float value = static_cast<float>(PyFloat_AsDouble(item));
        vec.push_back(value);
    }
    return vec;
}

vector<float> NeuralNetwork::getRawPredictionPython(vector<float> normalizedInput){
    //Py_Initialize();
    // Import the sys module
    PyObject* sysModule = PyImport_ImportModule("sys");
    if (sysModule) {
        // Get the sys.path list
        PyObject* sysPath = PyObject_GetAttrString(sysModule, "path");
        if (sysPath) {
            // Append the directory containing your script to sys.path
            PyObject* path = PyUnicode_FromString("/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/");
            PyList_Append(sysPath, path);
            Py_DECREF(path);
            Py_DECREF(sysPath);
        }
        Py_DECREF(sysModule);
    }
    PyObject* pModule = PyImport_ImportModule("makePrediction");
    if (!pModule) {
        PyErr_Print();
        //return 1;
    }

    // Get references to the Python functions
    PyObject* pFirstFunc = PyObject_GetAttrString(pModule, "makeSinglePrediction");
    if (!pFirstFunc || !PyCallable_Check(pFirstFunc)) {
        PyErr_Print();
        //return 1;
    }

    // Construct input for the first Python function
    PyObject* pInputList = vectorToPyList(normalizedInput);

    // Call the first Python function with the constructed input
    PyObject* pResult = PyObject_CallFunctionObjArgs(pFirstFunc, pInputList, NULL);
    if (!pResult) {
        PyErr_Print();
        Py_DECREF(pInputList);
        //return 1;
    }

    // Convert the result to a vector<float>
    std::vector<float> outputVec = pyListToVector(pResult);
    //for (const auto& value : outputVec) {
    //    std::cout << value << " ";
    //}
    //std::cout << std::endl;

    // Clean up
    Py_DECREF(pModule);
    Py_DECREF(pFirstFunc);
    Py_DECREF(pInputList);
    Py_DECREF(pResult);

    // Finalize Python interpreter
    //Py_Finalize();
    return outputVec;
}

vector<float> NeuralNetwork::getPrediction(vector<float> inputTensorValues){
    //int sentenceLength = 5;
    //int inChannels = 7;
    //int nodesLength = 24;
    //int fullLength = sentenceLength * nodesLength * inChannels;
    //if (fullLength != inputTensorValues.size()) 
    //    cout << fullLength << " " << inputTensorValues.size() << endl;

    //for (float x: inputTensorValues){ cout << ", " << x;} cout << endl << endl;

    //for (size_t i = 0; i < sentenceLength; i++){
    //    for (size_t j = 0; j < inChannels; j++){
    //        for (size_t k = 0; k < nodesLength; k++){
                //if (stdValues[j*nodesLength+k] != 0)
                    //inputTensorValues[i*inChannels*nodesLength+j*nodesLength+k] = (float)((inputTensorValues[i*inChannels*nodesLength+j*nodesLength+k]-meanValues[j*nodesLength+k])/stdValues[j*nodesLength+k]);
                //else
                    //inputTensorValues[i*inChannels*nodesLength+j*nodesLength+k] = (float)((inputTensorValues[i*inChannels*nodesLength+j*nodesLength+k]-meanValues[j*nodesLength+k])/1);
    //        }
    //    }
    //}

    //for (float x: inputTensorValues){ cout << ", " << x;} cout << endl << endl << endl;
    
    vector<float> inp = {-0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, 1.15, 1.19213, 1.28335, 1.69697, 1.50619, 1.43025, 0.114225, 1.2106, 0.557263, 0, 0, -0.577436, 1.20882, 1.2026, 1.11423, 0.922802, 1.2895, 1.31063, 1.86656, 1.04046, 1.25881, -0.913003, 1.24629, 1.23368, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, 1.15, 1.19213, 1.28335, 1.69697, 1.50619, 1.43025, 0.114225, 1.2106, 0.557263, 0, 0, -0.577436, 1.20882, 1.2026, 1.11423, 0.922802, 1.2895, 1.31063, 1.86656, 1.04046, 1.25881, -0.913003, 1.24629, 1.23368, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.66185, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, -1.4922, 1.15, 1.19213, 1.28335, 1.69697, 1.50619, 1.43025, 0.114225, 1.2106, 0.557263, 0, 0, -0.577436, 1.20882, 1.2026, 1.11423, 0.922802, 1.2895, 1.31063, 1.86656, 1.04046, 1.25881, -0.913003, 1.24629, 1.23368, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -2.24606, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.510324, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.65991, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, -1.5022, 1.01647, 1.22912, 1.08716, 1.45236, 1.41305, 1.36161, 0.124045, 1.14602, 0.576243, 0, 0, -0.649008, 1.21491, 1.2118, 1.10974, 0.907692, 1.3059, 1.28224, 1.83982, 1.02719, 1.25147, -0.889066, 1.2379, 1.2288, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -2.23603, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.627079, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.655752, -0.00819975, -0.0103306, -0.00819975, 0.00905479, 0.00596417, 0.0414289, -1.69931, -2.39766, -0.262997, -1.67573, -1.53179, -1.70409, 0, 0, 0, 0, 0, 0, 0.0106431, 0, 0, 0, 0, 0, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, -1.43841, 1.11889, 1.24665, 1.2449, 1.48192, 1.58487, 1.47091, 0.0309097, 1.2089, 0.552984, 0, 0, -0.623491, 1.21062, 1.21796, 1.11977, 0.892438, 1.29838, 1.28405, 1.823, 1.0153, 1.25138, -0.945185, 1.2379, 1.2335, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, 22.0054, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -2.21249, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924, -0.416924};
    vector<float> outputTensorValues = getRawPredictionPython(inputTensorValues);

    //const char* mInputName[] = {"input"};
    //const char* mOutputName[] = {"output"};
    //Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    //Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),inputTensorSize,mInputDims.data(),mInputDims.size());
    //vector<float> outputTensorValues(outputTensorSize);
    //Ort::Value outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize,mOutputDims.data(), mOutputDims.size());
    //mSession->Run(Ort::RunOptions{nullptr}, mInputName,&inputTensor, 1, mOutputName,&outputTensor, 1);


    //for (float x: outputTensorValues){ cout << " " << x;} cout << endl << endl;

    //outputTensorValues = vectorFunctions::transposeTensorDimensions(outputTensorValues, mOutputDims, {1,0});  // Swap the dimensions (onnx has column-major ordering and I don't specify order of the output explicitly)

    //onnxruntime::Transpose transposer(outputTensorValues.data(), mOutputDims);
    //std::vector<size_t> permutation = {1, 0};
    //transposer.ApplyToOutput(outputValues.data(), permutation);              
    
    //vector<float> outputProbs;
    //for (size_t i = 0; i < outputTensorSize; i++)
    //    outputProbs.push_back(outputTensorValues[i]);
    //cout << "output size is " << outputTensorValues.size() << " " << inputTensorSize << endl;

    vector<float> result = outputTensorValues;
    //for (size_t i = 0; i < nodesLength; i++){
        //cout << outputTensorValues[outputTensorValues.size()-nodesLength+i]*stdValues[stdValues.size()-2*nodesLength+i] + meanValues[meanValues.size()-2*nodesLength+i] << ", ";
        //cout << outputTensorValues[outputTensorValues.size()-nodesLength+i] << endl;
        //cout << outputTensorValues[(i+1)*sentenceLength-1]*stdValues[stdValues.size()-nodesLength+i] + meanValues[meanValues.size()-nodesLength+i] << endl;
        //cout <<          outputTensorValues[outputTensorValues.size()-nodesLength+i]*stdValues[stdValues.size()-nodesLength+i] + meanValues[meanValues.size()-nodesLength+i] << endl;
        //result.push_back(outputTensorValues[(i+1)*sentenceLength-1]*stdValues[stdValues.size()-nodesLength+i] + meanValues[meanValues.size()-nodesLength+i]);
        //result.push_back(outputTensorValues[(i+1)*sentenceLength-1]);
        //result.push_back(outputTensorValues[outputTensorValues.size()-nodesLength+i]*stdValues[stdValues.size()-2*nodesLength+i] + meanValues[meanValues.size()-2*nodesLength+i]);
    //    result.push_back(outputTensorValues[i]);
    //}
    //cout << "result " << result[0] << endl;
    //for (float x: result){ cout << ", " << x;} cout << endl << endl;
    //cout << endl;
    return result;
    //return outputTensorValues[outputTensorValues.size()-1]*stdValues[stdValues.size()-1] + meanValues[meanValues.size()-1];
}

void NeuralNetwork::drawTargetStability(TGraphErrors* targetsAll, TGraphErrors* targetsPP, TGraphErrors* targetsHH){
    targetsAll->SetMarkerStyle(23);
    targetsAll->SetMarkerSize(0.4);
    //targetsAll->SetMarkerColor(4);
    targetsAll->GetYaxis()->SetRangeUser(5,6.7);
    //targetsAll->GetYaxis()->SetRangeUser(0.75,1.25);
    targetsAll->GetXaxis()->SetTimeDisplay(1);
    targetsAll->GetXaxis()->SetTimeFormat("%d/%m");
    targetsAll->SetTitle(";date;-dE/dx [MeV g^{-1} cm^{2}]");

    targetsHH->SetMarkerStyle(23);
    targetsHH->SetMarkerSize(0.4);
    //targetsHH->SetMarkerColor(4);
    targetsHH->GetYaxis()->SetRangeUser(1.6,2.03);
    //targetsHH->GetYaxis()->SetRangeUser(0.75,1.25);
    targetsHH->GetXaxis()->SetTimeDisplay(1);
    targetsHH->GetXaxis()->SetTimeFormat("%d/%m");
    targetsHH->SetTitle(";date;-dE/dx [MeV g^{-1} cm^{2}]");

    targetsPP->SetMarkerStyle(23);
    targetsPP->SetMarkerSize(0.4);
    //targetsPP->SetMarkerColor(4);
    //targetsPP->SetLineColor(4);
    //targetsPP->SetLineWidth(1);
    targetsPP->SetName("grClb");
    targetsPP->GetYaxis()->SetRangeUser(12.5,18.6);
    //targetsPP->GetYaxis()->SetRangeUser(0.75,1.25);
    targetsPP->GetXaxis()->SetTimeDisplay(1);
    targetsPP->GetXaxis()->SetTimeFormat("%d/%m");
    targetsPP->SetTitle(";date;-dE/dx [MeV g^{-1} cm^{2}]");


        targetsAll->GetXaxis()->SetLabelSize(0.05);
        targetsAll->GetYaxis()->SetLabelSize(0.05);
        targetsAll->GetXaxis()->SetTitleSize(0.07);
        targetsAll->GetYaxis()->SetTitleSize(0.07);
        targetsAll->GetXaxis()->SetTitleOffset(0.65);
        targetsAll->GetYaxis()->SetTitleOffset(0.5);

        targetsHH->GetXaxis()->SetLabelSize(0.05);
        targetsHH->GetYaxis()->SetLabelSize(0.05);
        targetsHH->GetXaxis()->SetTitleSize(0.07);
        targetsHH->GetYaxis()->SetTitleSize(0.07);
        targetsHH->GetXaxis()->SetTitleOffset(0.65);
        targetsHH->GetYaxis()->SetTitleOffset(0.5);

        targetsPP->GetXaxis()->SetLabelSize(0.05);
        targetsPP->GetYaxis()->SetLabelSize(0.05);
        targetsPP->GetXaxis()->SetTitleSize(0.07);
        targetsPP->GetYaxis()->SetTitleSize(0.07);
        targetsPP->GetXaxis()->SetTitleOffset(0.65);
        targetsPP->GetYaxis()->SetTitleOffset(0.5);


    TCanvas* canvas = new TCanvas("canvas", "Canvas Title", 1220, 980);
    canvas->Divide(1, 3); // Divide canvas into 3 rows and 1 column
    canvas->cd(1);
    TPad* pad1 = new TPad("pad1", "pad1", 0.0, 0.0, 1.0, 1.0);
        pad1->SetBottomMargin(0.15);
        pad1->Draw();
        pad1->cd();
    targetsAll->Draw("AP");
    canvas->cd(2);
    TPad* pad2 = new TPad("pad2", "pad2", 0.0, 0.0, 1.0, 1.0);
        pad2->SetBottomMargin(0.15);
        pad2->Draw();
        pad2->cd();
    targetsPP->Draw("AP");
    canvas->cd(3);
    TPad* pad3 = new TPad("pad3", "pad3", 0.0, 0.0, 1.0, 1.0);
        pad3->SetBottomMargin(0.15);
        pad3->Draw();
        pad3->cd();
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
    target->GetXaxis()->SetTimeFormat("%d/%m");
    target->GetXaxis()->SetLabelSize(0.03);
    target->SetTitle(";date;normalized fluctuations");

    TCanvas* canvas = new TCanvas("canvas", "Canvas Title", 1520, 950);

    target->Draw("AP");
    target->GetXaxis()->SetNdivisions(19);

    for (size_t i = 1; i < inputs.size(); i++){
        target->Draw("AP");
        cout << inputs[0][0] << endl;
        TGraph* grInput  = new TGraph( 11000, &inputs[0][0], &inputs[i][0]);
        grInput->GetYaxis()->SetRangeUser(-5,5);
        grInput->SetMarkerStyle(22);grInput->SetMarkerSize(0.5);
        grInput->SetMarkerColor(2);grInput->SetLineColor(2);
        grInput->SetName("grInput");
        target->SetName("target");
        grInput->Draw("Psame");
        //TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->At(0);

        auto legend = new TLegend(0.1,0.75,0.4,0.9);
        legend->AddEntry("grInput","Atmospheric pressure","lp");
        legend->AddEntry("target","average dE/dx","lp");
        legend->Draw();
        canvas->Update();
        cin.get();
    }
}

void NeuralNetwork::drawInputTargetCorrelations(vector< vector<double> > targets, vector< vector<double> > inputs){

    vector<double> input, target, inputErr, targetErr;
    //TProfile* prof = new TProfile("prof","profile",200,990,1040);
    //TH2* h2 = new TH2F("h2","ionization losses vs atmospheric pressure sector 1;pressure [Pa];average ionization losses [MeV*cm^{2}/g]",100,990,1040,100,4,8);
    TProfile* prof = new TProfile("prof","profile",200,30,50);
    TH2* h2 = new TH2F("h2","ionization losses vs overpressure sector 4;overpressure [Pa];average ionization losses [MeV*cm^{2}/g]",100,30,50,100,4,8);
    // fill arrays 
    for (size_t i = 0; i<inputs[0].size(); i++){
        double run = inputs[0][i];
        //cout << run << "    " << targets[0][15] << endl;
        auto ps = std::find(targets[0].begin(), targets[0].end(), run);
        if (ps != targets[22].end()){
            int indexs = std::distance(targets[0].begin(), ps);
            input.push_back(inputs[1][i]);
            inputErr.push_back(0);
            target.push_back(targets[1][indexs]);
            targetErr.push_back(targets[3][indexs]);
            prof->Fill(inputs[22][i],targets[1][indexs]);
            h2->Fill(inputs[22][i],targets[1][indexs]);
            //cout << input[input.size()-1] << "  " << inputErr[input.size()-1] << "  " << target[input.size()-1] << "  " << targetErr[input.size()-1] << "  " << endl;
        }
    }

    //cout << inputs[0].size() << 

    TGraphErrors* gr  = new TGraphErrors( input.size(), &input[0], &target[0], &inputErr[0], &targetErr[0]);

    gr->SetMarkerStyle(22);
    gr->SetMarkerSize(0.5);
    gr->SetMarkerColor(4);
    gr->SetLineColor(4);
    gr->SetLineWidth(1);
    gr->SetName("gr");
    gr->GetYaxis()->SetRangeUser(4,8);
    prof->GetYaxis()->SetRangeUser(4,8);
    h2->SetStats(0);

    gr->Draw("AP");
    TCanvas* c = new TCanvas("canvas", "Canvas Title", 950, 950);
    c->Update();
    cin.get();
    prof->Draw();
    c->Update();
    cin.get();
    h2->Draw("colz");
    c->Update();
    cin.get();

}


void NeuralNetwork::drawTargetSectorComparison(){
    cout << "drawing target sector divided..." << endl;
    vector< pair< int, vector<double> > > tableClbAll = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCSecModPreciseFit1.dat");     
    cout << " a  " << endl;
    vector< vector<double> > arr[6];
    for (size_t s = 0; s < 6; s++){
        arr[s].resize(4,vector<double>(0));
        for (size_t i = 0; i < tableClbAll.size(); i++){
            if (tableClbAll[i].second[1] != s && tableClbAll[i].second[0] != 0)
                continue;
            arr[s][0].push_back((tableClbAll[i].first));
            arr[s][1].push_back(tableClbAll[i].second[2]);
            arr[s][2].push_back(0);
            arr[s][3].push_back(tableClbAll[i].second[3]);
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
        grClbAll[i]->SetTitle(Form("sector%zu;time;ratio %zu/1",i+1,i+1));
        grClbAll[i]->GetXaxis()->SetLabelSize(0.05);
        grClbAll[i]->GetYaxis()->SetLabelSize(0.05);
        grClbAll[i]->GetXaxis()->SetTitleSize(0.08);
        grClbAll[i]->GetYaxis()->SetTitleSize(0.08);
        grClbAll[i]->GetXaxis()->SetTitleOffset(0.65);
        grClbAll[i]->GetYaxis()->SetTitleOffset(0.5);
        if (i == 0)
            grClbAll[i]->SetTitle(Form("sector%zu;time;relative change",i+1));
    }

    gStyle->SetTitleFontSize(0.1); // Adjust the value as needed
    gStyle->SetLabelSize(0.45, "xy"); // Adjust the value as needed

    TCanvas* canvas = new TCanvas("canvas", "Canvas Title", 1920, 950);
    canvas->Divide(2, 3); // Divide canvas into 3 rows and 1 column
    for (size_t i = 0; i < 6; i++){
        canvas->cd(i+1);
        TPad* pad = new TPad(Form("pad%zu", i + 1), Form("Pad %zu", i + 1), 0.0, 0.0, 1.0, 1.0);
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

TCanvas* NeuralNetwork::drawTargetDimensionsComp(std::map< int, vector<double> > meanToTModSec, vector<int> shape){
    // draw in a loop
    // share_pointers (smart_pointers)

    int outShapeLength = 1; for (int x: shape){ outShapeLength*=x; }
    const int shapeLength = outShapeLength;

    TString title[outShapeLength];
    for (size_t i = 0; i < shapeLength; i++){
        for (size_t j = 0; j < shape.size(); j++){
            title[i] += Form("dim%zu %zu,",j+1, i+1);
        }
    }
    

    // Get vectors of runs and values
    vector<double> runArr;
    vector<double> valueArr[shapeLength];
    for (auto rowI: meanToTModSec){
        runArr.push_back(rowI.first);
        for (size_t s = 0; s < shapeLength; s++){
            valueArr[s].push_back(rowI.second[s]);
        }
    }

    // Convert runs to time axis
    runArr = timeVectToDateNumbers(runArr);

    // Normalization
    TGraph* grClbAll[shapeLength];
    double mean0 = 0;
    for (size_t i = 0; i < shapeLength; i++){
        valueArr[i] = runningMeanVector(valueArr[i],20);
        if (i == 0){
            for (size_t j = 0; j < valueArr[i].size(); j++){
                mean0+=valueArr[0][j];
            }
            mean0 = mean0/valueArr[0].size();
        }
        if (i>0){
            for (size_t j = 0; j < valueArr[i].size(); j++){
                valueArr[i][j] = valueArr[i][j]/valueArr[0][j];
            }
        }
    }
    for (size_t i = 0; i < 1; i++){
        //valueArr[i] = normalizeVectorNN(valueArr[i]);
        for (size_t j = 0; j < valueArr[i].size(); j++){
            valueArr[i][j] = valueArr[i][j]/mean0;
        }
    }

    // Just drawing below
    for (size_t i = 0; i < shapeLength; i++){
        grClbAll[i]  = new TGraph( runArr.size(), &runArr[0], &valueArr[i][0]);
        grClbAll[i]->SetMarkerStyle(22);
        grClbAll[i]->SetMarkerSize(0.5);
        grClbAll[i]->SetMarkerColor(1);
        grClbAll[i]->GetYaxis()->SetRangeUser(0.55,1.45);
        //if (i == 0)
        //    grClbAll[i]->GetYaxis()->SetRangeUser(-2,2);
        grClbAll[i]->GetXaxis()->SetTimeDisplay(1);
        grClbAll[i]->GetXaxis()->SetTimeFormat("%d/%m");
        //grClbAll[i]->SetTitle(Form("sector%d;time;ratio %d/1",i+1,i+1));
        grClbAll[i]->SetTitle(title[i]);
        grClbAll[i]->GetXaxis()->SetTitle("time");
        grClbAll[i]->GetYaxis()->SetTitle("ratio / 1");
        grClbAll[i]->GetXaxis()->SetLabelSize(0.05);
        grClbAll[i]->GetYaxis()->SetLabelSize(0.05);
        grClbAll[i]->GetXaxis()->SetTitleSize(0.08);
        grClbAll[i]->GetYaxis()->SetTitleSize(0.08);
        grClbAll[i]->GetXaxis()->SetTitleOffset(0.65);
        grClbAll[i]->GetYaxis()->SetTitleOffset(0.5);
        if (i == 0)
            //grClbAll[i]->SetTitle(Form("sector%d;time;relative change",i+1));
            grClbAll[i]->GetYaxis()->SetTitle("relative change");
    }

    gStyle->SetTitleFontSize(0.1); // Adjust the value as needed
    gStyle->SetLabelSize(0.45, "xy"); // Adjust the value as needed

    TCanvas* canvas = new TCanvas("canvas", "Canvas Title", 1920, 950);
    //canvas->Divide(2, 3); // Divide canvas into 3 rows and 1 column
    for (size_t i = 0; i < shapeLength; i++){
        //canvas->cd(i+1);
        TPad* pad = new TPad(Form("pad%zu", i + 1), Form("Pad %zu", i + 1), 0.0, 0.0, 1.0, 1.0);
        pad->SetBottomMargin(0.15);
        pad->Draw();
        pad->cd();
        pad->SetGridy();
        grClbAll[i]->Draw("AP");
        canvas->Update();
        //cin.get();
    }
    return canvas;
    //delete canvas;
}


template <typename T>
typename std::map<int, T>::const_iterator find_closest(const std::map<int, T>& m, int key) {
    if (m.empty()) return m.end();

    auto lower = m.upper_bound(key);

    if (lower == m.begin()) {
        return lower;
    }
    if (lower == m.end()) {
        return std::prev(lower);
    }

    auto prev = std::prev(lower);
    if (std::abs(prev->first - key) <= std::abs(lower->first - key)) {
        return prev;
    } else {
        return lower;
    }
}

template <typename T>
typename std::map<int, T>::const_iterator find_next(const std::map<int, T>& m, int key) {
    if (m.empty()) return m.end();

    auto lower = m.upper_bound(key);

    if (lower == m.end()) {
        return std::prev(lower);
    }

    return lower;
}
template <typename T>
int find_next_run(const std::map<int, T>& m, int key) {
    if (m.empty()) return 0;

    auto lower = m.upper_bound(key);

    if (lower == m.end()) {
        return std::prev(lower)->first + 600;
    }

    return lower->first-150;
}


int NeuralNetwork::remakeInputDataset(bool draw){
    vector<int> outShape = {4,6};
    int outShapeLength = 1; for (int x: outShape){ outShapeLength*=x; }

    vector< pair<int,int> > runBorders = loadrunlistWithEnds(0, 1e11); //444140006

    //vector< pair< string, vector<double> > > table = readDBTableFromFile(saveLocation+"info_tables/MDCALL1.dat");
    cout << "starting reading tables" << endl;
    //vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCALL.dat");
    //vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCALL12.dat");
    //vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCModSec1.dat");
    
    //vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCModSecPrecise.dat");
    //vector< pair< int, vector<double> > > table = readClbTableFromFileExtended(saveLocation+"info_tables/MDCModSecPreciseExtended.dat");  // ! main stuff feb22 the best now
    //vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCModSecPreciseFeb22ends9.dat");          // ! test feb22 with averaging until run ends and 9 inputs + CO2 concentration split (1,2, 34)
    //vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCModSecPreciseCosmic25VaryHV1.dat");
    vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCModSecPreciseCosmic25_106VaryHVends9.dat");

    //vector< pair< int, vector<double> > > table = readClbTableFromFile(saveLocation+"info_tables/MDCModSec1zxc.dat");
    cout << "a" << endl;
    vector< pair< int, vector<double> > > tableTrig = readClbTableFromFile(saveLocation+"info_tables/trigger_data2.dat");
    cout << "b" << endl;
    //vector< pair< int, vector<double> > > tableClb = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDC1New.dat");
    //vector< pair< int, vector<double> > > tableClbAll = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCAllNew.dat"); 
    //vector< pair< int, vector<double> > > tableClbHH = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCHHNew.dat"); 
    //cout << table[15].first << "    " << table[15].second[1] << endl;
    cout << "tableas are read" << endl;


    //vector< vector<double> > dbPars = dbTableToVectorsAveraged(table);
    vector< vector<double> > dbPars = clbTableToVectorsInput(table);
    vector< vector<double> > triggPars = clbTableToVectorsInput(tableTrig);
    //vector< vector<double> > meanToT = clbTableToVectorsTarget(tableClb);
    //vector< vector<double> > meanToTAll = clbTableToVectorsTarget(tableClbAll);
    //vector< vector<double> > meanToTHH = clbTableToVectorsTarget(tableClbHH);

    cout << "c" << endl;
    //vector< pair< int, vector<double> > > tableClbAllSec = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCSecAllNew.dat");
    //vector< pair< int, vector<double> > > tableClbModSec = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCSecModNew.dat");
    //vector< pair< int, vector<double> > > tableClbModSec = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCSecModzxc.dat");
    //vector< pair< int, vector<double> > > tableClbModSec = readClbTableFromFile(saveLocation+"info_tables/run-mean_dEdxMDCSecModPreciseFit2.dat");   //! main stuff feb22 the best now
    //vector< pair< int, vector<double> > > tableClbModSec = readClbTableFromFile("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOut.dat");
    //vector< pair< int, vector<double> > > tableClbModSec = readClbTableFromFile("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOutSeparative4.dat");     //10 minutes last
    //vector< pair< int, vector<double> > > tableClbModSec = readClbTableFromFile("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOutSeparative3.dat");   //30 minutes last
    //vector< pair< int, vector<double> > > tableClbModSec = readClbTableFromFile("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOutSeparative2properPars.dat");   //56 full runs last 16 05
    vector< pair< int, vector<double> > > tableClbModSec = readClbTableFromFile("/home/localadmin_jmesschendorp/gsiWorkFiles/analysisUlocal/testOutSeparative100_2properPars.dat");   //106 full runs last 16 05
    
    cout << "d" << endl;
    std::map< int, vector<double> > meanToTModSec = clbTableToVectorsTarget(tableClbModSec, outShape);

    /*vector< vector<double> > arr[6];
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
    }*/
    

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
    int runsWritten = 0;
    bool writeFile = true;
    if (writeFile){
        ofstream ofNN;
        //ofNN.open(saveLocation+"nn_input/outNNFitTargetExtended.dat"); //Test1
        //ofNN.open(saveLocation+"nn_input/outNNFitTargetRunEnds9pars.dat"); //Test1
        //ofNN.open(saveLocation+"nn_input/outNNFitTargetCosmic25_10mins.dat"); //cosmic 10 minutes
        ofNN.open(saveLocation+"nn_input/outNNFitTargetCosmic25_106_9.dat"); //cosmic 30 minutes
        for (auto entry: meanToTModSec){
            int run = entry.first;
            //cout << run << endl;
            auto p = std::find(dbPars[0].begin(), dbPars[0].end(), (double)run);
            auto pt = std::find(triggPars[0].begin(), triggPars[0].end(), (double)run);
            auto ps = meanToTModSec.find(run);
            //cout << "a  ";
            if (p != dbPars[0].end()){
                //cout << "b  ";
                /// getting dpPars run indices which are between the current run target and the next one for the case when dbPars are calculated for each minute e.g.
                /// valid mostly for cosmic
                //int nextTargetRun = find_next_run(meanToTModSec,run);
                //int nextTargetRun = (int)(meanToTModSec[run][0]);    // for the case with end run explicit in target table
                
                int nextTargetRun = 0;    // for the case with end run in a runList file
                auto itEndRun = std::find_if(runBorders.begin(), runBorders.end(), [run](const std::pair<int, int>& p) { return p.first == run; });
                if (itEndRun != runBorders.end()) {
                    nextTargetRun = itEndRun->second;
                }
                else {
                    continue; // Skip if the run is not found in the runBorders
                }
                //cout << "c" << endl;


                vector<int> indicesToAverage;
                /* 
                vector< vector<int> > indicesToAverage2;
                vector<int> separationsForInterpolation2;
                for (int i = run; i <= nextTargetRun; i += 180) {
                    separationsForInterpolation2.push_back(i);
                }
                if (separationsForInterpolation2.size()<1)
                    continue;
                for (size_t j = 0; j < separationsForInterpolation2.size()-1; j++){
                    vector<int> subRunInterpToAverage;
                    for (size_t i = 0; i < dbPars[0].size(); i++ ){
                        if (dbPars[0][i] >= separationsForInterpolation2[j] && dbPars[0][i] < separationsForInterpolation2[j+1]){
                            subRunInterpToAverage.push_back(i);
                        }
                    }
                    indicesToAverage2.push_back(subRunInterpToAverage);
                }
                */

                for (size_t i = 0; i < dbPars[0].size(); i++ ){
                    if (dbPars[0][i]+30 >= run && dbPars[0][i]+30 < nextTargetRun){
                        indicesToAverage.push_back(i);
                    }
                }
                //if (indicesToAverage.size() == 0 || nextTargetRun-run < 200)
                if (nextTargetRun-run < 1000)   //20 for beam time
                    continue;

                //cout << run << endl;
                ofNN << run;
                //ofNN << (int)((run+nextTargetRun)/2.);

                int index = std::distance(dbPars[0].begin(), p);
                //for (size_t k = 0; k < separationsForInterpolation2.size()-1; k++){    // loop needed for the interpolation in cosmic data
                    //ofNN << separationsForInterpolation2[k];
                    for (size_t j = 0; j<dbPars.size()-1; j++){
                        //if (std::find(additionalFilter.begin(), additionalFilter.end(), j) == additionalFilter.end()){
                            ofNN << " " << std::fixed << std::setprecision(5) << dbPars[j+1][index];
                            double averageToWrite = 0;
                            for (int ind: indicesToAverage){
                                averageToWrite += dbPars[j+1][ind]/indicesToAverage.size();
                            }
                            //ofNN << " " << std::fixed << std::setprecision(5) << averageToWrite;
                        //}
                    }
                //}

                //int indext = std::distance(triggPars[0].begin(), pt);
                //for (size_t j = 0; j<triggPars.size()-1; j++){
                    //ofNN << " " << std::fixed << std::setprecision(5) << triggPars[j+1][indext];
                //}

                //int indexs = std::distance(meanToTModSec.begin(), ps);
                for (size_t s = 0; s < 2*outShapeLength; s++){
                    ofNN << " " << std::fixed << std::setprecision(7) << meanToTModSec[run][s];   // +1 for end run
                }
                //for (size_t s = outShapeLength; s < 2*outShapeLength; s++){
                //    ofNN << " " << std::fixed << std::setprecision(7) << meanToTModSec[run][s];   // errors of target
                //}
                ofNN << endl;
                runsWritten+=1;
            }
        }
        ofNN.close();
    }


    /// _____   DRAWING
    if (!draw)
        return runsWritten;
    dbPars[0] = timeVectToDateNumbers(dbPars[0]);
    //meanToT[0]= timeVectToDateNumbers(meanToT[0]);
    //meanToTAll[0]= timeVectToDateNumbers(meanToTAll[0]);
    //meanToTHH[0]= timeVectToDateNumbers(meanToTHH[0]);
    triggPars[0]= timeVectToDateNumbers(triggPars[0]);

    // scaling
    //meanToTAll[1] = normalizeVectorNN(meanToTAll[1]);
    //meanToT[1] = normalizeVectorNN(meanToT[1]);
    //meanToTHH[1] = normalizeVectorNN(meanToTHH[1]);
    for (size_t i = 1; i < dbPars.size(); i++)
        dbPars[i] =  normalizeVectorNN(dbPars[i]);
    for (size_t i = 1; i < triggPars.size(); i++)
        triggPars[i] =  normalizeVectorNN(triggPars[i]);

    //double meanAll = normalizeVector(meanToTAll[1]);
    //double mean0 = normalizeVector(meanToT[1]);
    //double meanHH = normalizeVector(meanToTHH[1]);


    for (size_t i = 0; i< dbPars[0].size(); i++){
        dbPars[1][i] = dbPars[1][i]*(-0.7)-0.1;
    }


    TDatime da(2022,2,3,15,58,00);
    gStyle->SetTimeOffset(da.Convert());

    //cout << meanToTAll[0][0] << endl;
    //cout << meanToTAll[0][100] << endl << endl;

    //TGraphErrors* grClb  = new TGraphErrors( 12000, &meanToT[0][0], &meanToT[1][0], &meanToT[2][0], &meanToT[2][0]);
    //TGraphErrors* grClbAll  = new TGraphErrors( 12000, &meanToTAll[0][0], &meanToTAll[1][0], &meanToT[2][0], &meanToT[2][0]);
    //TGraphErrors* grClbHH  = new TGraphErrors( 12000, &meanToT[0][0], &meanToTHH[1][0], &meanToT[2][0], &meanToT[2][0]);

    //grClb->Draw("AP");

    //drawTrainPredictionComparison(meanToT[0], grClb);

    /// Check for stability of the calibration depending on selection criteria
    //TCanvas* cnvs = drawTargetDimensionsComp(meanToTModSec, outShape);

    //drawTargetStability(grClbAll,grClb,grClbHH);

    /// check for correlations between input and target for ml
    //drawInputTargetCorrelations(grClbAll,triggPars);
    //drawInputTargetCorrelations(grClbAll,dbPars);
    //drawInputTargetCorrelations(arr[3],dbPars);

    //auto legend = new TLegend(0.1,0.7,0.48,0.9);
    //legend->AddEntry("grClb","ToT from pp HF","lp");
    //legend->Draw();

    return runsWritten;
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

vector<float> NeuralNetwork::formNNInput(vector<double> db){

    vector<float> nnInpPars; // nodes * inChan
    for (size_t i = 0; i < db.size(); i++){
        double x = db[i];
        if (std::find(additionalFilter.begin(), additionalFilter.end(), i) == additionalFilter.end())
            nnInpPars.push_back((float)x);
    }
    return nnInpPars;
}

void NeuralNetwork::retrainModel(){
    gSystem->Exec("python3 /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/trainClassificationTemplate.py");
    gSystem->Exec("python3 /home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/function_prediction/predict.py");
}