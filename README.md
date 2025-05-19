
# drogonapp

**Real-Time Calibration Prediction Framework for HADES at GSI/FAIR**

This repository provides a prototype system for fast, automated prediction of calibration constants (e.g., ionization losses) in the HADES detector. It is currently focused on drift chamber gain corrections, using either beam-time or cosmic data, and supports input from EPICS or trigger-based monitoring systems.

The backend is based on the [Drogon C++ web framework](https://github.com/drogonframework/drogon) and integrates Python-based neural networks with a C++ inference and database system.

This project is under active development and **not yet in production state**. Most paths and parameters are hardcoded and must be adapted to your setup. Nevertheless, the code provides a complete working pipeline that can be reused or modified for similar detector environments.


## 🧠 Features

- Fast neural network inference with PyTorch + ONNX
- Input from EPICS via PostgreSQL, trigger data or Oracle db
- Graph-based multi-channel predictions using GConvLSTM for all 24 chambers
- ROOT-based data analysis and histogram generation
- Modular backend in C++ with Python training


## 📁 Code Structure

| Folder | Description |
|--------|-------------|
| `testDrogonApp/function_prediction/` | Python training and prediction logic: `dataHandling.py`, `model.py`, `trainNetwork.py`, `predict.py` |
| `testDrogonApp/source/` + `include/` | C++ backend: inference logic, database interaction, and integration with Drogon |
| `testDrogonApp/build/` | CMake build directory (create manually) |


## What to look for here

There is no production version yet, so you will need to change hardcoded paths and install all dependencies to really try running it.
Thus, for now you can mainly take a look at the source code to get some ideas:
1. testDrogonApp/function_prediction/ folder for python code with the network training. Mainly dataHandling.py, models/model.py, trainNetwork.py and predict.py
2. testDrogonApp/source/ and testDrogonApp/include/ for all the c++ backend code.



## ⚙ Setup & Usage (by chatGPT)

### 🔧 Prerequisites

- Python: `torch`, `torch_geometric`, `pyarrow`, `tqdm`
- C++: Drogon, ROOT, ONNX Runtime, PostgreSQL (`pqxx`) and/or Oracle (OCI)
- CMake for building the C++ components


### 📊 Generating the Calibration Dataset

#### 1. **Offline Target Calibration Table**
- Copy `/lustre/hades/user/vkladov/sub/testBatchFarm/`
- Use `makeList.cc` to list HLD files per day
- Compile and submit `analysisDST.cc` jobs to the batch farm
- Merge `.root` output files with `hadd`
- Use `analyseHists.cc` and `makeCalibTable()` to create final calibration tables

➡ Alternatively, use `fetchNewCalibrations.cc` for automated generation.

#### 2. **Preparing Input Features**
cd testDrogonApp/build
make
cd ../
./testDrogonApp

Before running, update:
 - saveLocation in source/constants.cc
 - Database credentials in connectToDB()
 - Run list path in loadrunlist() (inside functions.cc)
 - Neural network parameters and port in globalSettings.json

You may also need to:
 - Uncomment calls to makeTableWithEpicsData(...)
 - Configure neuralNetwork->remakeInputDataset(...)

#### 3. **Compiling a dataset**
Again do make and ./testDrogonApp with different setup in main:
neuralNetwork->remakeInputDataset(...)


### 📊 Generating the Calibration Dataset

1. Ensure required Python packages are installed
2. Adjust mainPath in trainNetwork.py
3. Update input handling logic in dataHandling.py
4. Run:
  python3 function_prediction/trainNetwork.py
  python3 function_prediction/predict.py


## Contacts
email: v.kladov@gsi.de


## Hand-written guide for basic usage

If you really interested in the package:


Creating MDC calibration dataset: 

    For the target:

1. Copy the directory /lustre/hades/user/vkladov/sub/testBatchFarm/, create the list of raw hld files for each day of experiment with the use of makeList.cc (given the directory of raw files)

2. with analysisDST.cc, Makefile and sendScript_raw_SL.sh ---- create an executable to analyse raw hld files, send jobs to batch farm and get resulting files in a specified directory 

3. Created files contain some directories and a number of histograms in the ones that correspond to run numbers (histograms are now separated by run-sec-mod).

4. "hadd file.root /.../***.root" to merge these files into one, then with /u/vkladov/testBatchFarm/analyseHists.cc and makeCalibTable() get the offline target calibration table.

This in principle can be done with fetchNewCalibrations script (check .cc file).

    For the input:

cd drogonapp/testDrogonApp/build -> make -> cd ../ -> ./testDrogonApp. But you need a "couple" of things to make it running properly...:

1. You need installed drogon, root, python, onnx, pqxx and currently osi for oracle database readout.
    Check CMake list:
    target_link_libraries(${PROJECT_NAME}
        PRIVATE Drogon::Drogon
        ${ROOT_LIBRARIES}
        ${Python_LIBRARIES}
        /home/localadmin_jmesschendorp/onnxruntime-linux-x64-1.14.1/lib/libonnxruntime.so.1.14.1
        ${OCI_LIB} ${OCI_CORE_LIB} ${OCI_COMMON_LIB}
        -lpqxx
    )

2. Change saveLocation in source/constants.cc; 
   Change connectToDB settings to your DB/passwords/usernames; 
   Change runlist path in source/functions.cc loadrunlist...();
   Change variables names, port, input shape in globalSettings.json 

3. Uncomment //controllerBase->epicsManager->makeTableWithEpicsData("new", 443786163, 1e10) or similar (check what you need);

4. make and launch

    Combining to make the input:

1. Change the files locations for input-target and output file in source/neuralNetwork.cc

2. Uncomment //controllerBase->neuralNetwork->remakeInputDataset(false);

3. Make and launch


Training / retraining the network: 

1. You need installed torch, pyarrow, tqdm, torch_geometric in addition to regular python dirs.

2. Change mainPath in the trainNetwork.py file;
   Change data input in the dataHandling file (manageDataset)

3. Just run python3 <>/trainNetwork.py and then <>/predict.py to get the tables with predicted values for evaluation.
