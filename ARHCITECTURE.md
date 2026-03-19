# Architecture

## Overview

This repository is split into two main parts:

- `backend/`: the main backend and ML pipeline glue.
- `reactDevelopment/`: a separate React development frontend.

The backend is the operational core. It combines:

- A Drogon-based C++ HTTP server.
- C++ data acquisition and dataset-generation utilities.
- Python training and inference code under `backend/function_prediction/`.
- ROOT-based analysis and plotting utilities.

The current implementation is prototype-style rather than productized. Many paths, filenames, and runtime assumptions are hardcoded for a specific local environment.

## Main Project Structure

- `backend/main.cc`: primary backend entrypoint.
- `backend/CMakeLists.txt`: CMake build definition for the Drogon app and helper binaries.
- `backend/controllers/`: Drogon HTTP controllers.
- `backend/source/` and `backend/include/`: core C++ services.
- `backend/function_prediction/`: Python model training, dataset handling, and batch prediction.
- `backend/fetchNewRuns.cc`: helper executable for discovering/appending runs.
- `backend/fetchNewCalibrations.cc`: helper executable for generating/appending offline calibration data.
- `backend/analyseHists.cc`: ROOT analysis for offline histogram/calibration workflows.
- `backend/serverData/`: runtime settings and persisted backend state.

## Backend Build Pattern

The backend is built with CMake from `backend/`.

Primary target:

- `testDrogonApp`: built from `backend/main.cc` and linked with Drogon, ROOT, Python, ONNX Runtime, Oracle client libraries, PostgreSQL `pqxx`, and JsonCpp.

Helper targets:

- `fetchNewRuns`
- `fetchNewCalibrations`

Build pattern:

```bash
cd backend
mkdir -p build
cd build
cmake ..
make
```

Important build assumptions from `backend/CMakeLists.txt`:

- ROOT is expected via `root-config`.
- Drogon must be installed and discoverable by CMake.
- Python development headers/libs are required.
- Oracle client include/lib directories are hardcoded.
- ONNX Runtime is linked from a hardcoded absolute path.
- The executable output directory is set to `backend/`, not `backend/build/`.

## Main Backend Run Pattern

The main executable is the Drogon backend process plus some manually selected data/ML actions in `backend/main.cc`.

Typical run pattern:

```bash
cd backend
./testDrogonApp
```

What `main.cc` currently does:

- Sets timezone to `Europe/Berlin`.
- Initializes Python embedding with `Py_Initialize()`.
- Instantiates `NeuralNetwork`.
- Currently calls `neuralNetwork->remakeInputDataset(false);` before server startup.
- Forks a child process that starts the Drogon HTTP server using `backend/config.json`.
- Waits for the child and finalizes Python.

The HTTP server listens on:

- `0.0.0.0:3000`

Static files are served from:

- `backend/static/`

This means the current backend startup is not a clean “serve only” flow. It also triggers dataset-generation behavior unless `main.cc` is changed.

## Training The Network

Training logic lives in `backend/function_prediction/`.

Main files:

- `trainNetwork.py`: training loop, loss definitions, early stopping.
- `dataHandling.py`: dataset loading, reshaping, normalization helpers, graph construction.
- `models/model.py`: model definitions.
- `predict.py`: batch evaluation/prediction against prepared datasets.
- `makePrediction.py`: single-input Python inference path used from C++.
- `config.py`: model type and tensor-shape configuration.

### Training pattern

The intended standalone Python pattern is:

```bash
cd backend
python3 function_prediction/trainNetwork.py
python3 function_prediction/predict.py
```

Observed behavior/patterns in code:

- `Config` defaults to `gConvLSTM`, `sentenceLength=10`, `cellsLength=24`, `channelsLength=9`.
- `dataHandling.load_dataset()` converts prepared tabular data into sequence tensors.
- For `gConvLSTM`, an additional delta-time channel is injected and a graph edge list is generated.
- Training uses PyTorch directly, with validation and early stopping utilities defined in `trainNetwork.py`.
- Batch prediction writes result files under the training data area.

### C++-triggered retraining pattern

The backend also exposes retraining through:

- `GET /maincontroller/retrainModel?remakeInput={bool}`

Controller behavior:

- Optionally calls `neuralNetwork->remakeInputDataset(true)`.
- Then calls `neuralNetwork->retrainModel()`.

Important caveat:

- `NeuralNetwork::retrainModel()` currently executes hardcoded Python commands that point to a different repository path and a script named `trainClassificationTemplate.py`, which is not present in this repository. The standalone Python flow in `backend/function_prediction/` is the more reliable documented path in the current tree.

## Dataset Acquisition And Preparation Patterns

Dataset acquisition is split across several layers.

### 1. Offline target calibration generation

This is the “target” side of supervised learning.

Relevant files:

- `backend/analyseHists.cc`
- `backend/fetchNewCalibrations.cc`

Observed pattern:

- Raw detector files are processed externally into ROOT histograms.
- ROOT outputs are merged and analyzed to derive calibration target tables.
- `fetchNewCalibrations.cc` is intended as a helper to automate parts of this flow and append new target entries into local tables.

This area is strongly tied to HADES-specific filesystem layout and batch-farm assumptions.

### 2. EPICS / DB feature-table generation

This is the main input-feature acquisition path for training and prediction.

Relevant files:

- `backend/source/workWithDB.cc`
- `backend/controllers/MainController.cc`

Primary methods:

- `EpicsDBManager::makeTableWithEpicsData(...)`
- `EpicsDBManager::makeTableWithEpicsDataExtended(...)`
- `EpicsDBManager::makeTableWithEpicsDataContinuousSplit(...)`

Observed pattern:

- Load run boundaries from a run list.
- Query database values over each run interval.
- Serialize per-run feature rows into files in `backend/serverData/info_tables/`.
- Different methods support simple per-run tables, extended historical windows, or minute-sliced continuous tables.

HTTP trigger:

- `GET /maincontroller/readEpicsTable?startRun={1},endRun={2}`

### 3. Trigger-derived tables

Relevant files:

- `backend/include/workWithTrigger.h`
- `backend/controllers/MainController.cc`

HTTP trigger:

- `GET /maincontroller/readTriggTable?startRun={1},endRun={2},formLists={3}`

Observed pattern:

- Optionally create trigger lists.
- Build trigger-based tables through `TriggerDataManager`.

### 4. Final NN input dataset compilation

Relevant file:

- `backend/source/neuralNetwork.cc`

Primary method:

- `NeuralNetwork::remakeInputDataset(bool draw)`

Observed pattern:

- Reads prepared feature tables and target tables from hardcoded files.
- Combines feature data, trigger data, and targets.
- Writes final NN training input files under `serverData/nn_input/`.

This is the main bridge from raw/prepared backend tables into ML-ready training data.

## Run Discovery / Incremental Acquisition Helpers

Two standalone helper executables are defined in CMake:

- `backend/fetchNewRuns.cc`
- `backend/fetchNewCalibrations.cc`

Intended run pattern:

```bash
cd backend
./fetchNewRuns
./fetchNewCalibrations
```

Code comments also suggest background usage such as:

```bash
./fetchNewRuns > outputRun.log 2>&1 &
./fetchNewCalibrations > output.log 2>&1 &
```

Observed roles:

- `fetchNewRuns`: appends newly discovered run IDs into runlist files.
- `fetchNewCalibrations`: derives or appends calibration target entries for new runs.

Both helpers rely on hardcoded directories and filenames and should be treated as environment-specific utilities.

## Real-Time Prediction Launch Pattern

There are two different real-time prediction concepts in the current backend.

### 1. Request/response prediction via HTTP

This is the clearest active path.

Endpoint:

- `GET /maincontroller/getPrediction?run1={run}`

Observed flow:

- `MainController::getPrediction()` calls `ControllerBase::makeNNInputTensor(run1)`.
- `makeNNInputTensor()` reconstructs a sequence window from historical EPICS/DB data and the run list.
- `NeuralNetwork::getPrediction()` forwards the tensor to Python.
- Python imports `makePrediction.py`, loads the PyTorch model, normalizes input, runs inference, denormalizes output, and returns the prediction list to C++.

This is currently Python-embedded inference, not ONNX inference, even though ONNX setup code exists in `NeuralNetwork`.

### 2. Continuous prediction loop

Relevant files:

- `backend/source/continuousPredictor.cc`
- `backend/controllers/MainController.cc`

Endpoint:

- `GET /maincontroller/getNextContinuousPrediction`

Observed state:

- `MainController::getContinuousPrediction()` is wired and returns a “next prediction” based on `ControllerBase::moveForwardCurrentNNInput()`.
- `ContinuousPredictor` exists as a class, but its background loop is effectively disabled with `while(0)`.

So continuous prediction is partially implemented at the controller/data level, but there is no active standalone continuous worker loop enabled in the current code.

## Server Control / Operational Endpoints

Main controller endpoints in `backend/controllers/MainController.h`:

- `/maincontroller/getSettings`
- `/maincontroller/setSettings`
- `/maincontroller/readEpicsTable`
- `/maincontroller/readTriggTable`
- `/maincontroller/retrainModel`
- `/maincontroller/getPrediction`
- `/maincontroller/getNextContinuousPrediction`
- `/maincontroller/getVectorsForTrainingCheck`
- `/maincontroller/getPredictedList`
- `/maincontroller/switchContinuousPredictionLoop`
- `/maincontroller/stopServer`

These endpoints are the primary operational surface for backend control.

## Test Pattern

There is no clear automated backend test suite wired in CMake or the Python backend.

Observed state:

- No `ctest`, `pytest`, or backend unit/integration test harness is configured.
- `backend/controllers/TestCtrl.cc` contains only a minimal sample Drogon controller method.
- The only explicit test file in the repository is the default React test:
  - `reactDevelopment/src/App.test.js`

Practical conclusion:

- Backend validation is currently driven by manual runs, ad hoc scripts, ROOT analysis, and HTTP endpoint invocation rather than automated tests.

## Operational Constraints / Risks

The current backend has several important constraints:

- Hardcoded absolute paths are widespread in C++ and Python.
- Runtime behavior is controlled partly by commenting/uncommenting lines in `main.cc`.
- Training and retraining paths are not fully aligned between Python-only and C++-triggered flows.
- ONNX infrastructure exists, but the active prediction path uses embedded Python and PyTorch weights.
- Helper binaries and dataset generation depend on local HADES/GSI filesystem layout and external data sources.

## Short Summary

The backend architecture is a prototype pipeline with C++ orchestration and Python ML:

- Build the backend with CMake in `backend/`.
- Run `backend/testDrogonApp` to start the Drogon server, noting that current startup also triggers dataset work.
- Acquire feature datasets through EPICS/trigger table builders and helper binaries.
- Build final training input through `NeuralNetwork::remakeInputDataset()`.
- Train primarily through `backend/function_prediction/trainNetwork.py`.
- Serve single-run predictions through Drogon endpoints that call embedded Python inference.
- Treat “continuous prediction” and backend tests as incomplete/work-in-progress areas.
