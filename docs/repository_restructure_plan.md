# Repository Restructure Plan

This note evaluates the current repository against a target architecture that keeps scientific and performance-critical logic in C++, while moving orchestration and web/API responsibilities to Python.

The assessment is based on the current repository contents, especially [ARHCITECTURE.md](/home/localadmin_jmesschendorp/gsiWorkFiles/realTimeCalibrations/ARHCITECTURE.md), `backend/`, and `reactDevelopment/`.

## Section 1 - Current repository assessment

### Existing repository layout, very brief

The repository currently has:

- `backend/` as a mixed C++/Python backend.
- `reactDevelopment/` as a separate React frontend project.
- top-level docs-like files `README.md`, `ARHCITECTURE.md`, and `AGENTS.md`.

Inside `backend/`, several concerns are combined:

- Drogon HTTP server code in `backend/main.cc` and `backend/controllers/`.
- C++ scientific/data-processing logic in `backend/source/` and `backend/include/`.
- Python ML code in `backend/function_prediction/`.
- helper executables such as `backend/fetchNewRuns.cc` and `backend/fetchNewCalibrations.cc`.
- runtime/config/data artifacts in `backend/serverData/`.
- static and built frontend assets in `backend/static/` and `backend/react/`.

### Parts already in a relatively good state

- `backend/function_prediction/`
  - This is already a recognizable Python ML/training area with a mostly coherent grouping of dataset handling, model code, training, and prediction.
- `backend/source/` and `backend/include/`
  - These contain a meaningful C++ domain core for EPICS/trigger access, ROOT-heavy processing, neural-network input construction, and calibration-related workflows.
- `backend/analyseHists.cc`
  - This is consistent with the desired C++ scientific-core direction because it is ROOT-heavy and detector-domain-specific.
- `backend/fetchNewRuns.cc` and `backend/fetchNewCalibrations.cc`
  - Even though they are fragile, the idea of standalone tools for acquisition pipelines is structurally compatible with the target design.
- `backend/serverData/globalSettings.json`
  - This indicates that the project already has a concept of externalized settings instead of keeping everything in code.

### Parts that are problematic

- `backend/main.cc`
  - Startup mixes process bootstrap, dataset generation, Python embedding, and web server lifecycle.
  - The current executable is not a clean service boundary.
- `backend/controllers/` and Drogon integration
  - Request handling is tightly coupled to core classes and local state.
  - This does not match the target direction of moving web/API responsibilities to Python.
- `backend/source/neuralNetwork.cc` and `backend/include/neuralNetwork.h`
  - The class combines several responsibilities: feature shaping, plotting, inference, dataset compilation, and training triggers.
  - It also embeds Python directly and still carries partial ONNX code and hardcoded absolute include paths.
- `backend/source/workWithOracleDB.cc`
  - Oracle access is incomplete, hardcoded, and not isolated cleanly from the rest of the backend.
  - This aligns with the stated Oracle integration problem.
- `backend/CMakeLists.txt`
  - Build logic mixes app/server concerns with helper tools.
  - It hardcodes Oracle and ONNX locations and directly links the web stack into the main backend build.
- `backend/serverData/`
  - This mixes configuration, training outputs, runtime state, and generated datasets inside the source tree.
  - The repository currently treats generated artifacts as project structure instead of runtime data.
- `backend/react/`, `backend/static/`, and `reactDevelopment/`
  - There is duplication between development frontend code, compiled frontend output, and static serving assets.
  - This is acceptable short-term, but structurally unclear.
- committed build/runtime artifacts
  - `backend/fetchNewRuns`, `backend/fetchNewCalibrations`, `backend/build/`, `reactDevelopment/build/`, and `reactDevelopment/node_modules/` should not define repository structure.
- top-level naming and packaging
  - `ARHCITECTURE.md` is misspelled.
  - `requirements.txt` is a mixed human note rather than an installable Python environment definition.

### Where the current repository already matches the intended mixed C++/Python design

- The project already separates C++ scientific logic from Python training logic to a meaningful degree.
- ROOT-heavy processing is already implemented in C++, which fits the target architecture well.
- Python is already used for model training and some inference.
- Standalone data-fetch and analysis executables already exist, which is compatible with a future orchestration layer calling them from Python.

### Where it does not match the intended mixed C++/Python design

- Web/API responsibilities are currently in C++ via Drogon instead of Python.
- Orchestration is scattered across `main.cc`, controller methods, helper binaries, and commented code paths rather than a dedicated Python layer.
- Configuration and runtime state are not cleanly separated from source code.
- The C++ core is not isolated as a reusable library boundary; it is entangled with HTTP control flow, Python embedding, and environment-specific file paths.
- Database access boundaries are unclear, especially around Oracle.

## Section 2 - Proposed target repository structure

The next iteration should preserve the current repository shape where possible, but separate responsibilities much more explicitly. A practical target is to keep `backend/` as the umbrella for backend code while splitting it internally into clear modules.

```text
repo/
├── backend/
│   ├── cpp_core/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   ├── src/
│   │   ├── tools/
│   │   └── bindings/                # optional later stage
│   ├── python_api/
│   │   ├── pyproject.toml
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── api/
│   │   │   ├── schemas/
│   │   │   ├── services/
│   │   │   └── deps/
│   │   └── tests/
│   ├── orchestration/
│   │   ├── pyproject.toml           # or shared with python_api
│   │   ├── jobs/
│   │   ├── workers/
│   │   ├── cli/
│   │   └── services/
│   ├── training/
│   │   ├── pyproject.toml           # or shared package workspace
│   │   ├── src/
│   │   │   └── calibration_training/
│   │   ├── scripts/
│   │   └── tests/
│   ├── configs/
│   │   ├── app/
│   │   ├── db/
│   │   ├── training/
│   │   └── examples/
│   └── runtime/
│       ├── data/
│       ├── cache/
│       ├── outputs/
│       └── logs/
├── frontend/
│   └── react_app/
├── scripts/
├── tests/
│   ├── cpp/
│   ├── python/
│   └── integration/
├── docs/
├── README.md
├── AGENTS.md
└── .gitignore
```

### `backend/cpp_core/`

- Purpose
  - Home for the scientific and computational core: ROOT-heavy analysis, feature extraction, calibration table generation, trigger/EPICS data handling, and optional C++ inference helpers.
- Language
  - C++
- Key technologies
  - CMake
  - ROOT
  - PostgreSQL client layer (`pqxx`) if retained in C++
  - Oracle access layer if retained in C++
  - optional ONNX Runtime only if C++ inference remains needed

Suggested internal split:

- `include/` and `src/` for reusable library code.
- `tools/` for standalone executables such as run fetchers, histogram analyzers, dataset builders, and migration helpers.
- `bindings/` only if Python needs direct bindings instead of calling executables or subprocesses.

### `backend/python_api/`

- Purpose
  - Python web/API layer replacing Drogon for operator-facing endpoints and lightweight control actions.
- Language
  - Python
- Key technologies
  - FastAPI
  - Pydantic
  - Uvicorn
  - optionally SQLAlchemy for non-ROOT metadata/config access
  - subprocess or Python service wrappers to invoke C++ tools

Expected responsibilities:

- request models and validation
- endpoint definitions
- service layer that calls orchestration jobs or C++ tools
- no ROOT-heavy detector logic inside the HTTP handlers

### `backend/orchestration/`

- Purpose
  - Python process orchestration, job control, pipeline sequencing, and long-running worker logic.
- Language
  - Python
- Key technologies
  - Typer or argparse for CLI entrypoints
  - subprocess / asyncio
  - FastAPI-compatible service helpers if shared with the API layer
  - optionally Celery, RQ, or APScheduler later if background scheduling becomes necessary

Expected responsibilities:

- launch dataset acquisition jobs
- run C++ tools in controlled steps
- launch training runs
- manage prediction workers separate from API request handlers
- write status files or lightweight state records

For the current project state, simple Python CLIs and worker processes are more realistic than introducing a full distributed queue.

### `backend/training/`

- Purpose
  - Python ML package for dataset preparation specific to model training, neural network code, offline evaluation, and prediction utilities.
- Language
  - Python
- Key technologies
  - PyTorch
  - torch-geometric
  - NumPy
  - pandas
  - pyarrow
  - uproot
  - matplotlib
  - possibly PyROOT for analysis helpers
  - pytest

This should absorb the current `backend/function_prediction/` code with cleaner package boundaries:

- models
- dataset transforms
- train/evaluate scripts
- prediction adapters

### `backend/configs/`

- Purpose
  - Versioned configuration templates and environment-specific examples.
- Language
  - text/JSON/YAML/TOML
- Key technologies
  - JSON or YAML
  - Pydantic settings for Python apps

Expected contents:

- API config
- DB connection examples
- training config
- detector channel configuration
- runtime mode examples

### `backend/runtime/`

- Purpose
  - Non-source runtime data, generated datasets, model outputs, and logs.
- Language
  - mixed data only
- Key technologies
  - filesystem layout only

This is the correct home for what is currently under `backend/serverData/`, but it should mostly be gitignored except for tiny examples/templates.

### `frontend/react_app/`

- Purpose
  - Optional operator UI if the React client is still wanted.
- Language
  - JavaScript
- Key technologies
  - React
  - plotly-related packages already used in the current frontend

This can remain a separate frontend app, but built artifacts should not live in multiple places inside `backend/`.

### `scripts/`

- Purpose
  - Small repository-level helper scripts for local setup, migration, dev runs, and one-off maintenance.
- Language
  - shell and Python
- Key technologies
  - bash
  - Python CLI wrappers

### `tests/`

- Purpose
  - Cross-cutting tests separated by technology and scope.
- Language
  - C++ and Python
- Key technologies
  - pytest
  - gtest or Catch2 for C++
  - integration tests for API plus tool invocation

### Packaging and dependency management recommendation

For the Python parts, move away from the current note-style `requirements.txt` and use a real package definition:

- preferred: `pyproject.toml`
- acceptable initial variant: one shared `pyproject.toml` under `backend/` covering `python_api`, `orchestration`, and `training`

This is more realistic than maintaining several unrelated `requirements.txt` files.

## Section 3 - Mapping from current structure to target structure

### Top level

- `README.md`
  - Keep, but rewrite to describe the new architecture and run modes.
- `ARHCITECTURE.md`
  - Keep content, but rename to `ARCHITECTURE.md` in a later cleanup step.
- `AGENTS.md`
  - Keep as repository instructions.
- `requirements.txt`
  - Rewrite or replace with `backend/pyproject.toml`.

### Current `backend/`

- `backend/include/` and `backend/source/`
  - Move into `backend/cpp_core/include/` and `backend/cpp_core/src/`.
  - Split by responsibility if practical:
    - data access
    - calibration/domain logic
    - model input preparation
    - plotting/QA helpers
- `backend/analyseHists.cc`
  - Move to `backend/cpp_core/tools/`.
  - Keep as C++.
- `backend/fetchNewRuns.cc`
  - Move to `backend/cpp_core/tools/`.
  - Keep initially, then wrap from Python orchestration.
- `backend/fetchNewCalibrations.cc`
  - Move to `backend/cpp_core/tools/`.
  - Keep initially, then wrap from Python orchestration.
- `backend/main.cc`
  - Deprecate as the main application entrypoint.
  - Its remaining reusable logic should be split into C++ library code or tool entrypoints.
- `backend/controllers/`
  - Deprecate and remove after Python API replacement.
  - Do not port controller code directly one-to-one without first separating service logic.
- `backend/config.json`
  - Deprecate with Drogon removal.
  - Replace by Python API config under `backend/configs/app/`.
- `backend/function_prediction/`
  - Move into `backend/training/src/calibration_training/` plus `backend/training/scripts/`.
  - Keep the model and dataset code, but separate training, prediction, and utility modules more clearly.
- `backend/analyseData.py`
  - Move to `backend/training/` or `backend/python_api/services/` depending on usage.
  - Current contents suggest it belongs closer to analysis/training than to web/API.
- `backend/serverData/`
  - Split.
  - Configuration-like files move to `backend/configs/`.
  - generated artifacts move to `backend/runtime/`.
  - example seed files can remain tracked in a small `examples/` subdirectory.
- `backend/static/`
  - Keep only if the backend still serves static content directly.
  - Otherwise move responsibility to `frontend/` deployment or a simple Python static mount.
- `backend/react/`
  - Remove from source control if it is only a built artifact.
  - If it is required for runtime demos, regenerate it from the frontend app instead of treating it as source.
- `backend/build/`
  - Remove from source control; keep as build output only.

### Current `reactDevelopment/`

- `reactDevelopment/`
  - Move to `frontend/react_app/`.
  - Keep source, remove committed `node_modules/` and build output from repository tracking.
- `reactDevelopment/build/`
  - Remove from source control.
- `reactDevelopment/node_modules/`
  - Remove from source control.
- `reactDevelopment/src/App.test.js`
  - Keep only if the frontend is maintained; otherwise it is not meaningful for backend quality.

### Specific code-level mapping notes

- `backend/source/workWithDB.cc` and `backend/include/workWithDB.h`
  - Keep in C++ core.
  - Refactor into a data-access module plus dataset-export helpers.
- `backend/source/workWithTrigger.cc` and `backend/include/workWithTrigger.h`
  - Keep in C++ core.
- `backend/source/workWithOracleDB.cc` and `backend/include/workWithOracleDB.h`
  - Keep only if Oracle access must remain in C++.
  - Otherwise consider rewriting this access layer in Python if operationally easier.
  - In either case, isolate it behind a narrow interface.
- `backend/source/neuralNetwork.cc` and `backend/include/neuralNetwork.h`
  - Split.
  - Keep C++ feature/input shaping that is genuinely performance- or ROOT-dependent.
  - Move training triggers and orchestration responsibilities out to Python.
  - Revisit whether embedded Python inference is still needed.
- `backend/source/controllerBase.cc` and `backend/include/controllerBase.h`
  - Split.
  - Preserve reusable domain logic.
  - Remove web-server lifecycle assumptions and stateful service-locator coupling.
- `backend/source/serverData.cc` and `backend/include/serverData.h`
  - Rewrite as configuration/state handling with a smaller scope.
  - Current class mixes config, runtime cursor state, and persisted predictions.

## Section 4 - Additional change list

Overview-style practical checklist:

- Define `backend/cpp_core/` as the only place for C++ library code and C++ tools.
- Remove Drogon from the target runtime path and replace it with a small Python API layer, preferably FastAPI.
- Separate request handling from job execution; API handlers should trigger orchestration services, not perform long-running work inline.
- Isolate Oracle DB access behind a dedicated module with a clear interface and minimal compile-time spread.
- Decide whether Oracle access remains in C++ or moves to Python; do not keep the current half-integrated state.
- Keep PostgreSQL/EPICS data extraction in C++ only if the current implementation is already the most stable path.
- Split `NeuralNetwork` responsibilities into:
  - feature/input preparation
  - inference adapter
  - training orchestration hooks
  - plotting/QA utilities
- Move training code from `backend/function_prediction/` into a proper Python package under `backend/training/`.
- Introduce a real Python environment definition with `pyproject.toml`.
- Move config templates out of `backend/serverData/` into `backend/configs/`.
- Move generated data, predicted outputs, and logs out of source directories into `backend/runtime/`.
- Add `.gitignore` coverage for build outputs, runtime data, frontend build output, and `node_modules/`.
- Remove committed binaries and build directories from versioned source.
- Define a minimal Python CLI for common jobs such as:
  - fetch runs
  - fetch calibrations
  - build feature tables
  - build training dataset
  - train model
  - run prediction worker
  - start API
- Introduce tests incrementally:
  - unit tests for Python training/data helpers with `pytest`
  - smoke tests for FastAPI endpoints
  - C++ tests for utility/data transformation code with `gtest` or Catch2
- Document run modes explicitly:
  - offline dataset building
  - model training
  - API serving
  - long-running prediction worker
  - one-shot calibration acquisition tools

## Section 5 - Risks and migration notes

### Likely technical risks

- Oracle integration may remain the most fragile part regardless of language choice, due to local client installation and environment constraints.
- Some current C++ classes mix domain logic, plotting, and orchestration so heavily that extracting a clean reusable library boundary will take several passes.
- The current repository relies on hardcoded paths and local file conventions. Untangling those into configs without breaking workflows will be tedious.
- Replacing Drogon with FastAPI is conceptually straightforward, but preserving current operator workflows and endpoint semantics may still require careful mapping.
- The current C++ to Python inference bridge is inconsistent: there is embedded Python, partial ONNX support, and separate Python prediction scripts. That should be reduced to one supported path, but only after identifying which path actually works in practice.

### Where incremental migration is better than a full rewrite

- Keep ROOT-heavy analysis and table-building code in C++ and move it first without redesigning algorithms.
- Keep helper executables initially and wrap them from Python orchestration instead of rewriting them immediately.
- Preserve `backend/function_prediction/` logic by moving and packaging it, not rewriting model code from scratch.
- Replace the web/API layer before attempting deep optimization of all core classes.
- Separate runtime data from source control early; that cleanup gives immediate maintainability gains without changing detector logic.

### Places where backward compatibility is worth preserving

- Existing generated file formats in `backend/serverData/info_tables/` and `backend/serverData/nn_input/`, at least during migration.
- Existing helper-tool semantics for fetching runs and calibrations, even if the entrypoints move.
- Existing prediction/training inputs and outputs used by offline analysis scripts.
- Existing frontend expectations around endpoint payloads, if the React client is still intended to be used during the transition.

### Recommended migration sequence

1. Clean the repository layout without changing core algorithms.
2. Move generated artifacts and build outputs out of tracked source structure.
3. Extract C++ core and tools into `backend/cpp_core/`.
4. Package Python training code cleanly.
5. Add Python orchestration CLIs around current working tools.
6. Introduce a minimal FastAPI layer that exposes the current essential operations.
7. Retire Drogon only after the Python API covers the needed paths.
8. Refactor Oracle and runtime configuration handling once the new boundaries are in place.

This sequence is lower risk than rewriting the repository around a new architecture all at once.
