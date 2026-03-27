# Models Summary

## Current model switch

Model selection is now centralized through the shared registry in `config.py`.

The supported high-level values are:
- `gConvLSTM`
- `ChebGConvLSTM`
- `PureFCN`
- `GraphTransformer2D`
- `BoostedTrees`

All of these are now routed automatically through:
- `config.py`
- `dataHandling.py`
- `trainNetwork.py`
- `predict.py`

This means that for the supported models above, train and predict no longer need manual line-by-line commenting or uncommenting just to pick the model family.

## How to switch models now

Set only one line in `config.py`:

```python
class Config:
    modelType = "gConvLSTM"
```

Replace the string with one of:
- `gConvLSTM`
- `ChebGConvLSTM`
- `PureFCN`
- `GraphTransformer2D`
- `BoostedTrees`

That single switch now changes all needed logic automatically in training and prediction.

## Shared registry idea

The registry lives in `config.py` as `MODEL_REGISTRY` and defines, per model type:
- model family
- input layout
- artifact type
- artifact filename
- for GCNLSTM-based variants, the internal graph-cell type and sequence source

Current registry meaning:
- `gConvLSTM`: GCNLSTM with the current spectral graph-recurrent cell and hidden-state sequence path
- `ChebGConvLSTM`: GCNLSTM with the Chebyshev-based graph-recurrent cell
- `PureFCN`: GCNLSTM with the FCN-style input projection path instead of the hidden-state sequence output
- `GraphTransformer2D`: transformer-based graph model
- `BoostedTrees`: tabular boosted-tree baseline

## Available models

### 1. `gConvLSTM`

Main idea:
- graph-aware recurrent model over run sequences
- per-node hidden state through time
- uses detector-cell graph structure and the `dT` channel

Implementation:
- family: `gcn_lstm`
- class: `GCNLSTM`
- current internal settings from registry:
  - `gcn_cell_type = "spectral"`
  - `sequence_source = "hidden_state"`

Artifact:
- `tempModelT.pt`

### 2. `ChebGConvLSTM`

Main idea:
- same outer `GCNLSTM` model family
- swaps the internal graph-recurrent cell to the Chebyshev-convolution version

Implementation:
- family: `gcn_lstm`
- class: `GCNLSTM`
- registry settings:
  - `gcn_cell_type = "cheb"`
  - `sequence_source = "hidden_state"`

Artifact:
- `tempModelT.pt`

### 3. `PureFCN`

Main idea:
- ablation of the graph-LSTM output path
- still uses the same outer GCNLSTM container and input handling
- switches the sequence output to the direct FCN-style `input_proj(...)` path

Implementation:
- family: `gcn_lstm`
- class: `GCNLSTM`
- registry settings:
  - `gcn_cell_type = "spectral"`
  - `sequence_source = "input_proj"`

Artifact:
- `tempModelT.pt`

### 4. `GraphTransformer2D`

Main idea:
- graph-aware spatial attention over cells
- temporal transformer over the sequence dimension
- replaces graph recurrence with masked attention

Implementation:
- family: `graph_transformer`
- class: `GraphTransformer2D`

Artifact:
- `tempModelT.pt`

### 5. `BoostedTrees`

Main idea:
- non-neural baseline
- converts the window into compact tabular summary features
- one boosted regressor per output node

Implementation:
- family: `boosted_trees`
- class: `BoostedTreesRegressor`

Artifact:
- `tempModelBoosted.pkl`

Performance output:
- `performance_BoostedTrees_.txt`

## What changes automatically now

### `dataHandling.py`

The dataset layout is chosen through the shared registry instead of direct `modelType` `if/else` checks.
Right now all supported models in the registry use the same graph-sequence input layout.

### `trainNetwork.py`

Training now reads the registry and automatically decides:
- whether the model is torch-based or pickle-based
- which class to instantiate
- which artifact filename to save
- for `GCNLSTM`, which internal cell type and sequence mode to use

### `predict.py`

Prediction now reads the same registry and automatically decides:
- how to load the artifact
- which class to construct
- whether the prediction path is torch-based or boosted-tree based

## What is no longer part of the supported switch list

`Conv2dLSTM` is still present in `models/model.py`, but it is no longer part of the shared config-driven model list and is not documented here as a supported comparison option.

## Minimum future improvements

The current registry is a good first step, but these small follow-ups would improve automated testing further.

### 1. Move model construction into one shared builder

Right now train and predict both read the registry, but they still each contain constructor logic.
A next small step would be one shared builder function that both call.

### 2. Move artifact save/load into one shared helper

The registry already carries the artifact name and type.
A next small helper should centralize:
- save torch state dict
- save pickle model
- load torch state dict
- load pickle model

### 3. Move the GCNLSTM variant names into their own explicit model classes later

At the moment:
- `ChebGConvLSTM`
- `PureFCN`

are registry-driven variants of `GCNLSTM`.
That is enough for comparison tests now.
Later they can become separate classes without changing the outer config switch pattern.

### 4. Add one benchmark loop

A simple benchmark script should:
- iterate over all registry model types
- train each model
- run prediction metrics
- write `performance_<model>.txt`
- build one comparison table

That would make the current registry directly useful for automatic performance scans.
