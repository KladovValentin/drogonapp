from dataclasses import dataclass

MODEL_REGISTRY = {
    "gConvLSTM": {
        "model_family": "gcn_lstm",
        "input_layout": "graph_sequence",
        "artifact_type": "torch",
        "artifact_name": "tempModelT.pt",
        "gcn_cell_type": "spectral",
        "sequence_source": "hidden_state",
    },
    "ChebGConvLSTM": {
        "model_family": "gcn_lstm",
        "input_layout": "graph_sequence",
        "artifact_type": "torch",
        "artifact_name": "tempModelT.pt",
        "gcn_cell_type": "cheb",
        "sequence_source": "hidden_state",
    },
    "PureFCN": {
        "model_family": "gcn_lstm",
        "input_layout": "graph_sequence",
        "artifact_type": "torch",
        "artifact_name": "tempModelT.pt",
        "gcn_cell_type": "spectral",
        "sequence_source": "input_proj",
    },
    "GraphTransformer2D": {
        "model_family": "graph_transformer",
        "input_layout": "graph_sequence",
        "artifact_type": "torch",
        "artifact_name": "tempModelT.pt",
    },
    "BoostedTrees": {
        "model_family": "boosted_trees",
        "input_layout": "graph_sequence",
        "artifact_type": "pickle",
        "artifact_name": "tempModelBoosted.pkl",
    },
}

SUPPORTED_MODEL_TYPES = tuple(MODEL_REGISTRY.keys())


def get_model_spec(model_type):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported modelType {model_type!r}. Supported values: {SUPPORTED_MODEL_TYPES}")
    return MODEL_REGISTRY[model_type]


@dataclass
class Config:
    modelType = "PureFCN"  # gConvLSTM / ChebGConvLSTM / PureFCN / GraphTransformer2D / BoostedTrees
    sentenceLength = 1
    cellsLength = 24
    channelsLength = 9
