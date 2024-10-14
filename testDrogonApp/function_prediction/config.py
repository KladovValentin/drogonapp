from dataclasses import dataclass

@dataclass
class Config:
    modelType = "gConvLSTM" # DNN / LSTM
    sentenceLength = 3
    cellsLength = 24
    channelsLength = 7
