from dataclasses import dataclass

@dataclass
class Config:
    modelType = "gConvLSTM" # DNN / LSTM
    sentenceLength = 5
    cellsLength = 24
    channelsLength = 7
