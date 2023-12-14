from dataclasses import dataclass

@dataclass
class Config:
    modelType = "gConvLSTM" # DNN / LSTM
    sentenceLength = 15
    cellsLength = 24
    channelsLength = 7
