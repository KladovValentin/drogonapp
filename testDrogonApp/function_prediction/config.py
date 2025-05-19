from dataclasses import dataclass

@dataclass
class Config:
    modelType = "gConvLSTM" # DNN / LSTM
    sentenceLength = 1
    cellsLength = 24
    channelsLength = 9
