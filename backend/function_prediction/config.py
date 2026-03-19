from dataclasses import dataclass

@dataclass
class Config:
    #modelType = "gConvLSTM" # ConvLSTM / gConvLSTM / BoostedTrees
    modelType = "BoostedTrees"
    sentenceLength = 10
    cellsLength = 24
    channelsLength = 9
