import torch
import torch.nn as nn
import torch.optim as optim
#import torch_geometric.data as PyGData
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import uproot
import pandas
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from pathlib import Path


class My_dataset(Dataset):
    def __init__(self, dataTable):
        self.datasetX, self.datasetY = dataTable[0], dataTable[1]

    def __len__(self):
        return len(self.datasetY)

    def __getitem__(self, index):
        return torch.tensor(self.datasetX[index]), torch.tensor(self.datasetY[index])
    

class Graph_dataset(Dataset):
    def __init__(self, dataTable):
        self.datasetX, self.datasetY, self.edge_index, self.edge_attr = dataTable[0], dataTable[1], dataTable[2], dataTable[3]

    def __len__(self):
        return len(self.datasetY)

    def __getitem__(self, index):
        #data = PyGData.Data(x=x, y=y, edge_index=hitEdges[0], edge_attr=hitEdges[1].float())
        return torch.tensor(self.datasetX[index]), torch.tensor(self.datasetY[index]), torch.LongTensor(self.edge_index), torch.Tensor(self.edge_attr)


def make_graph(config):
    cellsLength = config.cellsLength
    e_ind2 = []
    for i in range(cellsLength):
        rightLink, topLink = 0,0

        if (i/6 == (i+1)/6):
            rightLink = i+1
        elif (i/6 + 1 == (i+1)/6):
            rightLink = i-5

        topLink = i + 6

        if (i == cellsLength):
            rightLink = i-5

        e_ind2.append([i,rightLink])
        if (topLink < cellsLength and int(topLink)/6 != 2):
            e_ind2.append([i,topLink])
    e_ind2 = np.array(e_ind2)
    e_att2 = np.ones((len(e_ind2),))
    
    return e_ind2, e_att2


def load_dataset(config, dataTable):
    # transform to numpy, assign types, split on features-labels
    sentenceLength = config.sentenceLength
    cellsLength = config.cellsLength
    channelsLength = config.channelsLength
    df = dataTable
    dfn = df.to_numpy()

    if (config.modelType == "LSTM"):
        x = []
        y = []
        for i in range(dfn.shape[0]):
            xi = []
            yi = []
            if i<(sentenceLength-1):
                for j in range(sentenceLength-i-1):
                    xii = []
                    for s in range(channelsLength):
                        xii.append(dfn[0,24*(s)+0])
                    xi.append(xii)
                    #xi.append(dfn[0,:-1])
                    yi.append(dfn[0,-24])
                for j in range(i+1):
                    xii = []
                    for s in range(channelsLength):
                        xii.append(dfn[j,24*(s)+0])
                    xi.append(xii)
                    #xi.append(dfn[j,:-1])
                    yi.append(dfn[j,-24])
                    
            else:
                for j in range(sentenceLength):
                    xii = []
                    for s in range(channelsLength):
                        xii.append(dfn[i-sentenceLength+1+j,24*(s)+0])
                    xi.append(xii)
                    #xi.append(dfn[i-sentenceLength+1+j,:-1])
                    yi.append(dfn[i-sentenceLength+1+j,-24])
            x.append(xi)
            y.append(yi)
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)

    elif (config.modelType == "ConvLSTM"):    
        x = []
        y = []
        for i in range(dfn.shape[0]):
            xi = []
            yi = []
            if i<(sentenceLength-1):
                for j in range(sentenceLength-i-1):
                    xii = []
                    yii = dfn[0,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[0,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append(yii)
                for j in range(i+1):
                    xii = []
                    yii = dfn[j,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[j,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append(yii)
            else:
                for j in range(sentenceLength):
                    xii = []
                    yii = dfn[i-sentenceLength+1+j,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[i-sentenceLength+1+j,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append(yii)
            x.append(xi)
            y.append(yi)
        x = (np.array(x).astype(np.float32))[...,np.newaxis]
        y = np.array(y).astype(np.float32)

    elif (config.modelType == "gConvLSTM"):
        x = []
        y = []
        for i in range(dfn.shape[0]):
            xi = []
            yi = []
            if i<(sentenceLength-1):
                for j in range(sentenceLength-i-1):
                    xii = []
                    yii = dfn[0,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[0,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append(yii)
                for j in range(i+1):
                    xii = []
                    yii = dfn[j,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[j,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append(yii)
            else:
                for j in range(sentenceLength):
                    xii = []
                    yii = dfn[i-sentenceLength+1+j,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[i-sentenceLength+1+j,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append(yii)
            x.append(xi)
            y.append(yi)
        x = (np.array(x).astype(np.float32))
        y = np.array(y).astype(np.float32)

        e_ind2, e_att2 = make_graph(config)

        # extend to the events length size, to forward it to data loader
        np.tile(e_ind2, (len(y), 1, 1))
        np.tile(e_att2, (len(y), 1, 1))
            

    elif (config.modelType == "DNN"):
        x = dfn[:,:-1].astype(np.float32)
        y = dfn[:, -1].astype(np.float32)


    #print(y)
    print('x shape = ' + str(x.shape))
    print('y shape = ' + str(y.shape))
    if (config.modelType == "gConvLSTM"):
        print('e_ind shape = ' + str(e_ind2.shape))
        print('e_att shape = ' + str(e_att2.shape))
        return (x, y, e_ind2, e_att2)
    return (x, y)


class DataManager():
    def __init__(self, mainPath) -> None:
        self.poorColumnValues = []
        self.mainPath = mainPath


    def meanAndStdTable(self, dataTable):
        # find mean and std values for each column of the dataset (used for train dataset)
        df = dataTable
        dfn = df.to_numpy()

        x = dfn[:,:].astype(np.float32)
        mean = np.array( [np.mean(x[:,j+1]) for j in range(x.shape[1]-1)] )
        std  = np.array( [np.std( x[:,j+1]) for j in range(x.shape[1]-1)] )
    
        for i in range(x.shape[1]-1):
            if (std[i]!=0): 
                selection = ((x[:,i+1]-mean[i])/std[i]>-5) & ((x[:,i+1]-mean[i])/std[i]<5)
            #print(selection)
            x1 = x[selection,i+1]
            mean[i] = np.mean(x1)
            std[i] = np.std(x1)
        
        #__ if you have bad data sometimes in one of the columns - 
        # - you can calculate mean and std without these bad entries
        #   and then make them = 0 -> no effect on the first layer
        for i in range(len(self.poorColumnValues)):
            cPoor = df.columns.get_loc(self.poorColumnValues[i][0])
            vPoor = self.poorColumnValues[i][1]
            mean[cPoor] = np.mean(x[(x[:,cPoor]!=vPoor),cPoor])
            std[cPoor] = np.std(x[(x[:,cPoor]!=vPoor),cPoor])

        return mean, std


    def normalizeDataset(self, df):
        #print(df)
        pathFP = self.mainPath + 'function_prediction/'
        meanValues, stdValues = readTrainData(pathFP)
        columns = list(df.columns)
        masks = []
        for i in range(len(self.poorColumnValues)):
            masks.append(df[self.poorColumnValues[i][0]]==self.poorColumnValues[i][1])

        for i in range(len(columns)-1):
            if (stdValues[i]!=0):
                df[columns[i+1]] = (df[columns[i+1]]-meanValues[i])/stdValues[i]
            else:
                df[columns[i+1]] = (df[columns[i+1]]-meanValues[i])/1
        
        for i in range(len(self.poorColumnValues)):
            df[self.poorColumnValues[i][0]].mask(masks[i], 0, inplace=True)

        #print(df)

        return df
    
    def cutDataset(self, df0):
        #print("CUTTING")
        columns = list(df0.columns)
        df = df0.copy()
        df = self.normalizeDataset(df)
        #print(df)

        columns = list(df.columns)
        selection = (df[columns[1]]>-4) & (df[columns[1]]<4)
        for i in range(len(columns)-3):
            selection = selection & (df[columns[i+2]]>-4) & (df[columns[i+2]]<4)
        df0 = df0.loc[selection].copy()
        #print(df0)
        return df0


    def getDataset(self, rootPath,mod):
        # read data, select raws (pids) and columns (drop)

        setTable = pandas.read_table(rootPath,sep=' ',header=None)
    
        return setTable

    def manageDataset(self, mod):
        pathFP = self.mainPath + 'function_prediction/'

        #self.prepareTable
        dir = str(Path(__file__).parents[1])
        #print(dir)
        # outNNTestSM / outNNTest1
        dftCorr = self.getDataset(self.mainPath + "nn_input/outNNTestSM.dat", "simLabel")
        #print(dftCorr)

        dftTV = dftCorr.iloc[:int(dftCorr.shape[0]*0.7)].copy()
        dftTrainV = dftTV.iloc[:int(dftTV.shape[0]*0.7)].copy()
        dftTest = dftTV.drop(dftTrainV.index)
        dftCorr = dftTrainV.copy()

        #print(dftCorr)
        #print(dftTest)

        mean, std = 0, 0
        if (mod.startswith("train")):
            mean, std = self.meanAndStdTable(dftCorr)
            writeTrainData(mean,std, pathFP)
        elif (mod.startswith("test")):
            mean, std = readTrainData(pathFP)

        dftCorr = self.cutDataset(dftCorr).copy()
        dftTest = self.cutDataset(dftTest).copy()

        pq.write_table(pa.Table.from_pandas(dftCorr), pathFP + 'simu.parquet')
        pq.write_table(pa.Table.from_pandas(dftTest), pathFP + 'tesu.parquet')

        dftCorr1 = self.normalizeDataset(dftCorr).copy()

        mean1, std1 = self.meanAndStdTable(dftCorr1)
        print("mean values: " + str(mean1))
        print("std  values: " + str(std1))
        #print(dftCorr)

        #pq.write_table(pa.Table.from_pandas(dftCorr), pathFP + 'simu1.parquetmanageDataset')
        #pq.write_table(pa.Table.from_pandas(dftTest), pathFP + 'tesu1.parquet')
        #print(dftCorr)




def writeTrainData(meanArr,stdArr, path):
    np.savetxt(path + 'meanValues.txt', meanArr, fmt='%s')
    np.savetxt(path + 'stdValues.txt', stdArr, fmt='%s')

def readTrainData(path):
    meanValues = np.loadtxt(path + 'meanValues.txt')
    stdValues = np.loadtxt(path + 'stdValues.txt')
    return meanValues, stdValues

