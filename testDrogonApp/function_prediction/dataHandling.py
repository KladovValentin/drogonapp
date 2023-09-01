import torch
import torch.nn as nn
import torch.optim as optim
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


def load_dataset(config, dataTable):
    # transform to numpy, assign types, split on features-labels
    sentenceLength = 15
    df = dataTable
    dfn = df.to_numpy()
    print(dfn)


    if (config.modelType == "LSTM"):
        x = []
        y = []
        for i in range(dfn.shape[0]):
            xi = []
            yi = []
            if i<(sentenceLength-1):
                for j in range(sentenceLength-i-1):
                    xi.append(dfn[0,:-1])
                    yi.append(dfn[0,-1])
                for j in range(i+1):
                    yi.append(dfn[j,-1])
                    xi.append(dfn[j,:-1])
            else:
                for j in range(sentenceLength):
                    xi.append(dfn[i-sentenceLength+1+j,:-1])
                    yi.append(dfn[i-sentenceLength+1+j,-1])
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
                    yii = []
                    for s in range(6):
                        xii.append(dfn[0,7*(s):7*(s+1)])
                        yii.append(dfn[0,-6+s])
                    xiii = []
                    xiii.append(xii)
                    xi.append(xiii)
                    yi.append(yii)
                for j in range(i+1):
                    xii = []
                    yii = []
                    for s in range(6):
                        xii.append(dfn[j,7*(s):7*(s+1)])
                        yii.append(dfn[j,-6+s])
                    xiii = []
                    xiii.append(xii)
                    xi.append(xiii)
                    yi.append(yii)
            else:
                for j in range(sentenceLength):
                    xii = []
                    yii = []
                    for s in range(6):
                        xii.append(dfn[i-sentenceLength+1+j,7*(s):7*(s+1)])
                        yii.append(dfn[i-sentenceLength+1+j,-6+s])
                    xiii = []
                    xiii.append(xii)
                    xi.append(xiii)
                    yi.append(yii)
            x.append(xi)
            y.append(yi)
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)

    elif (config.modelType == "DNN"):
        x = dfn[:,:-1].astype(np.float32)
        y = dfn[:, -1].astype(np.float32)


    #print(x)
    print('x shape = ' + str(x.shape))
    print('y shape = ' + str(y.shape))
    return (x, y)



class DataManager():
    def __init__(self) -> None:
        self.poorColumnValues = []



    def meanAndStdTable(self, dataTable):
        # find mean and std values for each column of the dataset (used for train dataset)
        df = dataTable
        dfn = df.to_numpy()

        x = dfn[:,:].astype(np.float32)
        mean = np.array( [np.mean(x[:,j+1]) for j in range(x.shape[1]-1)] )
        std  = np.array( [np.std( x[:,j+1]) for j in range(x.shape[1]-1)] )
        print(mean)
        print(std)

    
        for i in range(x.shape[1]-1):
            selection = ((x[:,i+1]-mean[i])/std[i]>-5) & ((x[:,i+1]-mean[i])/std[i]<5)
            print(selection)
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


    def normalizeDataset(self, df, meanValues, stdValues):
        print(df)
        columns = list(df.columns)
        masks = []
        for i in range(len(self.poorColumnValues)):
            masks.append(df[self.poorColumnValues[i][0]]==self.poorColumnValues[i][1])

        for i in range(len(columns)-1):
            df[columns[i+1]] = (df[columns[i+1]]-meanValues[i])/stdValues[i]
        
        for i in range(len(self.poorColumnValues)):
            df[self.poorColumnValues[i][0]].mask(masks[i], 0, inplace=True)

        return df
    
    def cutDataset(self, df0, meanValues, stdValues):
        #print("CUTTING")
        columns = list(df0.columns)
        df = df0.copy()
        df = self.normalizeDataset(df, meanValues, stdValues)
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

    def manageDataset(self, mod, path):
        pathFP = path + 'function_prediction/'

        #self.prepareTable
        dir = str(Path(__file__).parents[1])
        print(dir)
        dftCorr = self.getDataset(path + "nn_input/outNNTest.dat", "simLabel")
        print(dftCorr)

        dftTV = dftCorr.iloc[:int(dftCorr.shape[0]*0.7)].copy()
        dftTrainV = dftTV.iloc[:int(dftTV.shape[0]*0.7)].copy()
        dftTest = dftTV.drop(dftTrainV.index)
        dftCorr = dftTrainV.copy()

        print(dftCorr)
        print(dftTest)
        #print(dftCorr)

        mean, std = 0, 0
        if (mod == "train_nn"):
            mean, std = self.meanAndStdTable(dftCorr)
        elif (mod.startswith("test")):
            mean, std = readTrainData(pathFP)

        dftCorr = self.cutDataset(dftCorr,mean,std).copy()
        dftTest = self.cutDataset(dftTest,mean,std).copy()

        pq.write_table(pa.Table.from_pandas(dftCorr), pathFP + 'simu.parquet')
        pq.write_table(pa.Table.from_pandas(dftTest), pathFP + 'tesu.parquet')


        dftCorr = self.normalizeDataset(dftCorr,mean,std).copy()
        dftTest = self.normalizeDataset(dftTest,mean,std).copy()

        mean1, std1 = self.meanAndStdTable(dftCorr)
        print("mean values: " + str(mean1))
        print("std  values: " + str(std1))
        print(dftCorr)

        pq.write_table(pa.Table.from_pandas(dftCorr), pathFP + 'simu1.parquet')
        pq.write_table(pa.Table.from_pandas(dftTest), pathFP + 'tesu1.parquet')
        print(dftCorr)

        if (mod.startswith("train")):
            writeTrainData(mean,std, pathFP)



def writeTrainData(meanArr,stdArr, path):
    np.savetxt(path + 'meanValues.txt', meanArr, fmt='%s')
    np.savetxt(path + 'stdValues.txt', stdArr, fmt='%s')

def readTrainData(path):
    meanValues = np.loadtxt(path + 'meanValues.txt')
    stdValues = np.loadtxt(path + 'stdValues.txt')
    return meanValues, stdValues

