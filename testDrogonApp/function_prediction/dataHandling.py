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
from config import Config


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
        rightLink, topLink, leftLink = 0,0,0

        rightLink = i+1
        #if (i == 5 or i == 11 or i == 17 or i == 23):
        #    rightLink = i-5

        topLink = i + 6

        if (i != 5 and i != 11 and i != 17 and i != 23):
            e_ind2.append([i,rightLink])
        if (i == 0 or i == 6 or i == 12 or i == 18):
            leftLink = i + 5
            e_ind2.append([i,leftLink])
        
        
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
                        xii.append(dfn[0,cellsLength*(s)+0])
                    xi.append(xii)
                    #xi.append(dfn[0,:-1])
                    yi.append(dfn[0,-cellsLength])
                for j in range(i+1):
                    xii = []
                    for s in range(channelsLength):
                        xii.append(dfn[j,cellsLength*(s)+0])
                    xi.append(xii)
                    #xi.append(dfn[j,:-1])
                    yi.append(dfn[j,-cellsLength])
                    
            else:
                for j in range(sentenceLength):
                    xii = []
                    for s in range(channelsLength):
                        xii.append(dfn[i-sentenceLength+1+j,cellsLength*(s)+0])
                    xi.append(xii)
                    #xi.append(dfn[i-sentenceLength+1+j,:-1])
                    yi.append(dfn[i-sentenceLength+1+j,-cellsLength])
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
                    yii = dfn[0,-2*cellsLength:-cellsLength]
                    yiie = dfn[0,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[0,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append([yii,yiie])
                for j in range(i+1):
                    xii = []
                    yii = dfn[j,-2*cellsLength:-cellsLength]
                    yiie = dfn[j,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[j,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append([yii,yiie])
            else:
                for j in range(sentenceLength):
                    xii = []
                    yii = dfn[i-sentenceLength+1+j,-2*cellsLength:-cellsLength]
                    yiie = dfn[i-sentenceLength+1+j,-cellsLength:]
                    for s in range(channelsLength):
                        xii.append(dfn[i-sentenceLength+1+j,cellsLength*(s):cellsLength*(s+1)])
                    xi.append(xii)
                    yi.append([yii,yiie])
            x.append(xi)
            y.append(yi)
        x = (np.array(x).astype(np.float32))
        y = np.array(y).astype(np.float32)

        e_ind2, e_att2 = make_graph(config)

        # extend to the events length size, to forward it to data loader
        #np.tile(e_ind2, (len(y), 1, 1))
        #np.tile(e_att2, (len(y), 1, 1))
            

    elif (config.modelType == "DNN"):
        x = dfn[:,:-1].astype(np.float32)
        y = dfn[:, -1].astype(np.float32)

    # for each entry (x[i]) -> x[i][-1] = (7,24) -> if (x[i][-j-2] - x[i][-1] > 4)
    
    
    listt = list()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]-1):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    listt.append(np.abs(x[i][j][k][l] - x[i][-j-2][k][l]))
                    if (np.abs(x[i][j][k][l] - x[i][-j-2][k][l]) < 0.5):
                        continue
                    x[i][-j-2][k][l] = x[i][-j-2+1][k][l]

    
    """
    data = np.array(listt)
    hist, bins = np.histogram(data, bins=100)
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')  # Plot the histogram
    plt.title('Histogram of Random Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    """

    #print(y)
    print('x shape = ' + str(x.shape))
    print('y shape = ' + str(y.shape))
    if (config.modelType == "gConvLSTM"):
        print('e_ind shape = ' + str(e_ind2.shape))
        print('e_att shape = ' + str(e_att2.shape))
        print(torch.LongTensor(e_ind2).movedim(-2,-1))
        print(e_att2)
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
        cellsLength = Config().cellsLength

        pathFP = self.mainPath + 'function_prediction/'
        meanValues, stdValues = readTrainData(pathFP,"")
        columns = list(df.columns)
        masks = []
        for i in range(len(self.poorColumnValues)):
            masks.append(df[self.poorColumnValues[i][0]]==self.poorColumnValues[i][1])

        for i in range(len(columns)-1):
            if (i >= len(columns)-1-cellsLength): #errors
                df[columns[i+1]] = df[columns[i+1]].replace(0, meanValues[i]*10)
                df[columns[i+1]] = (df[columns[i+1]])/stdValues[i-cellsLength]
                continue
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
        cut = 7

        columns = list(df.columns)
        criteria = ((df.iloc[:,1:]<-5) | (df.iloc[:,1:]>5))
        selection = (df[columns[1]]>-cut) & (df[columns[1]]<cut)
        for i in range(len(columns)-4):
            selection = selection & (df[columns[i+2]]>-cut) & (df[columns[i+2]]<cut)
        #df0.iloc[:,1:][criteria] = 0
        df0 = df0.loc[selection].copy()
        #print(df0)
        return df0


    def getDataset(self, rootPath,mod):
        # read data, select raws (pids) and columns (drop)

        setTable = pandas.read_table(rootPath,sep=' ',header=None)
    
        return setTable

    def manageDataset(self, mod,ind):
        pathFP = self.mainPath + 'function_prediction/'

        #self.prepareTable
        dir = str(Path(__file__).parents[1])
        #print(dir)
        # outNNTestSM / outNNTest1
        dftCorr = self.getDataset(self.mainPath + "nn_input/outNNTestSMzxc.dat", "simLabel")
        #print(dftCorr)

        baseTrainRange = int(dftCorr.shape[0]*0.2)
        retrainIndex = ind
        retrain0 = max(baseTrainRange + 60*(retrainIndex-3), baseTrainRange)
        retrain1 = baseTrainRange + 60*retrainIndex
        retrain2 = baseTrainRange + 60*(retrainIndex+1)
        if (retrainIndex == 0):
            dftTV = dftCorr.iloc[:baseTrainRange].copy()
        else:
            dftTV = dftCorr.iloc[retrain0:retrain1].copy()
        dftTrainV = dftTV.copy()
        #dftTrainV = dftTV.iloc[:int(dftTV.shape[0]*1.0)].copy()
        #startRetrain = dftCorr.shape[0]-1000
        #if startRetrain < 0:
        #    startRetrain = 0
        #dftTV = dftCorr.iloc[startRetrain:int(dftCorr.shape[0])].copy()
        #dftTrainV = dftTV.iloc[:int(dftTV.shape[0])].copy()
        #dftTest = dftTV.drop(dftTrainV.index)
        dftTest = dftCorr.iloc[retrain1:retrain2].copy()
        dftCorr = dftTrainV.copy()

        #print(dftCorr)
        #print(dftTest)

        mean, std = 0, 0
        if (mod.startswith("train")):
            mean, std = self.meanAndStdTable(dftCorr)
            writeTrainData(mean,std, pathFP, "")
        elif (mod.startswith("test")):
            mean, std = readTrainData(pathFP,"")

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




def writeTrainData(meanArr,stdArr, path, mod):
    np.savetxt(path + 'meanValues'+mod+'.txt', meanArr, fmt='%s')
    np.savetxt(path + 'stdValues'+mod+'.txt', stdArr, fmt='%s')

def readTrainData(path,mod):
    meanValues = np.loadtxt(path + 'meanValues'+mod+'.txt')
    stdValues = np.loadtxt(path + 'stdValues'+mod+'.txt')
    return meanValues, stdValues

