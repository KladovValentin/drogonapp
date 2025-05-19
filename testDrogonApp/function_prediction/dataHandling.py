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

import torch
import numpy as np
import pandas as pd
from config import Config
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib.pyplot as plt


class My_dataset(torch.utils.data.Dataset):
    def __init__(self, data_table):
        self.X, self.Y = data_table[0], data_table[1]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])
    

class Graph_dataset(torch.utils.data.Dataset):
    def __init__(self, data_table):
        self.X, self.Y, self.edge_index, self.edge_attr = data_table

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx]),
                torch.tensor(self.Y[idx]),
                torch.LongTensor(self.edge_index),
                torch.tensor(self.edge_attr))


# --- Graph Creation ---

def make_graph(config):
    cells_length = 12  # FIXME: this overrides config.cellsLength, clarify this intention
    edges = []

    for i in range(cells_length):
        right, top = i + 1, i + 6

        if i not in {5, 11, 17, 23}:
            edges.append([i, right])
        if i in {0, 6, 12, 18}:
            edges.append([i, i + 5])
        if top < cells_length:
            edges.append([i, top])

    return np.array(edges), np.ones(len(edges))


def load_dataset(config, df):
    # transform to numpy, assign types, split on features-labels
    cellsLengthToUse = 12
    sentenceLength, cellsLength, channelsLength = config.sentenceLength, config.cellsLength, config.channelsLength
    dfn = df.to_numpy()

    if (config.modelType == "ConvLSTM"):    
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

    if (config.modelType == "gConvLSTM"):
        dfnRun = dfn[:,0]
        dfnX = dfn[:,1:-2*cellsLength]
        dfnY = dfn[:,-2*cellsLength:]
        dfnX = dfnX.reshape((-1,channelsLength,cellsLength))
        dfnY = dfnY.reshape((-1,2,cellsLength))

        dfnXExtended = np.zeros((dfnX.shape[0],sentenceLength,channelsLength,cellsLength))
        dfnYExtended = np.zeros((dfnY.shape[0],sentenceLength,2,cellsLength))
        # Fill the new array
        for i in range(dfnX.shape[0]):
            sentenceCut = False
            sentenceCutIndex = 0
            for j in range(sentenceLength):
                j1 = sentenceLength - j - 1
                if j != 0 and ((i - j >= 0) and (dfnRun[i-j+1] - dfnRun[i-j] >= 450)):       # If the previous run is too old - cut it and substitute with the last valid
                    sentenceCut = True
                    sentenceCutIndex = j-1
                
                if j == 0 or ((i - j > 0) and not sentenceCut):    # Only take valid previous indices and check time difference
                    dt_row = np.full((1, cellsLength), dfnRun[i-j] - dfnRun[i-j-1])
                    #dfnXExtended[i, j1] = np.vstack([dfnX[i - j],dt_row])
                    dfnXExtended[i, j1] = dfnX[i - j]
                    dfnYExtended[i, j1] = dfnY[i - j]
                    #dfnXExtended[i, j1] = dfnX[i]
                elif (i - j <= 0):                                   # If we are at the beginning of the sequence
                    dt_row = torch.full((1, cellsLength), 0) 
                    #dfnXExtended[i, j1] = np.vstack([dfnX[0],dt_row])
                    dfnXExtended[i, j1] = dfnX[0]
                    dfnYExtended[i, j1] = dfnY[0]
                    #dfnXExtended[i, j1] = dfnX[i]

                elif (sentenceCut):                                 # If we have a situation with a gap between runs
                    dt_row = torch.full((1, cellsLength), 0)
                    #dfnXExtended[i, j1] = np.vstack([dfnX[i - sentenceCutIndex],dt_row])
                    dfnXExtended[i, j1] = dfnX[i - sentenceCutIndex]
                    dfnYExtended[i, j1] = dfnY[i - sentenceCutIndex]

        x = dfnXExtended[:,:,:,:cellsLengthToUse].astype(np.float32)#.reshape((dfnX.shape[0],sentenceLength,7,12))
        y = dfnYExtended[:,:,:,:cellsLengthToUse].astype(np.float32)


        arrayToPlot1 = x[:,-1,0,0]
        arrayToPlot2 = y[:,-1,0,0]
        xToPlot = np.arange(0,dfnX.shape[0])

        plt.plot(xToPlot, arrayToPlot1, color='#0504aa', label = 'input pressure', marker='o', linestyle="None", markersize=0.8)
        plt.plot(xToPlot, arrayToPlot2, color='#8b2522', label = 'target values', marker='o', linestyle="None", markersize=1.7)

        #plt.plot(arrayToPlot1, arrayToPlot2, color='#0504aa', label = 'target values', marker='o', linestyle="None", markersize=1.7)

        plt.show()

        e_ind2, e_att2 = make_graph(config)

        # extend to the events length size, to forward it to data loader
        #np.tile(e_ind2, (len(y), 1, 1))
        #np.tile(e_att2, (len(y), 1, 1))

    #shape: batch, sentence, in_channels, nodes
    print('x shape = ' + str(x.shape))
    print('y shape = ' + str(y.shape))
    if (config.modelType == "gConvLSTM"):
        print('e_ind shape = ' + str(e_ind2.shape))
        print('e_att shape = ' + str(e_att2.shape))
        print(torch.LongTensor(e_ind2).movedim(-2,-1))
        print(e_att2)
        return (x, y, e_ind2, e_att2)
    return (x, y)


def compute_scaling_factor(mean_value, target_range=(1.0, 10.0)):
    """
    Given a mean value, return a factor (power of 10) to divide the feature by
    so that the mean is in the target range [0.1, 10].
    """
    if mean_value == 0 or not np.isfinite(mean_value):
        return 1.0  # no scaling if mean is 0 or invalid

    scale = 10 ** round(np.log10(abs(mean_value)))
    scaled_mean = abs(mean_value) / scale

    if scaled_mean < target_range[0]:
        scale /= 10  # bring up
    elif scaled_mean > target_range[1]:
        scale *= 10  # bring down

    return scale


class DataManager():
    def __init__(self, mainPath) -> None:
        self.poorColumnValues = []
        self.mainPath = mainPath


    def meanAndStdTable(self, dataTable):
        # find mean and std values for each column of the dataset (used for train dataset)
        df = dataTable
        dfn = df.to_numpy()

        cellsLength = Config().cellsLength
        channelsLength = Config().channelsLength

        x = dfn[:,1:-2*cellsLength].reshape((-1,1,channelsLength,cellsLength))
        y = dfn[:,-2*cellsLength:].reshape((-1,2,cellsLength))


        meanX = np.ones((x.shape[2],x.shape[3]))
        stdX = np.ones((x.shape[2],x.shape[3]))
        for i in range(x.shape[3]):
            meanX[:,i] = np.array( [np.mean(x[:,:,j,i]) for j in range(x.shape[2])] )
            stdX[:,i]  = np.array( [np.std( x[:,:,j,i]) for j in range(x.shape[2])] )
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                if (stdX[i][j]!=0): 
                    selection = ((x[:,:,i,j]-meanX[i][j])/stdX[i][j]>-5) & ((x[:,:,i,j]-meanX[i][j])/stdX[i][j]<5)
                x1 = x[:,:,i,j][selection]
                meanX[i][j] = np.mean(x1)
                stdX[i][j] = np.std(x1)


        meanY = np.array( [np.mean(y[:,0,j]) for j in range(y.shape[-1])] )
        stdY  = np.array( [np.std( y[:,0,j]) for j in range(y.shape[-1])] )
        meanYerr = np.array( [np.mean(y[:,1,j]) for j in range(y.shape[-1])] )
        stdYerr  = np.array( [np.std( y[:,1,j]) for j in range(y.shape[-1])] )
        for i in range(y.shape[-1]):
            if (stdY[i]!=0): 
                selection = ((y[:,0,i]-meanY[i])/stdY[i]>-5) & ((y[:,0,i]-meanY[i])/stdY[i]<5)
            y1 = y[selection,0,i]
            y1err = y[selection,1,i]
            meanY[i] = np.mean(y1)
            stdY[i] = np.std(y1)
            meanYerr[i] = np.mean(y1err)
            stdYerr[i] = np.std(y1err)

            
        
        #__ if you have bad data sometimes in one of the columns - 
        # - you can calculate mean and std without these bad entries
        #   and then make them = 0 -> no effect on the first layer
        #for i in range(len(self.poorColumnValues)):
        #    cPoor = df.columns.get_loc(self.poorColumnValues[i][0])
        #    vPoor = self.poorColumnValues[i][1]
        #    mean[cPoor] = np.mean(x[(x[:,cPoor]!=vPoor),cPoor])
        #    std[cPoor] = np.std(x[(x[:,cPoor]!=vPoor),cPoor])

        mean = np.concatenate((meanX.ravel(),meanY,meanYerr))
        std = np.concatenate((stdX.ravel(),stdY,stdYerr))

        return mean, std


    def normalizeDatasetScale(self, df):
        #print(df)
        cellsLength, channelsLength = Config().cellsLength, Config().channelsLength

        sentenceLengthSource = 1

        meanValues, stdValues = readTrainData(f"{self.mainPath}/function_prediction","")

        meanValuesFeatures = meanValues[:cellsLength*channelsLength].reshape((channelsLength,cellsLength))

        meanValuesTargets = meanValues[cellsLength*channelsLength:]

        columns = list(df.columns)

        featureColumns = np.arange(1,len(columns)-2*cellsLength).reshape(sentenceLengthSource,channelsLength,cellsLength)
        targetColumns  = np.arange(len(columns)-2*cellsLength, len(columns)).reshape(2,cellsLength)

        for i in range(channelsLength):
            for j in range(sentenceLengthSource):
                for k in range(cellsLength):
                    df[columns[featureColumns[j,i,k]]] = df[columns[featureColumns[j,i,k]]]/compute_scaling_factor(meanValuesFeatures[i][k])

        for i in range(cellsLength):
            df[columns[targetColumns[0,i]]] = (df[columns[targetColumns[0,i]]])/compute_scaling_factor(meanValuesTargets[i])
            df[columns[targetColumns[1,i]]] = (df[columns[targetColumns[1,i]]]/compute_scaling_factor(meanValuesTargets[i+cellsLength]))     # for target errors to be around 1 for custom MSE loss

        return df
    
    def normalizeDatasetNormal(self, df):
        #print(df)
        cellsLength, channelsLength = Config().cellsLength, Config().channelsLength

        sentenceLengthSource = 1

        meanValues, stdValues = readTrainData(f"{self.mainPath}/function_prediction","")

        meanValuesFeatures = meanValues[:cellsLength*channelsLength].reshape((channelsLength,cellsLength))
        stdValuesFeatures = stdValues[:cellsLength*channelsLength].reshape((channelsLength,cellsLength))

        meanValuesTargets = meanValues[cellsLength*channelsLength:]
        stdValuesTargets = stdValues[cellsLength*channelsLength:]
        #print(stdValuesTargets)

        columns = list(df.columns)
        masks = []
        #for i in range(len(self.poorColumnValues)):
        #    masks.append(df[self.poorColumnValues[i][0]]==self.poorColumnValues[i][1])

        featureColumns = np.arange(1,len(columns)-2*cellsLength).reshape(sentenceLengthSource,channelsLength,cellsLength)
        targetColumns  = np.arange(len(columns)-2*cellsLength, len(columns)).reshape(2,cellsLength)

        for i in range(channelsLength):
            for j in range(sentenceLengthSource):
                for k in range(cellsLength):
                    if (stdValuesFeatures[i][k] > 0.001):
                        df[columns[featureColumns[j,i,k]]] = (df[columns[featureColumns[j,i,k]]]-meanValuesFeatures[i][k])/stdValuesFeatures[i][k]
                    else:
                        df[columns[featureColumns[j,i,k]]] = (df[columns[featureColumns[j,i,k]]]-meanValuesFeatures[i][k])/1

        for i in range(cellsLength):
            df[columns[targetColumns[0,i]]] = (df[columns[targetColumns[0,i]]]-meanValuesTargets[i])/stdValuesTargets[i]
            df[columns[targetColumns[1,i]]] = (df[columns[targetColumns[1,i]]]/meanValuesTargets[i+cellsLength])     # for target errors to be around 1 for custom MSE loss
            
        #for i in range(len(self.poorColumnValues)):
        #    df[self.poorColumnValues[i][0]].mask(masks[i], 0, inplace=True)

        return df

    def cutDataset(self, df0, threshold=8):
        #print("CUTTING")
        cellsLength = Config().cellsLength
        columns = list(df0.columns)
        df = df0.copy()
        df = self.normalizeDatasetNormal(df)

        cols = df.columns[1:-Config().cellsLength]
        sel = (df[cols] > -threshold) & (df[cols] < threshold)
        keep_rows = sel.all(axis=1)

        for i in range(len(columns)-2 - 2*cellsLength):
            data = (df.loc[keep_rows].copy())[columns[i+2]].to_numpy()
            #hist, bins = np.histogram(data, bins=100)
            #plt.figure(figsize=(8, 6))
            #plt.hist(data, bins=bins, color='skyblue', edgecolor='black')  # Plot the histogram
            #plt.title('Histogram of Random Data')
            #plt.xlabel('Value')
            #plt.ylabel('Frequency')
            #plt.grid(True)
            #plt.show()

        return df0.loc[keep_rows].reset_index(drop=True)


    def getDataset(self, rootPath):
        # read data, select raws (pids) and columns (drop)

        setTable = pandas.read_table(rootPath,sep=' ',header=None)
        print(setTable)
        columns = setTable.columns


        ### recombine pressure values to sum and difference (overpressure is in Pa, atm pressure is in mBar)
        #for i in range(24):
        #    newPressureSum = setTable[columns[1+i]]+setTable[columns[1+i+24*3]]/100
        #    newPressureDif = setTable[columns[1+i]]-setTable[columns[1+i+24*3]]/100
        #    setTable[columns[1+i]] = newPressureSum
        #    setTable[columns[1+i+24*3]] = newPressureDif


        ### drop runs with low duration, with difference between run (columns[0]) and the next run is low
        #runsDuration = setTable[columns[0]].shift(-1) - setTable[columns[0]]
        #setTable = setTable[runsDuration>20].copy()



        ### drop columns that are not needed    
        #for i in range(7+2):
        #    for j in range(12):
        #        print(columns[24*i+j+12+1])
        #        setTable = setTable.drop(columns[24*i+j+12+1], axis=1)
        #print(setTable)
        #for i in range(12):
        #    setTable = setTable.drop(columns[24*1+i+1], axis=1)
        #    setTable = setTable.drop(columns[24*2+i+1], axis=1)
        #    setTable = setTable.drop(columns[24*4+i+1], axis=1)
        #    setTable = setTable.drop(columns[24*5+i+1], axis=1)
        #    setTable = setTable.drop(columns[24*6+i+1], axis=1)
        #print(setTable)
        

        return setTable
    

    def manageDataset(self, mod,ind):
        pathFP = self.mainPath + 'function_prediction/'

        #print(str(Path(__file__).parents[1]))

        #dftCorr = self.getDataset(self.mainPath + "nn_input/outNNTestSMzxc.dat")
        #dftCorr = self.getDataset(self.mainPath + "nn_input/outNNFitTarget.dat")      # main that was used before cosmic tries
        dftCorr = self.getDataset(self.mainPath + "nn_input/outNNFitTargetRunEnds9pars.dat")      # new with 9 pars and run ends, doesn't work well?
        #dftCorr = self.getDataset(self.mainPath + "nn_input/outNNFitTargetCosmic25.dat")     # for cosmics
        #dftCorr = self.getDataset(self.mainPath + "nn_input/outNNFitTargetExtended.dat")
        #print(dftCorr)
        

        baseTrainRange = int(dftCorr.shape[0]*0.7)
        leftRange = int((dftCorr.shape[0]*0.3-10))
        retrainIndex = ind
        retrain0 = baseTrainRange + leftRange*(retrainIndex-3)
        #retrain0 = max(max(baseTrainRange + 200*(int(retrainIndex/2)-4), baseTrainRange + 100*(int(retrainIndex)-10)), int(baseTrainRange/20))
        #retrain1 = baseTrainRange + leftRange*retrainIndex
        #retrain2 = baseTrainRange + leftRange*(retrainIndex+1)
        retrain1 = int(baseTrainRange*0.7 + leftRange*retrainIndex)
        retrain2 = int(baseTrainRange)
        if (retrainIndex == 0):
            #dftTV = dftCorr.iloc[int(dftCorr.shape[0]*0.2):baseTrainRange].copy()
            dftTV = dftCorr.iloc[:baseTrainRange].copy()
        else:
            dftTV = dftCorr.iloc[retrain0:retrain1].copy()
        dftTrainV = dftTV.copy()
        dftTest = dftCorr.iloc[retrain1:retrain2].copy()

        print("dftCorr length is " + str(dftCorr.shape[0]) + "  train is " + str(dftTrainV.shape[0]) + "  range test is " + str(retrain1) + " - " + str(retrain2))

        mean, std = 0, 0
        if (mod.startswith("train")):
            mean, std = self.meanAndStdTable(dftTrainV)
            writeTrainData(mean,std, pathFP, "")
        elif (mod.startswith("test")):
            mean, std = readTrainData(pathFP,"")

        dftTrainV = self.cutDataset(dftTrainV, threshold=10).copy()
        dftTest = self.cutDataset(dftTest,  threshold=10).copy()

        # additional recalculation of mean with cut dataset
        if (mod.startswith("train")):
            mean, std = self.meanAndStdTable(dftTrainV)
            writeTrainData(mean,std, pathFP, "")
            dftTrainV = self.cutDataset(dftTrainV)
            dftTest = self.cutDataset(dftTest)

        print(dftTrainV)
        print(dftTest)
        pq.write_table(pa.Table.from_pandas(dftTrainV), pathFP + 'simu.parquet')
        pq.write_table(pa.Table.from_pandas(dftTest), pathFP + 'tesu.parquet')

        dftTrainV1 = self.normalizeDatasetNormal(dftTrainV)

        mean1, std1 = self.meanAndStdTable(dftTrainV1)
        print("mean values: " + str(mean1))
        print("std  values: " + str(std1))


    def manageDatasetCosmics(self, mod,ind):
        pathFP = self.mainPath + 'function_prediction/'

        dir = str(Path(__file__).parents[1])
        #dftCorr = self.getDataset(self.mainPath + "nn_input/outNNFitTarget.dat")      # main that was used before cosmic tries
        #dftTrainV = self.getDataset(self.mainPath + "nn_input/outNNFitTargetCosmic25_10mins.dat")     # for cosmics
        dftTrainV = self.getDataset(self.mainPath + "nn_input/outNNFitTargetCosmic25_106_9.dat")     # for cosmics
        #print(dftCorr)
        
        #dftCorr = self.cutDataset(dftCorr).copy()

        dftTest = dftTrainV.copy()
        #dftCorr

        #print(dftCorr)
        #print(dftTest)

        mean, std = 0, 0
        if (mod.startswith("train")):
            mean, std = self.meanAndStdTable(dftTrainV)
            meanHV = mean[24:36]
            stdHV = std[24:36]
            #mean, std = readTrainData(pathFP,"")
            #mean[24:36]=meanHV
            #std[24:36]=stdHV
            writeTrainData(mean,std, pathFP, "")
        elif (mod.startswith("test")):
            mean, std = readTrainData(pathFP,"")

        #dftCorr = self.cutDataset(dftCorr).copy()
        #dftTest = self.cutDataset(dftTest).copy()

        print(dftTrainV)
        print(dftTest)
        pq.write_table(pa.Table.from_pandas(dftTrainV), pathFP + 'simuCosmic.parquet')
        pq.write_table(pa.Table.from_pandas(dftTest), pathFP + 'tesuCosmic.parquet')

        dftCorr1 = self.normalizeDatasetNormal(dftTrainV).copy()

        mean1, std1 = self.meanAndStdTable(dftCorr1)
        print("mean values: " + str(mean1))
        print("std  values: " + str(std1))
        #print(dftCorr)

        #pq.write_table(pa.Table.from_pandas(dftCorr), pathFP + 'simu1.parquetmanageDataset')
        #pq.write_table(pa.Table.from_pandas(dftTest), pathFP + 'tesu1.parquet')
        #print(dftCorr)




# --- Training Data I/O ---

def writeTrainData(mean_arr, std_arr, path, mod):
    np.savetxt(f"{path}/meanValuesT{mod}.txt", mean_arr, fmt='%s')
    np.savetxt(f"{path}/stdValuesT{mod}.txt", std_arr, fmt='%s')


def readTrainData(path, mod):
    mean = np.loadtxt(f"{path}/meanValuesT{mod}.txt")
    std = np.loadtxt(f"{path}/stdValuesT{mod}.txt")
    return mean, std

