
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from models.model import DNN
from models.model import LSTM
from models.model import Conv2dLSTM
from models.model import GCN
from dataHandling import My_dataset, Graph_dataset, DataManager, load_dataset, readTrainData
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from tqdm.auto import tqdm
from tqdm import trange
from config import Config


mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/"
dataManager = DataManager(mainPath)


def loadModel(config, input_dim, nClasses, path):
    if (config.modelType == "DNN"):
        nn_model = DNN(input_dim=input_dim[-1], output_dim=nClasses, nLayers=3, nNeurons=200).type(torch.FloatTensor)
    elif (config.modelType == "LSTM"):
        nn_model = LSTM(input_dim=input_dim[-1], embedding_dim=64, hidden_dim=64, output_dim=1, num_layers=1, sentence_length=input_dim[0]).type(torch.FloatTensor)
    elif (config.modelType == "ConvLSTM"):
        nn_model = Conv2dLSTM(input_size=(input_dim[-3],input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=(3,3), num_layers=1, bias=0, output_size=1)
    elif (config.modelType == "gConvLSTM"):
        nn_model = GCN(input_size=(input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=2, num_layers=1)
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load(path+"tempModelNew.pt"))
    return nn_model

def constructPredictionFrame(index,nparray):
    # return new DataFrame with 4 columns corresponding to probabilities of resulting classes and 5th column to chi2 
    df2 = pandas.DataFrame(index=index)
    for i in range(nparray.shape[1]):
        df2[str(i)] = nparray[:i].tolist()
    return df2


def checkDistributions():
    dfm = pandas.read_table("testSet" + "Mod" + ".txt",sep='	',header=None)
    dfe = pandas.read_table("testSet" + "Exp" + ".txt",sep='	',header=None)
    numberOfColumns = len(dfm.columns)
    columnNames = list(dfe.columns)
    dfe[(dfe[columnNames[10]]==1000.0) & (dfe[columnNames[0]]<0.35)][columnNames[2]].plot(kind='kde')
    dfm[(dfm[columnNames[10]]==1000.0) & (dfm[columnNames[0]]<0.35)][columnNames[2]].plot(kind='kde')
    plt.show()


def makePredicionList(config, experiment_path, savePath, path):
    dftCorr = pandas.read_parquet(path+ experiment_path+'.parquet')
    dftCorr = dataManager.normalizeDataset(dftCorr)
    dftCorr.reset_index(drop=True, inplace=True)
    print("<ASDSADSAD")
    print(dftCorr)
    dftCorr.rename(columns={list(dftCorr.columns)[0] : 'run'}, inplace=True)
    runColumn = dftCorr['run']

    dftCorrExp = dftCorr.drop(list(dftCorr.columns)[0],axis=1).copy()
    exp_dataset = My_dataset(load_dataset(config, dftCorrExp))
    if (config.modelType == "gConvLSTM"):
        exp_dataset = Graph_dataset(load_dataset(config, dftCorrExp))
    exp_dataLoader = DataLoader(exp_dataset, batch_size=512, drop_last=False)

    mean, std = readTrainData(path)
    #mean = mean[-6:]
    #std = std[-6:]
    
    #shape exp_dataset: []

    #nClasses = dftCorrExp[list(dftCorrExp.columns)[-1]].nunique()
    nClasses = 1
    input_dim = exp_dataset[0][0].shape
    print("input shape is ",input_dim)

    #load nn and predict
    nn_model = loadModel(config, input_dim, nClasses, path)
    nn_model.eval()

    dat_list = []
    tepoch = tqdm(exp_dataLoader)
    for i_step, (x, y, e_i, e_a) in enumerate(tepoch):
        tepoch.set_description(f"Epoch {1}")
        if (config.modelType == "LSTM"):
            prediction = (nn_model(x).detach().numpy()[:,-1])*std[-24] + mean[-24]
            print(prediction.shape)
        elif(config.modelType == "ConvLSTM"):
            prediction = (nn_model(x).detach().numpy())[:,-1,:]
            n_nodes = input_dim[-2]
            for i in range(n_nodes):
                prediction[:,i] = prediction[:,i]*std[-n_nodes+i] + mean[-n_nodes+i]
            print(prediction)
        elif(config.modelType == "gConvLSTM"):
            prediction = (nn_model(x,e_i,e_a).detach().numpy())[:,-1,:]
            n_nodes = input_dim[-1]
            for i in range(n_nodes):
                prediction[:,i] = prediction[:,i]*std[-n_nodes+i] + mean[-n_nodes+i]
            print(prediction)
        dat_list.append(pandas.DataFrame(prediction))

    fullPredictionList = pandas.concat(list(dat_list),ignore_index=True)
    pq.write_table(pa.Table.from_pandas(fullPredictionList), path + savePath+'.parquet')

    #print(runColumn)
    #print(fullPredictionList)
    fullPreductionListWithRun = pandas.concat([runColumn,fullPredictionList],axis=1)
    np.savetxt(path + savePath+'.txt', fullPreductionListWithRun.values)
    return fullPredictionList


def draw_predictions_spread(outputs):
    class_hist = outputs.to_numpy()

    bins = np.linspace(-5, 5, 20)

    plt.hist(class_hist[:,1]-class_hist[:,25], bins, color='#0504aa',
                            alpha=0.7, rwidth=0.95, label = '$\Delta prediction$')
    plt.legend(loc=[0.6,0.8])
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('$\Delta prediction$')
    plt.show()

def draw_parameter_spread(tables, column):
    class_hist0 = tables[0][column].to_numpy()
    class_hist1 = tables[1][column].to_numpy()
    class_hist2 = tables[2][column].to_numpy()

    bins = np.linspace(-0.5,2.5,300)

    plt.hist(class_hist0[:], bins, color='#0504aa',
                            alpha=0.7, rwidth=1.0, label = '$\pi$')
    plt.hist(class_hist1[:], bins, color='#228B22',
                            alpha=0.7, rwidth=1.0, label = '$K$')
    plt.hist(class_hist2[:], bins, color='#cf4c00',
                            alpha=0.7, rwidth=1.0, label = '$P$')
    plt.xlabel('mass2')
    plt.show()

def draw_2d_param_spread(tables, column1, column2):
    class_hist0 = [tables[0][column1].to_numpy(),tables[0][column2].to_numpy()]
    class_hist1 = [tables[1][column1].to_numpy(),tables[1][column2].to_numpy()]
    class_hist2 = [tables[2][column1].to_numpy(),tables[2][column2].to_numpy()]

    #h = plt.hist2d(class_hist0[0],class_hist0[1], bins = 300, cmap = "RdYlBu_r", norm = colors.LogNorm())
    h1 = plt.hist2d(class_hist1[0],class_hist1[1], bins = 300, cmap="RdYlBu_r", norm = colors.LogNorm())
    plt.colorbar(h1[3])
    #plt.hist2d(class_hist2[0],class_hist2[1], bins = 300, cmap = "RdYlBu_r", norm = colors.LogNorm())
    #plt.ylim([0,1.5])
    plt.show()


def draw_pred_and_target_vs_run(dftable, dftable2):
    print(dftable)
    
    # Get length of spacial dimension (e.g. number of channels). Can be changed
    chLength = int(len(dftable.columns)/2)

    indexes = dftable[list(dftable.columns)[0]].to_numpy()
    indexes2 = dftable2[list(dftable2.columns)[0]].to_numpy()
    nptable = dftable.to_numpy()
    nptable2 = dftable2.to_numpy()

    mean, std = readTrainData(path)
    for i in range(10):

        # Get difference target-prediction in terms of sigma (std) for train and test parts
        nptable[:,i+1+chLength] = (nptable[:,i+1+chLength] - nptable[:,i+1])/std[-chLength+i]
        nptable2[:,i+1+chLength] = (nptable2[:,i+1+chLength] - nptable2[:,i+1])/std[-chLength+i]

        #plt.plot(indexes, nptable[:,i+1], color='#0504aa', label = 'target test'+str(i), marker='o', linestyle="None", markersize=0.8)
        plt.plot(indexes, nptable[:,i+1+chLength], color='#8b2522', label = 'prediction test'+str(i), marker='o', linestyle="None", markersize=1.7)
        #plt.plot(indexes2, nptable2[:,i+1], color='#0504aa', label = 'target train'+str(i), marker='o', linestyle="None", markersize=0.8)
        plt.plot(indexes2, nptable2[:,i+1+chLength], color='#228B22', label = 'prediction train'+str(i), marker='o', linestyle="None", markersize=1.7)
    
    #plt.legend(loc=[0.6,0.8])
    #plt.ylim(4, 7.5)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('run number')
    plt.ylabel('dE/dx, a.u.')
    plt.show()

def draw_pred_vs_target(dftable):
    nptable = dftable.to_numpy()
    h1 = plt.hist2d(nptable[:,1],nptable[:,2], bins = (60,30), cmin=5, cmap=plt.cm.jet)
    plt.colorbar(h1[3])
    plt.plot(np.arange(0,1000), np.arange(0,1000))
    plt.xlabel('target')
    plt.ylabel('prediction')
    #plt.xlim([-3, 3])
    #plt.ylim([-3, 3])
    plt.show()


def analyseOutput(predFileName, experiment_path, predFileNameS, experiment_pathS):
    pT = pandas.read_parquet(predFileName+'.parquet')
    dftCorrExp = pandas.read_parquet(experiment_path+'.parquet').reset_index(drop=True)
    pT.rename(columns={list(pT.columns)[0] : '0'}, inplace=True)
    print(pT)
    print(dftCorrExp)

    localCNames = list(dftCorrExp.columns)
    for i in range(len(localCNames)-25):
        dftCorrExp.drop(localCNames[i+1],axis=1,inplace=True)
    for i in range(0):
        dftCorrExp.drop(localCNames[len(localCNames) -i -1],axis=1,inplace=True)
    dftCorrExp = dftCorrExp.join(pT[list(pT.columns)])


    pTS = pandas.read_parquet(predFileNameS+'.parquet')
    dftCorrExpS = pandas.read_parquet(experiment_pathS+'.parquet').reset_index(drop=True)
    pTS.rename(columns={list(pTS.columns)[0] : '0'}, inplace=True)

    localCNames = list(dftCorrExpS.columns)
    for i in range(len(localCNames)-25):
        dftCorrExpS.drop(localCNames[i+1],axis=1,inplace=True)
    for i in range(0):
        dftCorrExpS.drop(localCNames[len(localCNames) - i -1],axis=1,inplace=True)
    dftCorrExpS = dftCorrExpS.join(pTS[list(pTS.columns)])
    print(dftCorrExpS)

    draw_predictions_spread(dftCorrExp)
    draw_pred_and_target_vs_run(dftCorrExp, dftCorrExpS)
    #draw_pred_vs_target(dftCorrExp)


def predict_nn(fName, oName, path):
    config = Config()

    predictionList = makePredicionList(config, fName, oName, path)
    #print(predictionList)

    #write_output(predictionList,mod,enlist)

mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/"
path = mainPath+"function_prediction/"

#print("start python predict")
predict_nn("tesu",'predicted1', path)
predict_nn("simu",'predicted', path)
analyseOutput(path+"predicted1",path+"tesu", path+"predicted",path+"simu")
#analyseOutput(path+"predictedSim.parquet",path+"simu.parquet")
