
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
from models.model import Conv2dLSTMCell
from dataHandling import My_dataset, DataManager, load_dataset, readTrainData
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from tqdm.auto import tqdm
from tqdm import trange
from config import Config


def loadModel(config, input_dim, nClasses, path):
    if (config.modelType == "DNN"):
        nn_model = DNN(input_dim=input_dim, output_dim=nClasses, nLayers=3, nNeurons=200).type(torch.FloatTensor)
    elif (config.modelType == "LSTM"):
        nn_model = LSTM(input_dim=input_dim, embedding_dim=64, hidden_dim=64, output_dim=1, num_layers=1, sentence_length=15).type(torch.FloatTensor)
    elif (config.modelType == "ConvLSTM"):
        nn_model = Conv2dLSTM(input_size=1, hidden_size=4, kernel_size=(1,1), num_layers=1, bias=0, output_size=6)
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load(path+"tempModel.pt"))
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
    dftCorr.reset_index(drop=True, inplace=True)
    print("<ASDSADSAD")
    print(dftCorr)
    dftCorr.rename(columns={list(dftCorr.columns)[0] : 'run'}, inplace=True)
    runColumn = dftCorr['run']

    dftCorrExp = dftCorr.drop(list(dftCorr.columns)[0],axis=1).copy()
    exp_dataset = My_dataset(load_dataset(config, dftCorrExp))
    exp_dataLoader = DataLoader(exp_dataset, batch_size=512, drop_last=False)

    mean, std = readTrainData(path)
    mean = mean[-1]
    std = std[-1]
    

    #nClasses = dftCorrExp[list(dftCorrExp.columns)[-1]].nunique()
    nClasses = 1
    input_dim = exp_dataset[0][0].shape[-1]

    #load nn and predict
    nn_model = loadModel(config, input_dim, nClasses, path)
    nn_model.eval()

    dat_list = []
    tepoch = tqdm(exp_dataLoader)
    for i_step, (x, y) in enumerate(tepoch):
        tepoch.set_description(f"Epoch {1}")
        prediction = (nn_model(x).detach().numpy()[:,-1])*std + mean
        dat_list.append(pandas.DataFrame(prediction))

    fullPredictionList = pandas.concat(list(dat_list),ignore_index=True)
    pq.write_table(pa.Table.from_pandas(fullPredictionList), path + savePath+'.parquet')

    print(runColumn)
    print(fullPredictionList)
    fullPreductionListWithRun = pandas.concat([runColumn,fullPredictionList],axis=1)
    np.savetxt(path + savePath+'.txt', fullPreductionListWithRun.values)
    return fullPredictionList


def draw_predictions_spread(outputs):
    class_hist = outputs.to_numpy()

    bins = np.linspace(-5, 5, 20)

    plt.hist(class_hist[:,1]-class_hist[:,2], bins, color='#0504aa',
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
    indexes = dftable[list(dftable.columns)[0]].to_numpy()
    indexes2 = dftable2[list(dftable2.columns)[0]].to_numpy()
    nptable = dftable.to_numpy()
    nptable2 = dftable2.to_numpy()

    plt.plot(indexes, nptable[:,1], color='#0504aa', label = 'target test', marker='o', linestyle="None", markersize=0.8)
    plt.plot(indexes, nptable[:,2], color='#8b2522', label = 'prediction test', marker='o', linestyle="None", markersize=0.7)
    plt.plot(indexes2, nptable2[:,1], color='#0504aa', label = 'target train', marker='o', linestyle="None", markersize=0.8)
    plt.plot(indexes2, nptable2[:,2], color='#228B22', label = 'prediction train', marker='o', linestyle="None", markersize=0.7)
    #plt.legend(loc=[0.6,0.8])
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('run number')
    plt.ylabel('dE/dx, a.u.')
    plt.show()

def draw_pred_vs_target(dftable):
    nptable = dftable.to_numpy()
    h1 = plt.hist2d(nptable[:,1],nptable[:,2], bins = (60,30), cmin=5, cmap=plt.cm.jet)
    plt.colorbar(h1[3])
    plt.plot(np.arange(-3,3), np.arange(-3,3))
    plt.xlabel('target')
    plt.ylabel('prediction')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.show()


def analyseOutput(predFileName, experiment_path, predFileNameS, experiment_pathS):
    pT = pandas.read_parquet(predFileName+'.parquet')
    dftCorrExp = pandas.read_parquet(experiment_path+'.parquet').reset_index(drop=True)
    pT.rename(columns={list(pT.columns)[0] : '0'}, inplace=True)
    #print(pT)
    #print(dftCorrExp)

    localCNames = list(dftCorrExp.columns)
    for i in range(len(dftCorrExp.columns)-2):
        dftCorrExp.drop(localCNames[i+1],axis=1,inplace=True)
    dftCorrExp = dftCorrExp.join(pT[list(pT.columns)])


    pTS = pandas.read_parquet(predFileNameS+'.parquet')
    dftCorrExpS = pandas.read_parquet(experiment_pathS+'.parquet').reset_index(drop=True)
    pTS.rename(columns={list(pTS.columns)[0] : '0'}, inplace=True)

    localCNames = list(dftCorrExpS.columns)
    for i in range(len(dftCorrExpS.columns)-2):
        dftCorrExpS.drop(localCNames[i+1],axis=1,inplace=True)
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
predict_nn("tesu1",'predicted1', path)
predict_nn("simu1",'predicted', path)
analyseOutput(path+"predicted1",path+"tesu", path+"predicted",path+"simu")
#analyseOutput(path+"predictedSim.parquet",path+"simu.parquet")