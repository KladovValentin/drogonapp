
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
from models.model import Conv2dLSTM
from models.model import GCNLSTM
from dataHandling import My_dataset, Graph_dataset, DataManager, load_dataset, readTrainData, compute_scaling_factor
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from tqdm.auto import tqdm
from tqdm import trange
from config import Config
import torch.jit



def loadModel(config, input_dim, nClasses, path, e_i=0,e_a=0):
    if (config.modelType == "ConvLSTM"):
        nn_model = Conv2dLSTM(input_size=(input_dim[-3],input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=(3,3), num_layers=1, bias=0, output_size=1)
    elif (config.modelType == "gConvLSTM"):
        nn_model = GCNLSTM(input_size=(input_dim[-2],input_dim[-1]), embedding_size=8, hidden_size=8, kernel_size=3, num_layers=1, e_i=e_i, e_a=e_a)
        #nn_model = GCNLSTM(input_size=(input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=2, num_layers=1)
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load(path+"tempModelT.pt"))
    nn_model.eval()

    if (config.modelType == "DNN"):
        modelRandInput = torch.randn(1, input_dim[-1])
    elif (config.modelType == "LSTM"):
        modelRandInput = torch.randn(1, input_dim[0], input_dim[-1])
    elif (config.modelType == "ConvLSTM"):
        modelRandInput = torch.randn(1, input_dim[0], input_dim[-3], input_dim[-2], input_dim[-1])
    elif (config.modelType == "gConvLSTM"):
        n_nodes = input_dim[-1]
        in_channels = input_dim[-2]
        sent_length = input_dim[-3]
        #e_i_r = torch.randint(0,n_nodes-1,(1,edge_length,2))
        #e_a_r = torch.ones(1,edge_length)
        modelRandInput = (torch.randn(1, sent_length, in_channels, n_nodes))
    #torch.onnx.export(nn_model,                                # model being run
    #                  modelRandInput,    # model input (or a tuple for multiple inputs)
    #                  mainPath+"function_prediction/tempModel.onnx",           # where to save the model (can be a file or file-like object)
    #                  input_names = ["input"],              # the model's input names
    #                  output_names = ["output"], opset_version=11)            # the model's output names
    #scripted_model = torch.jit.script(nn_model)
    #scripted_model.save("gcn_model.pt")
    #torch.jit.save(torch.jit.script(nn_model), 'gcn_model.pt')
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


def getHV(model,X):
    # Example input X with features in each node (batch_size, num_nodes, feature_dim)

    feature_index = 1  # The specific feature in the node you want to modify

    # Set requires_grad=True for the specific feature in the specific node
    x_i = X[:, -1, feature_index, :].clone().detach().requires_grad_(True)
    #X.requires_grad_(True)

    # Optimizer setup to optimize only the specific feature at node_index and feature_index
    print(X.shape)
    optimizer = optim.Adam([x_i], lr=0.1)

    # Loss function
    criterion = nn.MSELoss()

    # Number of iterations for the optimization process
    num_iterations = 1000

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        
        Y_fixed = torch.zeros(X.size(0), 24)  # Normalized target value (mean=0, std=1)

        X[:, -1, feature_index, :] = x_i
        
        Y_pred = (model(X))[:,-1,:]  # Forward pass through the model

        loss = criterion(Y_pred, Y_fixed)  # Compute the loss between predicted Y and Y_fixed

        loss.backward(retain_graph=True)  # Backward pass to compute the gradients
        
        optimizer.step()  # Update the specific feature at node_index and feature_index

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}, Y(x_i): {Y_pred[0,0].item()}")

    # After optimization, X[:, -1, :, feature_index] should be the value that produces Y_fixed
    # Shape = (batch, nodes)
    final_x_i = x_i.item()
    return final_x_i, Y_pred


def makePredicionList(config, experiment_path, savePath, path):
    dftCorr = pandas.read_parquet(path+ experiment_path+'.parquet')
    #dftCorr = dataManager.normalizeDatasetNormal(dftCorr)
    dftCorr = dataManager.normalizeDatasetScale(dftCorr)
    dftCorr.reset_index(drop=True, inplace=True)
    print("<ASDSADSAD")
    print(dftCorr)
    dftCorr.rename(columns={list(dftCorr.columns)[0] : 'run'}, inplace=True)
    runColumn = dftCorr['run']

    #dftCorrExp = dftCorr.drop('run',axis=1).copy()             
    dftCorrExp = dftCorr.copy()
    #exp_dataset = My_dataset(load_dataset(config, dftCorrExp))
    #if (config.modelType == "gConvLSTM"):
    xt, yt, e_i, e_a = load_dataset(config,dftCorrExp)
    exp_dataset = My_dataset((xt,yt))
    #exp_dataset = Graph_dataset(load_dataset(config, dftCorrExp))
    exp_dataLoader = DataLoader(exp_dataset, batch_size=256, drop_last=False)

    mean, std = readTrainData(path,"")
    #mean = mean[-6:]
    #std = std[-6:]
    
    #shape exp_dataset: []

    #nClasses = dftCorrExp[list(dftCorrExp.columns)[-1]].nunique()
    nClasses = 1
    input_dim = exp_dataset[0][0].shape
    print("input shape is ",input_dim)

    #load nn and predict
    if (config.modelType == "gConvLSTM"):
        nn_model = loadModel(config, input_dim, nClasses, path, e_i=(torch.LongTensor(e_i).movedim(-2,-1)), e_a=torch.Tensor(e_a))
    #else:
    #nn_model = loadModel(config, input_dim, nClasses, path)
    nn_model.eval()

    dat_list = []
    hv_list = []
    stable_dat_list = []
    tepoch = tqdm(exp_dataLoader)
    
    print("")
    print("")
    for i_step, (x, y) in enumerate(tepoch):
        tepoch.set_description(f"Epoch {1}")
        if(config.modelType == "ConvLSTM"):
            prediction = (nn_model(x).detach().numpy())[:,-1,:]
            n_nodes = input_dim[-2]
            for i in range(n_nodes):
                prediction[:,i] = prediction[:,i]*std[-n_nodes+i] + mean[-n_nodes+i]
            #print(prediction)
        elif(config.modelType == "gConvLSTM"):
            #print(x)
            prediction = (nn_model(x).detach().numpy())[:,-1,:]
            #predictedHV, predictionWithHV = getHV(nn_model,x)
            n_nodes = config.cellsLength
            for i in range(input_dim[-1]):
                #prediction[:,i] = prediction[:,i]*std[-2*n_nodes+i] + mean[-2*n_nodes+i]
                prediction[:,i] = prediction[:,i]*compute_scaling_factor(mean[-2*n_nodes+i])
                #prediction[:,i] = prediction[:,i]*std[-24*2+i] + mean[-24*2+i]
            #print(prediction)
        dat_list.append(pandas.DataFrame(prediction))
        #hv_list.append(pandas.DataFrame(predictedHV))
        #stable_dat_list.append(pandas.DataFrame(predictionWithHV))

    fullPredictionList = pandas.concat(list(dat_list),ignore_index=True)
    pq.write_table(pa.Table.from_pandas(fullPredictionList), path + "predicted/" + savePath + '.parquet')

    #print(runColumn)
    #print(fullPredictionList)
    fullPreductionListWithRun = pandas.concat([runColumn,fullPredictionList],axis=1)
    np.savetxt(path + "predicted/" + savePath+'.txt', fullPreductionListWithRun.values)


    #fullHVList = pandas.concat(list(hv_list),ignore_index=True)
    #fullHVListWithRun = pandas.concat([runColumn,fullHVList],axis=1)
    #np.savetxt(path + "predicted/" + savePath+'HV.txt', fullHVListWithRun.values)

    #fullPredictionHVList = pandas.concat(list(stable_dat_list),ignore_index=True)
    #fullPredictionHVListWithRun = pandas.concat([runColumn,fullPredictionHVList],axis=1)
    #np.savetxt(path + "predicted/" + savePath+'Stable.txt', fullPredictionHVListWithRun.values)
    
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
    chLength = int(len(dftable.columns)/3)

    indexes = dftable[list(dftable.columns)[0]].to_numpy()
    indexes2 = dftable2[list(dftable2.columns)[0]].to_numpy()
    nptable = dftable.to_numpy()
    nptable2 = dftable2.to_numpy()

    print(nptable.shape)
    print(nptable2.shape)


    mean, std = readTrainData(path,"")
    for i in range(1):
        i = i
        # Get difference target-prediction in terms of sigma (std) for train and test parts
        #nptable[:,i+1+2*chLength] = (nptable[:,i+1+2*chLength] - nptable[:,i+1])/nptable[:,i+1+chLength]
        #nptable2[:,i+1+2*chLength] = (nptable2[:,i+1+2*chLength] - nptable2[:,i+1])/nptable2[:,i+1+chLength]

        plt.plot(indexes, nptable[:,i+1], color='#0504aa', label = 'target test'+str(i), marker='o', linestyle="None", markersize=0.8)
        plt.plot(indexes, nptable[:,i+1+2*chLength], color='#8b2522', label = 'prediction test'+str(i), marker='o', linestyle="None", markersize=1.7)
        plt.plot(indexes2, nptable2[:,i+1], color='#0504aa', label = 'target train'+str(i), marker='o', linestyle="None", markersize=0.8)
        plt.plot(indexes2, nptable2[:,i+1+2*chLength], color='#228B22', label = 'prediction train'+str(i), marker='o', linestyle="None", markersize=1.7)
    

    #with open("testPred.txt","a") as file:
    #    for i in range(indexes.size):
    #        for j in range(24):
    #            file.write(indexes[i]+" "+nptable[i,j+1+chLength]+"\n")

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

    nNodes = 24
    localCNames = list(dftCorrExp.columns)
    for i in range(len(localCNames)-1-2*nNodes):
        dftCorrExp.drop(localCNames[i+1],axis=1,inplace=True)
    #for i in range(nNodes):
    #    dftCorrExp.drop(localCNames[len(localCNames) -i -1],axis=1,inplace=True)
    dftCorrExp = dftCorrExp.join(pT[list(pT.columns)])


    pTS = pandas.read_parquet(predFileNameS+'.parquet')
    dftCorrExpS = pandas.read_parquet(experiment_pathS+'.parquet').reset_index(drop=True)
    pTS.rename(columns={list(pTS.columns)[0] : '0'}, inplace=True)


    localCNames = list(dftCorrExpS.columns)
    for i in range(len(localCNames)-1-2*nNodes):
        dftCorrExpS.drop(localCNames[i+1],axis=1,inplace=True)
    #for i in range(nNodes):  
    #    dftCorrExpS.drop(localCNames[len(localCNames) - i -1],axis=1,inplace=True)
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

def predict_cicle(testNum):
    mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/"
    predict_nn("tesu",'predicted1_'+str(testNum), path)
    predict_nn("simu",'predicted_'+str(testNum), path)


mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/"
dataManager = DataManager(mainPath)
path = mainPath+"function_prediction/"

#print("start python predict")
test = 0
predict_nn("tesu",'predicted1_'+str(test), path)
predict_nn("simu",'predicted_'+str(test), path)

#analyseOutput(path+"predicted/predicted1_"+str(test),path+"tesu", path+"predicted/predicted_"+str(test),path+"simu")
#analyseOutput(path+"predictedSim.parquet",path+"simu.parquet")


#predict_nn("tesuCosmic",'predictedCosmics1_', path)
#predict_nn("simuCosmic",'predictedCosmics_', path)

#analyseOutput(path+"predicted/predictedCosmics1_",path+"tesuCosmic", path+"predicted/predictedCosmics_",path+"simuCosmic")
#analyseOutput(path+"predictedCosmics.parquet",path+"simu.parquet")
