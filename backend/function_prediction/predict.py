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
from models.model import GraphTransformer2D
from dataHandling import My_dataset, Graph_dataset, DataManager, load_dataset, readTrainData, compute_scaling_factor
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from tqdm.auto import tqdm
from tqdm import trange
from config import Config
import torch.jit
from scipy.stats import norm
import time
import locale
locale.setlocale(locale.LC_NUMERIC, "C")
import ROOT

activations = {}

def capture_input(name):
    def hook(module, input, output):
        if name not in activations:
            activations[name] = []
        activations[name].append(input[0].detach().cpu())
    return hook


def loadModel(config, input_dim, nClasses, path, e_i=0,e_a=0):
    if (config.modelType == "ConvLSTM"):
        nn_model = Conv2dLSTM(input_size=(input_dim[-3],input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=(3,3), num_layers=1, bias=0, output_size=1)
    elif (config.modelType == "gConvLSTM"):
        nn_model = GCNLSTM(input_dims=(input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=3, num_layers=1, e_i=e_i, e_a=e_a)
        #nn_model = GCNLSTM(input_size=(input_dim[-2],input_dim[-1]), embedding_size=8, hidden_size=16, kernel_size=3, num_layers=1, e_i=e_i, e_a=e_a)
        #nn_model = GraphTransformer2D(input_size = input_dim[-2], embedding_size=8, num_nodes = input_dim[-1], sentence_length = input_dim[-3], e_i=e_i, e_a=e_a)
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
    model.eval()
    nodes_length = 24
    # Example input X with features in each node (batch_size, num_nodes, feature_dim)

    feature_index = 1  # The specific feature in the node you want to modify

    # Set requires_grad=True for the specific feature in the specific node
    x_i = X[:, -1, feature_index, :].clone().detach().requires_grad_(True)
    #X.requires_grad_(True)

    # Optimizer setup to optimize only the specific feature at node_index and feature_index
    print(X.shape)
    optimizer = optim.Adam([x_i], lr=0.01)

    # Loss function
    criterion = nn.MSELoss()

    # Number of iterations for the optimization process
    num_iterations = 10   # ~300 for a normal operation

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        
        Y_fixed = torch.zeros(X.size(0), nodes_length)  # Normalized target value (mean=0, std=1)

        # Create a copy of X and replace HV feature
        X_temp = X.clone().detach()  # avoid modifying original X
        X_temp[:, -1, feature_index, :] = x_i
        
        Y_pred = (model(X_temp))[:,-1,:]  # Forward pass through the model   shape (batch, nodes)

        loss = criterion(Y_pred, Y_fixed)  # Compute the loss between predicted Y and Y_fixed

        loss.backward(retain_graph=True)  # Backward pass to compute the gradients
        
        optimizer.step()  # Update the specific feature at node_index and feature_index

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}, Y(x_i): {Y_pred[0,0].item()}")

    # After optimization, X[:, -1, :, feature_index] should be the value that produces Y_fixed
    # Shape = (batch, nodes)
    final_x_i = x_i
    return final_x_i.clone().detach(), Y_pred


def makePredicionList(config, dataManager, experiment_path, savePath, path):
    dftCorr = pandas.read_parquet(path+ experiment_path+'.parquet')
    #dftCorr shape is (batch, 1+features*nodes+2*nodes)

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
    exp_dataLoader = DataLoader(exp_dataset, batch_size=256*4, drop_last=False)

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


    # Register the hook on the whole nn_model block
    #nn_model.linear.register_forward_hook(capture_input("input_to_nn_model"))
    #for i, submodel in enumerate(nn_model.nn_model):
    #    submodel[0].register_forward_hook(capture_input(f"nn_model_branch_{i}"))

    dat_list = []
    hv_list = []
    stable_dat_list = []
    tepoch = tqdm(exp_dataLoader)
    
    print("")
    print("")
    start_time = time.perf_counter()
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
            predictedHV, predictionWithHV = getHV(nn_model,x)
            n_nodes = config.cellsLength
            for i in range(input_dim[-1]):
                #prediction[:,i] = prediction[:,i]*std[-2*n_nodes+i] + mean[-2*n_nodes+i]
                #prediction[:,i] = prediction[:,i]*compute_scaling_factor(mean[-2*n_nodes+i])
                #prediction[:,i] = prediction[:,i]*100.0
                prediction[:,i] = prediction[:,i]*std[-24*2+i] + mean[-24*2+i]
                #predictionWithHV[:,i] = predictionWithHV[:,i]*std[-2*n_nodes+i] + mean[-2*n_nodes+i]
                predictedHV[:,i] = predictedHV[:,i]*compute_scaling_factor(mean[1*n_nodes+i])
            #print(prediction)
        dat_list.append(pandas.DataFrame(prediction))
        hv_list.append(pandas.DataFrame(predictedHV))
        stable_dat_list.append(pandas.DataFrame(predictionWithHV.detach().cpu().numpy()))
    end_time = time.perf_counter()

    fullPredictionList = pandas.concat(list(dat_list),ignore_index=True)
    pq.write_table(pa.Table.from_pandas(fullPredictionList), path + "predicted/" + savePath + '.parquet')

    #print(runColumn)
    #print(fullPredictionList)
    fullPreductionListWithRun = pandas.concat([runColumn,fullPredictionList],axis=1)
    np.savetxt(path + "predicted/" + savePath+'.txt', fullPreductionListWithRun.values)


    fullHVList = pandas.concat(list(hv_list),ignore_index=True)
    fullHVListWithRun = pandas.concat([runColumn,fullHVList],axis=1)
    np.savetxt(path + "predicted/" + savePath+'HV.txt', fullHVListWithRun.values)

    fullPredictionHVList = pandas.concat(list(stable_dat_list),ignore_index=True)
    fullPredictionHVListWithRun = pandas.concat([runColumn,fullPredictionHVList],axis=1)
    np.savetxt(path + "predicted/" + savePath+'Stable.txt', fullPredictionHVListWithRun.values)


    #plotHiddenFeatures()

    #inject_and_predict_hv_sweep(nn_model, config, dataManager, path, experiment_path)

    print(f"Prediction time: {end_time - start_time:.6f} seconds")
    print(f"Time per prediction: {(end_time - start_time)*1000.0/xt.shape[0]:.6f} ms")
    
    return fullPredictionList


def plotHiddenFeatures():
    activationsT = activations.copy()
    for key in activations:
        activationsT[key] = torch.cat(activationsT[key], dim=0)  # Now shape: (total_batches, features)
    feat = activationsT[f"nn_model_branch_{0}"]
    print(feat.shape)
    plt.hist(feat[:,10].numpy(), bins=100)
    plt.title("Feature Distribution: Input to nn_model")
    plt.show()


def inject_and_predict_hv_sweep(model, config, dataManager, path, experiment_path):
    activations.clear()  # Clear previous activations
    #for i, submodel in enumerate(model.nn_model):
    #    submodel[0].register_forward_hook(capture_input(f"nn_model_branch_{i}"))
    # Parameters
    hv_values = list(range(1650, 1800, 10))  # 1700 to 1800 inclusive
    subset_size = 30
    nodes_length = 12
    hv_channel_index = 1
    cellsLength = config.cellsLength

    df = pandas.read_parquet(path+ experiment_path+'.parquet')

    # Get middle 30 runs
    n_rows = df.shape[0]
    print(n_rows)
    mid = n_rows // 2
    df_subset = df.iloc[max(mid - subset_size // 2,0): mid + subset_size // 2].copy()

    run_column = df_subset.iloc[:, 0].reset_index(drop=True)
    
    # Column indices for HV in the flat dataframe
    hv_column_indices = list(range(1 + hv_channel_index * cellsLength, 1 + (hv_channel_index+1) * cellsLength))

    # Load training mean/std for renormalization
    mean, std = readTrainData(path, "")

    for hv in hv_values:
        print(f"Running prediction for HV = {hv} V")

        df_hv = df_subset.copy()
        for col in hv_column_indices:
            df_hv.iloc[:, col] = hv

        # Normalize input
        df_hv_norm = dataManager.normalizeDatasetScale(df_hv)

        # Load and shape data
        xt, _, e_i, e_a = load_dataset(config, df_hv_norm)
        x_tensor = torch.tensor(xt.astype(np.float32))
        e_i_torch = torch.LongTensor(e_i).movedim(-2, -1)
        e_a_torch = torch.Tensor(e_a)

        #print(x_tensor)

        # Reload model in eval mode
        model.eval()
        with torch.no_grad():
            prediction = model(x_tensor).numpy()[:, -1, :]  # shape (batch, nodes)
        #print(prediction)

        # Un-normalize prediction
        for i in range(nodes_length):
            prediction[:, i] = prediction[:, i] * std[-2 * cellsLength + i] + mean[-2 * cellsLength + i]
            #prediction[:, i] = prediction[:, i] * 100

        #print(prediction)

        output_with_run = np.concatenate([run_column.to_numpy().reshape(-1, 1), prediction[:, :nodes_length]], axis=1)

        # Save results
        np.savetxt(path + "predicted/" + 'predicted_'+str(hv)+'.txt',  output_with_run, fmt="%.5f")
    #plotHiddenFeatures()
    #plot_hv_channel_sensitivity(model, x_tensor, hv_index=hv_channel_index)


def plot_hv_channel_sensitivity(model, x_tensor, hv_index=1):
    x_tensor.requires_grad_(True)
    output = model(x_tensor)
    output.mean().backward()

    grads = x_tensor.grad[:, -1, hv_index, :]  # sentence[-1], HV feature, all nodes
    sensitivity = grads.abs().mean(dim=0).detach().numpy()

    import matplotlib.pyplot as plt
    plt.bar(range(len(sensitivity)), sensitivity)
    plt.xlabel("Node index")
    plt.ylabel("Gradient sensitivity to HV")
    plt.title("Per-node HV sensitivity in final forward pass")
    plt.show()

def plot_tot_vs_hv_and_heatmap(path, base_filename="predicted_", hv_range=range(1650, 1800, 10), cellsLength=12):
    """
    Plots:
    1. ToT vs HV (line plot per chamber)
    2. Heatmap of ToT per HV and chamber
    """
    hv_values = list(hv_range)
    all_predictions = []

    for hv in hv_values:
        file_path = path + "predicted/" + 'predicted_'+str(hv)+'.txt'
        pred = np.loadtxt(file_path)  # shape: (30, cellsLength)
        mean_tot = np.mean(pred[:,1:], axis=0)  # shape: (cellsLength,)
        all_predictions.append(mean_tot)

    all_predictions = np.array(all_predictions)  # shape: (len(hv_values), cellsLength)

    # --- 1. Line plot: ToT vs HV per chamber
    plt.figure(figsize=(10, 6))
    for ch in range(cellsLength):
        plt.plot(hv_values, all_predictions[:, ch], label=f'Chamber {ch}')
    plt.xlabel("High Voltage [V]")
    plt.ylabel("Predicted ToT [a.u.]")
    plt.title("ToT vs HV per Chamber")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

    # --- 2. Heatmap: HV vs Chamber
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(all_predictions.T, aspect='auto', origin='lower',
                    extent=[min(hv_values), max(hv_values), 0, cellsLength],
                    cmap="viridis")
    plt.colorbar(im, label="Mean ToT")
    plt.xlabel("High Voltage [V]")
    plt.ylabel("Chamber Index")
    plt.title("Heatmap of ToT per HV and Chamber")
    plt.tight_layout()
    plt.show()
    """



def draw_predictions_spread(outputs):
    class_hist = outputs.to_numpy()

    bins = np.linspace(-5, 5, 20)

    plt.hist(class_hist[:,1]-class_hist[:,25], bins, color='#0504aa',
                            alpha=0.7, rwidth=0.95, label = '$\Delta prediction$')
    plt.legend(loc=[0.6,0.8])
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('$\Delta prediction$')
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
    #chLength = int(len(dftable.columns))
    chLength = 24

    indexes = dftable[list(dftable.columns)[0]].to_numpy()
    indexes2 = dftable2[list(dftable2.columns)[0]].to_numpy()
    nptable = dftable.to_numpy()
    nptable2 = dftable2.to_numpy()

    #print(nptable[:,1+1])
    #print(nptable[:,1+1+2*chLength])
    #print(nptable2[:,1+1])
    #print(nptable2[:,1+1+2*chLength])


    mean, std = readTrainData(path,"")
    for i in range(1):
        i = 5
        # Get difference target-prediction in terms of sigma (std) for train and test parts
        #nptable[:,i+1+2*chLength] = (nptable[:,i+1+2*chLength] - nptable[:,i+1])/nptable[:,i+1+chLength]
        #nptable2[:,i+1+2*chLength] = (nptable2[:,i+1+2*chLength] - nptable2[:,i+1])/nptable2[:,i+1+chLength]

        plt.plot(indexes2, nptable2[:,i+1], color='#0504aa', label = 'target test'+str(i), marker='o', linestyle="None", markersize=0.8)
        plt.plot(indexes2, nptable2[:,i+1+2*chLength], color='#8b2522', label = 'prediction test'+str(i), marker='o', linestyle="None", markersize=1.7)
        plt.plot(indexes, nptable[:,i+1], color='#0504aa', label = 'target train'+str(i), marker='o', linestyle="None", markersize=0.8)
        plt.plot(indexes, nptable[:,i+1+2*chLength], color='#228B22', label = 'prediction train'+str(i), marker='o', linestyle="None", markersize=1.7)

        plt.show()
    

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



def draw_predictions_minus_target(dftable, dftable2):

    # Get length of spacial dimension (e.g. number of channels). Can be changed
    #chLength = int(len(dftable.columns))
    chLength = 24

    indexes = dftable[list(dftable.columns)[0]].to_numpy()
    indexes2 = dftable2[list(dftable2.columns)[0]].to_numpy()
    nptable = dftable.to_numpy()
    nptable2 = dftable2.to_numpy()

    bins = np.linspace(-5, 5, 200)

    mean, std = readTrainData(path,"")
    rangeNodes = 12
    rms_train_list = []
    rms_test_list = []
    rolling_std_window = 5  # Adjust this as needed
    rolling_std_train = []
    rolling_std_train_targets = []
    rolling_std_test_targets = []
    rolling_std_test = []

    htrain_diff = ROOT.TH1D("diffClass", "", 200, -5, 5)
    htrain_diff.GetXaxis().SetTitle("#Delta ToT, #sigma")
    htrain_diff.GetYaxis().SetTitle("Counts")
    htrain_diff.GetXaxis().SetTitleSize(0.05)
    htrain_diff.GetYaxis().SetTitleSize(0.05)
    htrain_diff.SetLineColor(4)
    htrain_diff_delayedCalib = ROOT.TH1D("diffDelayed", "", 200, -5, 5)
    htrain_diff_referencePoint = ROOT.TH1D("diffReference", "", 200, -5, 5)
    htrain_diff_delayedCalib.SetLineColor(2)
    htrain_diff_referencePoint.SetLineColor(1)
    htrain_diff.SetStats(0)
    htrain_diff.SetLineWidth(2)
    htrain_diff_delayedCalib.SetLineWidth(2)
    htrain_diff_referencePoint.SetLineWidth(2)

    ROOT.TGaxis.SetMaxDigits(3)
    ROOT.gStyle.SetStripDecimals(True)

    for i in range(rangeNodes):
        # Get prediction array for rolling std and variance
        predictions_train = nptable[:, i + 1 + 2 * chLength].astype(float)
        predictions_test = nptable2[:, i + 1 + 2 * chLength].astype(float)
        targets_train = nptable[:, i + 1].astype(float)
        targets_test = nptable2[:, i + 1].astype(float)
        avgShiftTrain, avgShiftTest, avgShiftTrainTargets, avgShiftTestTargets = 0, 0, 0, 0
        for j in range (predictions_train.size-1):
            avgShiftTrain += abs(predictions_train[j+1] - predictions_train[j])
        for j in range (predictions_test.size-1):
            avgShiftTest += abs(predictions_test[j+1] - predictions_test[j])
        for j in range (targets_train.size-1):
            avgShiftTrainTargets += abs(targets_train[j+1] - targets_train[j])
        for j in range (targets_test.size-1):
            avgShiftTestTargets += abs(targets_test[j+1] - targets_test[j])
        rolling_std_train.append(avgShiftTrain/(predictions_train.size-1))
        rolling_std_test.append(avgShiftTest/(predictions_test.size-1))
        rolling_std_train_targets.append(avgShiftTrainTargets/(targets_train.size-1))
        rolling_std_test_targets.append(avgShiftTestTargets/(targets_test.size-1))
        
        #rolling_std_train.append(pandas.Series(predictions_train).rolling(window=rolling_std_window, min_periods=1).std())
        #rolling_std_test.append(pandas.Series(predictions_test).rolling(window=rolling_std_window, min_periods=1).std())
        #rolling_std_train_targets.append(pandas.Series(targets_train).rolling(window=rolling_std_window, min_periods=1).std())
        #rolling_std_test_targets.append(pandas.Series(targets_test).rolling(window=rolling_std_window, min_periods=1).std())

        # Get difference target-prediction in terms of sigma (std) for train and test parts
        #scale = np.mean(nptable2[:, i + 1 + chLength].astype(float))
        scale = nptable[:, i + 1 + chLength].astype(float)
        scale2 = nptable2[:, i + 1 + chLength].astype(float)
        #scale = np.std(nptable2[:, i + 1].astype(float))
        train_diff = (nptable[:, i + 1] - nptable[:, i + 1 + 2 * chLength]) / 3 / scale
        test_diff = (nptable2[:, i + 1] - nptable2[:, i + 1 + 2 * chLength]) / 3 / scale2

        shift = 100
        train_diff_delayedCalib = (nptable[shift:, i+1] - nptable[:-shift, i+1]) / (3 * scale[shift:])  #train_diff[k] corresponds to original j = k + 20
        train_diff_referencePoint = (nptable[shift:, i+1] - (nptable[:-shift, i+1] + nptable[shift:, i+1+2*chLength] - nptable[:-shift, i+1+2*chLength] ) ) / (3 * scale[shift:])  

        for val in train_diff[shift:]:
            htrain_diff.Fill(val)
        for val in train_diff_delayedCalib:
            htrain_diff_delayedCalib.Fill(val)
        for val in train_diff_referencePoint:
            htrain_diff_referencePoint.Fill(val)

        rms_train = np.sqrt(np.mean(train_diff ** 2))
        rms_test = np.sqrt(np.mean(test_diff ** 2))
        rms_train_list.append(rms_train)
        rms_test_list.append(rms_test)
        plt.hist(train_diff, bins, color='#8b2522', alpha=0.7, rwidth=0.95, label = '$\Delta train$')
        plt.hist(train_diff_delayedCalib, bins, color="#228b2b", alpha=0.7, rwidth=0.95, label = '$\Delta delayed$')
        plt.hist(train_diff_referencePoint, bins, color="#22248b", alpha=0.7, rwidth=0.95, label = '$\Delta reference$')
        #plt.hist(test_diff, bins, color='#228B22', alpha=0.7, rwidth=0.95, label = '$\Delta test$')
        #if i == 0:
        #    plt.show()
    avg_rms_train = np.mean(rms_train_list)
    avg_rms_test = np.mean(rms_test_list)
    avg_variance_train = np.mean([std_val.mean() for std_val in rolling_std_train])
    avg_variance_test = np.mean([std_val.mean() for std_val in rolling_std_test])
    avg_variance_train_targets = np.mean([std_val.mean() for std_val in rolling_std_train_targets])
    rolling_std_test_targets = np.mean([std_val.mean() for std_val in rolling_std_test_targets])
    print(f"RMS (train) per node: {rms_train_list}")
    print(f"RMS (test) per node: {rms_test_list}")
    print(f"Average RMS (train): {avg_rms_train:.4f}")
    print(f"Average RMS (test: {avg_rms_test:.4f}")
    print(f"Robustness: {(avg_rms_train/avg_rms_test)*100:.4f}")
    print(f"Average Precision (train) over {rangeNodes} nodes: {avg_variance_train/avg_variance_train_targets*100:.4f}")
    print(f"Average Precision (test) over {rangeNodes} nodes: {avg_variance_test/rolling_std_test_targets*100:.4f}")
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('run number')
    plt.ylabel('dE/dx, a.u.')
    #plt.show()

    c = ROOT.TCanvas("c_corr", "corr", 1000, 1000)
    c.SetGridx()
    htrain_diff.Draw()
    htrain_diff_delayedCalib.Draw("same")
    htrain_diff_referencePoint.Draw("same")
    leg_abs = ROOT.TLegend(0.55, 0.75, 0.9, 0.9)
    leg_abs.AddEntry(htrain_diff, "Predictor", "lpe")
    leg_abs.AddEntry(htrain_diff_delayedCalib, "Delayed offline", "lpe")
    leg_abs.AddEntry(htrain_diff_referencePoint, "From reference", "lpe")
    leg_abs.Draw()
    c.Update()

    stats = ROOT.TPaveText(0.55, 0.55, 0.9, 0.74, "NDC")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetTextAlign(12)

    def add_stats(label, h):
        stats.AddText(f"{label}: mean={h.GetMean():.3f}, RMS={h.GetRMS():.3f}")

    add_stats("Predictor", htrain_diff)
    add_stats("Delayed offline", htrain_diff_delayedCalib)
    add_stats("From reference", htrain_diff_referencePoint)

    stats.Draw()
    c.Update()

    if not ROOT.gROOT.IsBatch():
        try:
            input("Press Enter to exit and close the plot window...")
        except KeyboardInterrupt:
            pass



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
    #print(dftCorrExpS)

    draw_predictions_spread(dftCorrExp)
    draw_pred_and_target_vs_run(dftCorrExp, dftCorrExpS)
    draw_predictions_minus_target(dftCorrExp, dftCorrExpS)
    #draw_pred_vs_target(dftCorrExp)


def predict_nn(fName, oName, path):
    config = Config()

    mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/realTimeCalibrations/backend/serverData/"
    dataManager = DataManager(mainPath)

    predictionList = makePredicionList(config, dataManager, fName, oName, path)
    #print(predictionList)

    #write_output(predictionList,mod,enlist)

def predict_cicle(testNum):
    mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/realTimeCalibrations/backend/serverData/"
    path = mainPath+"function_prediction/"
    predict_nn("tesu",'predicted1_'+str(testNum), path)
    predict_nn("simu",'predicted_'+str(testNum), path)


if __name__ == "__main__":
    mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/realTimeCalibrations/backend/serverData/"
    path = mainPath+"function_prediction/"

    #print("start python predict")
    test = 0
    #predict_nn("tesu",'predicted1_'+str(test), path)
    #predict_nn("simu",'predicted_'+str(test), path)

    #analyseOutput(path+"predicted/predicted_"+str(test),path+"simu", path+"predicted/predicted1_"+str(test),path+"tesu")

    #analyseOutput(path+"predictedSim.parquet",path+"simu.parquet")


    predict_nn("tesuCosmic",'predictedCosmics1_', path)
    predict_nn("simuCosmic",'predictedCosmics_', path)

    analyseOutput(path+"predicted/predictedCosmics1_",path+"tesuCosmic", path+"predicted/predictedCosmics_",path+"simuCosmic")

    #analyseOutput(path+"predictedCosmics.parquet",path+"simu.parquet")


    #plot_tot_vs_hv_and_heatmap(path=path)
