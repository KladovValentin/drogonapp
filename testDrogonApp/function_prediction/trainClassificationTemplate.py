
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import uproot
import pandas
import random
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from tqdm import trange
from pympler import asizeof
from models.model import DNN
from models.model import LSTM
from models.model import Conv2dLSTM
from models.model import Conv2dLSTMCell
from dataHandling import My_dataset, DataManager, load_dataset


"""
If new model is being added:
1) edit dataHandling.loadDataset() to include how dataset is reshaped for the input
2) add model to model.py and include it in train.py and predict.py
3) add option to train_NN() function in train.py
4) add option to load_model() function in predict.py
5) change config.py
"""



class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.value = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (validation_loss - self.min_validation_loss >= self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        self.value = validation_loss
        return False


def train_DN_model(model, train_loader, loss, optimizer, num_epochs, valid_loader, scheduler=None):
    print("start model nn train")
    loss_history = []
    train_history = []
    validLoss_history = []
    early_stopper = EarlyStopper()

    for epoch in range(num_epochs):
        model.train()

        loss_train = 0
        accuracy_train = 0
        isteps = 0
        #tepoch = tqdm(train_loader,unit=" batch")
        for i_step, (x, y) in enumerate(train_loader):
            #tepoch.set_description(f"Epoch {epoch}")
            prediction = model(x)
            #print(*prediction[0][14])
            #print(*y[0][14])

            #print(prediction.shape)
            #print(y.shape)

            running_loss = loss(prediction, y)
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            
            fullYAmount = y.shape[0]
            if (y.ndim>1):
                for i in range(y.ndim-1):
                    fullYAmount *= y.shape[i+1] 
            running_acc = torch.sum( ((prediction-y)>-0.3) & ((prediction-y)<0.3) )/ fullYAmount
            accuracy_train += running_acc
            loss_train += float(running_loss)
            isteps += 1

            loss_history.append(float(running_loss))

            #tepoch.set_postfix(loss=float(running_loss), accuracy=float(running_acc)*100)

        accuracy_train = accuracy_train/isteps
        loss_train = loss_train/isteps

        #<<<< Validation >>>>#
        model.eval()
        loss_valid = 0
        accuracy_valid = 0
        validLosses = []
        validAccuracies = []
        with torch.no_grad():
            for v_step, (x, y) in enumerate(valid_loader):
                prediction = model(x)
                validLosses.append(float(loss(prediction, y)))

                fullYAmount = y.shape[0]
                if (y.ndim>1):
                    for i in range(y.ndim-1):
                        fullYAmount *= y.shape[i+1] 
                validAccuracies.append(torch.sum( ((prediction-y)>-0.3) & ((prediction-y)<0.3) )/fullYAmount)

            loss_valid = np.mean(np.array(validLosses))
            accuracy_valid = np.mean(np.array(validAccuracies))
        model.train() 


        if scheduler is not None:
            #scheduler.step(loss_valid)
            scheduler.step()


        #<<<< Printing and drawing >>>>#
        #loss_history.append(loss_train)
        train_history.append(accuracy_train)
        validLoss_history.append(float(loss_valid))
        ep = np.arange(1,(epoch+1)*(i_step+1)+1,1)
        lv = np.array(validLoss_history)
        lt = np.array(loss_history)
        plt.clf()
        plt.plot(ep,lt,"blue",label="train")
        #plt.plot(ep,lv,"orange",label="validation")
        plt.legend(loc=[0.5,0.6])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #if ((epoch+1)%100 == 0):
        #    plt.show()

        print("Average loss: %f, valid loss: %f, Train accuracy: %f, V acc: %f, epoch: %f" % (loss_train, loss_valid, accuracy_train*100, accuracy_valid*100, epoch+1))

        #if early_stopper.early_stop(loss_valid):             
        #    break
    
    return accuracy_valid

def train_NN(mainPath, simulation_path="simu1.parquet"):
    from config import Config
    config = Config()
    print("start nn training")
    
    f = open(mainPath+"function_prediction/trainresults1.txt", "w")

    for i in range(1):
        batch_size, lr, nNeurons, nLayers, weight_decay = varyHyperparameters()
        print(str(batch_size) + " " + str(lr) + " " + str(nNeurons) + " " + str(nLayers) + " " + str(weight_decay))
        
        batch_size = 512 #512
        weight_decay = 0.03
        lr = 0.005
        nLayers = 3
        nNeurons = 200
        fullset = pandas.read_parquet(mainPath+"function_prediction/" + simulation_path)
        fullset.drop(list(fullset.columns)[0],axis=1,inplace=True)
        dftCorr = fullset.sample(frac=1.0).reset_index(drop=True) # shuffling
        dataTable = dftCorr.sample(frac=0.8).sort_index()
        validTable = dftCorr.drop(dataTable.index)

        
        train_dataset = My_dataset(load_dataset(config, dataTable))
        valid_dataset = My_dataset(load_dataset(config, validTable))

        dropLastT = False
        if (dataTable.shape[0]%batch_size==1):
            dropLastT = True
        dropLastV = False
        if (validTable.shape[0]%batch_size==1):
            dropLastV = True
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=dropLastT)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=dropLastV)

        input_dim = train_dataset[0][0].shape

        del validTable, valid_dataset, dftCorr

        if (config.modelType == "DNN"):
            nn_model = DNN(input_dim=input_dim[-1], output_dim=1, nLayers=nLayers, nNeurons=nNeurons).type(torch.FloatTensor)
        elif (config.modelType == "LSTM"):
            nn_model = LSTM(input_dim=input_dim[-1], embedding_dim=64, hidden_dim=64, output_dim=1, num_layers=1, sentence_length=input_dim[0]).type(torch.FloatTensor)
        elif (config.modelType == "ConvLSTM"):
            nn_model = Conv2dLSTM(input_size=(input_dim[-3],input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=(3,3), num_layers=1, bias=0, output_size=1) #16 16

        loss = nn.MSELoss()

        #optimizer = optim.SGD(nn_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.05)
        optimizer = optim.Adam(nn_model.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.2, factor=0.2)

        print("prepared to train nn")
        loc_acc = train_DN_model(nn_model, train_loader, loss, optimizer, 200, valid_loader, scheduler = scheduler)

        print("trained nn, valid acc = " + str(loc_acc))

        f.write(str(batch_size) + " " + str(lr) + " " + str(nNeurons) + " " + str(nLayers) + " " + str(weight_decay) + " " + str(100*loc_acc) + "\n")

        nn_model.eval()
        torch.save(nn_model.state_dict(), mainPath+"function_prediction/tempModel.pt")

        if (config.modelType == "DNN"):
            modelRandInput = torch.randn(1, input_dim[-1])
        elif (config.modelType == "LSTM"):
            modelRandInput = torch.randn(1, input_dim[0], input_dim[-1])
        elif (config.modelType == "ConvLSTM"):
            modelRandInput = torch.randn(1, input_dim[0], input_dim[-3], input_dim[-2], input_dim[-1])
        torch.onnx.export(nn_model,                                # model being run
                  modelRandInput,    # model input (or a tuple for multiple inputs)
                  mainPath+"function_prediction/tempModel.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ["input"],              # the model's input names
                  output_names = ["output"])            # the model's output names
    
    f.close()
    #nn_model.eval()
    #torch.save(nn_model.state_dict(), "tempModel.pt")
    #model_scripted = torch.jit.trace(nn_model,torch.tensor(np.array([load_dataset(dataTable)[0][0],load_dataset(dataTable)[0][1]])))
    #model_scripted.save('modelScr.pt')


def varyHyperparameters():

    batch_size = int(1024*2**(random.uniform(-4,2)))
    learn_rate = 0.001*2**(random.uniform(-10,3))
    nNeurons = int(256*2**(random.uniform(-4,2)))
    weight_decay = 0.1*2**(random.uniform(-2,2))
    nLayers = random.randint(2,4)

    return batch_size, learn_rate, nNeurons, nLayers, weight_decay

print("start_train_python")

mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/"

dataManager = DataManager()
dataManager.manageDataset("train_nn",mainPath)


train_NN(mainPath)

