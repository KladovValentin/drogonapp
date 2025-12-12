
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
from models.model import Conv2dLSTM
from models.model import GCNLSTM
from models.model import GraphTransformer2D
from dataHandling import My_dataset, Graph_dataset, DataManager, load_dataset
#from predict import predict_cicle


"""
If new model is being added:
1) edit dataHandling.loadDataset() to include how dataset is reshaped for the input
2) add model to model.py and include it in train.py and predict.py
3) add option to train_NN() function in train.py
4) add option to load_model() function in predict.py
5) change config.py
"""

mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/drogonapp/testDrogonApp/serverData/"
dataManager = DataManager(mainPath)


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predictions, targets):
        squared_errors = torch.pow(predictions - targets[:,0,:], 2)
        clipped_sigmas = torch.where(targets[:,1,:] < 0.1, torch.ones_like(targets[:,1,:]), targets[:,1,:])
        normalized_errors = squared_errors / torch.pow(clipped_sigmas, 2)
        #normalized_errors = squared_errors / targets[:,:,1,:]
        #print(normalized_errors)
        loss = torch.mean(normalized_errors)
        #print(loss)
        return loss

class EarlyStopper:
    def __init__(self, minmax, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.minmax = minmax
        self.counter = 0
        self.value = 0
        self.min_validation_loss = np.inf
        if minmax == 0:
            self.min_validation_loss = -np.inf

    def early_stop(self, validation_loss):
        if (((self.minmax == 1) & (validation_loss < self.min_validation_loss)) | ((self.minmax == 0) & (validation_loss > self.min_validation_loss))):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (abs(validation_loss - self.min_validation_loss) >= self.min_delta):
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
    early_stopper = EarlyStopper(1)
    #xt0 = torch.tensor()
    #xv0 = torch.tensor()

    for epoch in range(num_epochs):
        model.train()

        loss_train = 0
        accuracy_train = 0
        isteps = 0
        fulfulYAmount = 0
        #train_iter = iter(train_loader)
        #tepoch = tqdm(train_loader,unit=" batch")
        for i_step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            #print(x)
            #print(y)
        #i_step_count = 0
        #for i_step in tepoch:
        #    i_step_count+=1
        #    tepoch.set_description(f"Epoch {epoch}")
        #    (x,y) = next(train_iter)
            prediction = model(x)
            #xt0 = prediction.clone()
            #print(*prediction[0][14])
            #print(*y[0][14])

            #print(prediction.shape)
            #print(y.shape)

            #if (i_step == 10):
            #    print("testing")
            #    print(x[0,-1,:,:],y[0,-1,0,:])
            #    print(prediction[0,-1,:],y[0,-1,0,:])

            #running_loss = loss(prediction[:,14,:], y[:,14,:])
            #print(y[0])
            #running_loss = loss(prediction0[:,:,:], y[:,:,0,:])
            #running_loss = loss(prediction0[:,-1,:], y[:,-1,0,:])

            if getattr(loss, "loss_type", None) == "nnMSE":
                running_loss = loss(prediction[:,-1,:], y[:,-1,0,:])
            elif getattr(loss, "loss_type", None) == "customMSE":
                running_loss = loss(prediction[:,-1,:], y[:,-1,:,:])
            running_loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            
            #prediction = model(x)
            yAccTest = y[:,-1,0,:]
            #print((prediction[0,-1,0],yAccTest[0,0]))
            #yAccTest = y[:,0,:]
            fullYAmount = yAccTest.shape[0]
            if (yAccTest.ndim>1):
                for i in range(yAccTest.ndim-1):
                    fullYAmount *= yAccTest.shape[i+1] 
            fulfulYAmount+=fullYAmount
            std = yAccTest.std(dim=0, unbiased=False).mean()
            #std = y[:,-1,1,:]*2
            running_acc = torch.sum( ((prediction[:,-1,:]-yAccTest)>-std) & ((prediction[:,-1,:]-yAccTest)<std) )
            #running_acc = torch.sum( ((prediction-yAccTest)>-0.3) & ((prediction-yAccTest)<0.3) )
            #print(running_acc)
            accuracy_train += running_acc
            loss_train += float(running_loss)
            isteps += 1

            loss_history.append(float(running_loss))

        #    tepoch.set_postfix(loss=float(running_loss), accuracy=float(running_acc)*100)

        accuracy_train = accuracy_train/fulfulYAmount
        loss_train = loss_train/isteps

        #<<<< Validation >>>>#
        model.eval()
        loss_valid = 0
        accuracy_valid = 0
        fulfulYAmount = 0
        validLosses = []
        validAccuracies = []
        with torch.no_grad():
            for v_step, (xval, yval)  in enumerate(valid_loader):
                #print(x)
                #print(y)
                prediction1 = model(xval)
                xv0 = prediction1.clone()
                #validLosses.append(float(loss(prediction1, yval[:,:,0,:])))
                #validLosses.append(float(loss(prediction1[:,-1,:], yval[:,-1,0,:])))
                if getattr(loss, "loss_type", None) == "nnMSE":
                    validLosses.append(float(loss(prediction1[:,-1,:], yval[:,-1,0,:])))
                if getattr(loss, "loss_type", None) == "customMSE":
                    validLosses.append(float(loss(prediction1[:,-1,:], yval[:,-1,:,:])))

                yAccTest = yval[:,-1,0,:]
                #yAccTest = yval[:,0,:]
                fullYAmount = yAccTest.shape[0]
                if (yAccTest.ndim>1):
                    for i in range(yAccTest.ndim-1):
                        fullYAmount *= yAccTest.shape[i+1] 
                fulfulYAmount+=fullYAmount
                std = yAccTest.std(dim=0, unbiased=False).mean()
                #std = yval[:,-1,1,:]*2
                validAccuracies.append(torch.sum( ((prediction1[:,-1,:]-yAccTest)>-std) & ((prediction1[:,-1,:]-yAccTest)<std) ))
                #print(validAccuracies[-1])

            loss_valid = np.mean(np.array(validLosses))
            accuracy_valid = np.sum(np.array(validAccuracies))/fulfulYAmount


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

        #if (not torch.equal(xt0, xv0)):
        #    print(xt0,xv0)
        print("Average loss: %f, valid loss: %f, Train accuracy: %f, V acc: %f, epoch: %f" % (loss_train, loss_valid, accuracy_train*100, accuracy_valid*100, epoch+1))

        #if (epoch > 50):
        #    if (early_stopper.early_stop(loss_valid*100)):
        #        break
    
    return accuracy_valid

def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def train_NN(ind,transfer,mainPath, simulation_path="simu.parquet"):
    from config import Config
    config = Config()
    print("start nn training")

    transferTraining = transfer

    datasetMod = "train_nn"
    if (transferTraining):
        datasetMod = "test_nn"   #keep mean and std from before
    #dataManager.manageDataset(datasetMod,ind)
    
    f = open(mainPath+"function_prediction/trainresults1.txt", "w")

    for i in range(1):
        batch_size, lr, nNeurons, nLayers, weight_decay = varyHyperparameters()
        print(str(batch_size) + " " + str(lr) + " " + str(nNeurons) + " " + str(nLayers) + " " + str(weight_decay))
        
        batch_size = 32 #50000 #512
        #batch_size = 32 #50000 #512
        weight_decay = 0.0000
        lr = 0.0005
        epochs = 100
        if (transferTraining):
            #lr = 0.02
            lr = 0.1
            if (ind == 0):
                epochs = 100  #cosmic
            #epochs = 40  #new beam time
            if (ind == 1):
                epochs = 60  #going back from cosmic
                lr = lr*0.1
            #weight_decay = 0.0005
            #weight_decay = 0.0001
            weight_decay = 0.0
        nLayers = 1
        nNeurons = 200
        fullset = pandas.read_parquet(mainPath+"function_prediction/" + simulation_path)
        #fullset = dataManager.normalizeDatasetNormal(fullset)
        fullset = dataManager.normalizeDatasetScale(fullset)
        #fullset.drop(list(fullset.columns)[0],axis=1,inplace=True) #drop indices
        print(fullset)
        #dftCorr = fullset.sample(frac=1.0).reset_index(drop=True) # shuffling
        dftCorr = fullset.reset_index(drop=True)
        #print(dftCorr)
        #if transferTraining:
        #dataTable = dftCorr#.sort_index()
        #validTable = dataTable.copy()
        #else:
        
        if (transferTraining):
            dataTable = dftCorr.iloc[:int(dftCorr.shape[0]*1.0)].copy()
        else:
            dataTable = dftCorr.iloc[:int(dftCorr.shape[0]*0.8)].copy()
        validTable = dftCorr.drop(dataTable.index)

        
        if (config.modelType == "gConvLSTM"):
            xt, yt, e_i, e_a = load_dataset(config,dataTable)
            #if (ind == -1):
            #    xt[:,:,1,:] = 1.750 + np.random.normal(0,0.05)  # set HV for constant to not affect the experimental training
            xv, yv, _, _ = load_dataset(config,validTable)
            shuffled_indicest = np.random.permutation(xt.shape[0])
            shuffled_indicest1 = np.random.permutation(xv.shape[0])
            #print(shuffled_indicest)
            xt = xt[shuffled_indicest]
            yt = yt[shuffled_indicest]
            xv = xv[shuffled_indicest1]
            yv = yv[shuffled_indicest1]
            train_dataset = My_dataset((xt,yt))
            valid_dataset = My_dataset((xv,yv))
        else:
            xt, yt = load_dataset(config, dataTable)
            xv, yv = load_dataset(config,validTable)
            shuffled_indicest = np.random.permutation(xt.shape[0])
            #xt = xt[shuffled_indicest]
            #yt = yt[shuffled_indicest]
            train_dataset = My_dataset((xt,yt))
            valid_dataset = My_dataset((xv,yv))

        dropLastT = False
        if (dataTable.shape[0]%batch_size==1):
            dropLastT = True
        dropLastV = False
        if (validTable.shape[0]%batch_size==1):
            dropLastV = True
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=dropLastT)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=dropLastV)

        input_dim = train_dataset[0][0].shape

        del validTable, valid_dataset

        if (config.modelType == "ConvLSTM"):
            nn_model = Conv2dLSTM(input_size=(input_dim[-3],input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=(5,5), num_layers=1, bias=0, output_size=1) #16 16
        elif (config.modelType == "gConvLSTM"):
            #nn_model = GCNLSTM(input_size=(input_dim[-2],input_dim[-1]), embedding_size=8, hidden_size=16, kernel_size=3, num_layers=1, e_i=(torch.LongTensor(e_i).movedim(-2,-1)), e_a=torch.Tensor(e_a))
            #nn_model = GCNLSTM(input_dims=(input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=3, num_layers=1, e_i=(torch.LongTensor(e_i).movedim(-2,-1)), e_a=torch.Tensor(e_a))
            nn_model = GraphTransformer2D(input_size = input_dim[-2], embedding_size=8, num_nodes = input_dim[-1], sentence_length = input_dim[-3], e_i=(torch.LongTensor(e_i).movedim(-2,-1)), e_a=torch.Tensor(e_a))
            

            n_nodes = input_dim[-1]
            in_channels = input_dim[-2]
            sent_length = input_dim[-3]
            #exported_program = torch.export.export(GCNLSTM(input_size=(input_dim[-2],input_dim[-1]), embedding_size=16, hidden_size=16, kernel_size=2, num_layers=1), (torch.randn(2, sent_length, in_channels, n_nodes),))
            #torch.export.save(exported_program, 'exported_gConvLSTM.pt2')

        if (transferTraining):
            nn_model.load_state_dict(torch.load(mainPath+"function_prediction/tempModelT.pt"))
            #nn_model.hv_mlp.apply(reset_weights)
            nn_model.train()
            # Freeze the lower layers
            for name, param in nn_model.named_parameters():
                trainable = False
                #if "gcn_cell_list" in name:  # Adjust the condition based on your model's naming convention
                #if "nn_model." in name or "nn_model2." in name:  # Adjust the condition based on your model's naming convention
                #if "nn_model." in name or "gcn_cell" in name or "scale_shift" in name:  # Adjust the condition based on your model's naming convention
                #if "input_normalizer" in name or "scale_shift" in name or "nn_model." or "linear" in name:           # cosmic things
                if (ind == 0):
                    #if "input_normalizer" in name or "scale_shift" in name or "nn_model." in name or "hv_mlp" in name or "linear" in name:           # cosmic things
                    if "input_normalizer" in name or "scale_shift" in name or "nn_model." in name or "hv_mlp" in name:           # cosmic things
                    #if "input_normalizer" in name or "scale_shift" in name or "nn_model2.0" in name or "nn_model2.1" in name or "nn_model2.3" in name or "film" in name:           # cosmic things
                    #if "input_normalizer" in name or "scale_shift" in name or "gcn_cell" in name or "film" in name:           # cosmic things
                    #if "input_normalizer" in name or "scale_shift" in name or "linear" in name or "film" in name:           # cosmic things
                    #if "input_normalizer" in name or "scale_shift" in name or "film" in name:           # cosmic things
                    #if "input_normalizer" in name or "scale_shift" in name or "nn_model." in name or "film" in name:           # cosmic things
                    #if "input_normalizer" in name or "scale_shift"  in name or "gcn_cell_list" in name:           # cosmic things
                    #if "input_normalizer" in name or "scale_shift" in name or "hv_mlp" in name:           # cosmic things
                    #if "input_normalizer" in name or "gcn_cell" in name or "scale_shift" in name:  # Adjust the condition based on your model's naming convention
                        trainable = True
                #if "input_normalizer" in name or "scale_shift" in name or "nn_model." or "gcn_cell_list.0" in name:           # cosmic things
                
                elif (ind == 1):
                    #if "input_normalizer" in name or "scale_shift" in name or "linear" in name:                               #exp hv inference from cosmic
                    if "input_normalizer" in name or "scale_shift" in name:
                        trainable = True
                #if "input_normalizer" in name or "scale_shift" in name or "nn_model2." in name:
                if (trainable):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        #for name, param in nn_model.named_parameters():
        #    print(name, param.requires_grad)

        lossType = "customMSE"
        #lossType = "nnMSE"

        if (lossType == "nnMSE"):
            loss = nn.MSELoss()
            loss.loss_type = "nnMSE"
        elif (lossType == "customMSE"):
            print("using custom MSE loss")
            loss = CustomMSELoss()
            loss.loss_type = "customMSE"

        #optimizer = optim.SGD(nn_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.05)
        optimizer = optim.Adam(nn_model.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.2, factor=0.2)

        print("prepared to train nn")
        loc_acc = train_DN_model(nn_model, train_loader, loss, optimizer, epochs, valid_loader, scheduler = scheduler)

        print("trained nn, valid acc = " + str(loc_acc))

        f.write(str(batch_size) + " " + str(lr) + " " + str(nNeurons) + " " + str(nLayers) + " " + str(weight_decay) + " " + str(100*loc_acc) + "\n")

        nn_model.eval()
        torch.save(nn_model.state_dict(), mainPath+"function_prediction/tempModelT.pt")
        #nn_model = torch.jit.script(nn_model)

        if (config.modelType == "ConvLSTM"):
            modelRandInput = torch.randn(1, input_dim[0], input_dim[-3], input_dim[-2], input_dim[-1])
        elif (config.modelType == "gConvLSTM"):
            n_nodes = input_dim[-1]
            in_channels = input_dim[-2]
            sent_length = input_dim[-3]
            #e_i_r = torch.randint(0,n_nodes-1,(1,edge_length,2))
            #e_a_r = torch.ones(1,edge_length)
            modelRandInput = (torch.randn(1, sent_length, in_channels, n_nodes))
        torch.onnx.export(nn_model,                                # model being run
                  modelRandInput,    # model input (or a tuple for multiple inputs)
                  mainPath+"function_prediction/tempModel.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ["input"],              # the model's input names
                  output_names = ["output"], opset_version=11)            # the model's output names
    
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

dataManager.manageDataset("train_nn",0)
#dataManager.manageDataset("test_nn",0)

#dataManager.manageDataset("test_nn",0)
train_NN(0,False,mainPath)
#train_NN(0,True,mainPath)


#dataManager.manageDatasetCosmics("train_nn",0)
#dataManager.manageDatasetCosmics("test_nn",0)

#train_NN(0,True,mainPath,"simuCosmic.parquet")
#train_NN(0,False,mainPath,"simuCosmic.parquet")

#dataManager.manageDataset("test_nn",0)
#predict_cicle(0)


#for i in range(20):
#    train_NN(i+1,True,mainPath)
#    predict_cicle(i+1)

# back to beam time with fixed hv_mlp and nn_model
#dataManager.manageDataset("train_nn",0)
#train_NN(1,True,mainPath)