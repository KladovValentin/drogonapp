import torch.nn as nn
import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, nLayers, nNeurons):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


        self.nn_model = nn.Sequential(
            nn.Linear(input_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(1024, 1024),
            #nn.BatchNorm1d(1024),
            #nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, output_dim)
        )


        self.layers = nn.ModuleList()
        self.input_size=input_dim
        self.size = nNeurons
        for i in range(nLayers):
            self.layers.append(nn.Linear(self.input_size, self.size))
            self.layers.append(nn.BatchNorm1d(self.size))
            self.layers.append(nn.LeakyReLU(inplace=True))
            self.input_size = self.size
            self.size = self.input_size*2
            if i >= (nLayers-1)/2:
                self.size = self.input_size//2
        self.layers.append(nn.Linear(self.input_size, self.output_dim))


    def forward(self, text):
        text = self.nn_model(text)
        #for layer in self.layers:
        #    text = layer(text)
        output = torch.reshape(text,(-1,))
        return output


class LSTM(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers, sentence_length):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.sentence_length = sentence_length
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.nn_model = nn.Sequential(
            nn.Linear(input_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, text):
        # text dim: [batch size, sentence length, input_dim]
        batch_size = text.size(0)

        hidden, cellst = self.init_hidden(batch_size)

        embedded = self.nn_model(text.reshape((batch_size*self.sentence_length, self.input_dim))).reshape((batch_size, self.sentence_length, self.embedding_dim))
        # embedded dim: [batch size, sentence length, embedding dim]
        output, (hidden,cellst) = self.rnn(embedded, (hidden,cellst))
        #output, (hidden,cellst) = self.rnn(text, (hidden,cellst))

        #embedded = self.nn_model(output.reshape((batch_size*self.sentence_length, self.hidden_dim))).reshape((batch_size, self.sentence_length, self.embedding_dim))
        
        # output dim: [batch size, sentence length, hidden dim]
        # hidden dim: [nLayers, batch size, hidden dim]

        #output = torch.reshape(output,(batch_size, self.sentence_length))
        output = self.fc(torch.reshape(output,(batch_size * self.sentence_length, self.hidden_dim))).reshape((batch_size , self.sentence_length))
        #output1 = self.fc(hidden[self.num_layers-1])
        #output1 = torch.reshape(output1,(-1,))
        return output

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        cellst = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden, cellst#.to(device)





class Conv2dLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(Conv2dLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.bias = bias
        self.x2h = nn.Conv2d(in_channels=input_size[0],
                             out_channels=hidden_size * 4,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)

        self.h2h = nn.Conv2d(in_channels=hidden_size,
                             out_channels=hidden_size * 4,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)
        
        self.Wc = nn.Parameter(torch.zeros((1, self.hidden_size * 3, input_size[1], input_size[2])))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size, height_size, width_size)
        #       hx: of shape (batch_size, hidden_size, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size, height_size, width_size)
        #       cy: of shape (batch_size, hidden_size, height_size, width_size)

        #if self.Wc == 1:
        #    self.Wc = nn.Parameter(torch.zeros((1, self.hidden_size * 3, input.size(2), input.size(3))))

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size, self.input_size[1], self.input_size[2]))
            hx = (hx, hx)
        hx, cx = hx

        gates = self.x2h(input) + self.h2h(hx)

        #print(input.shape, hx.shape, self.x2h(input).shape)

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        Wci, Wcf, Wco = self.Wc.chunk(3, 1)

        #print(Wci.shape, cx.shape, input_gate.shape)
        i_t = torch.sigmoid(input_gate + Wci * cx)
        f_t = torch.sigmoid(forget_gate + Wcf * cx)
        g_t = torch.tanh(cell_gate)

        cy = f_t * cx + i_t * torch.tanh(g_t)
        o_t = torch.sigmoid(output_gate + Wco * cy)

        hy = o_t * torch.tanh(cy)


        return (hy, cy)


class Conv2dLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, kernel_size, num_layers, bias, output_size):
        super(Conv2dLSTM, self).__init__()

        self.input_size = input_size[0]
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.height = input_size[1]
        self.width = input_size[2]

        if type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList([Conv2dLSTMCell((self.embedding_size,self.height,self.width),
                                            self.hidden_size,
                                            self.kernel_size,
                                            self.bias) for _ in range(self.num_layers)])

        #self.rnn_cell_list.append()
        #for l in range(1, self.num_layers):
        #    self.rnn_cell_list.append(Conv2dLSTMCell(self.hidden_size,
        #                                        self.hidden_size,
        #                                        self.kernel_size,
        #                                        self.bias))

        #self.conv = nn.Conv2d(in_channels=self.hidden_size,
        #                     out_channels=self.output_size,
        #                     kernel_size=self.kernel_size,
        #                     padding=self.padding,
        #                     bias=self.bias)
        #self.conv1 = nn.Conv2d(in_channels=self.hidden_size, out_channels=1, kernel_size=(6, 1))
        #self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels=self.hidden_size, out_channels=1, kernel_size=(1, 1))
        #self.conv1 = nn.Conv2d(in_channels=self.hidden_size, out_channels=1, kernel_size=(1, 1))
        #self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6, 1))
        #self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))

        #self.nn_model = nn.ModuleList([nn.Sequential(
        #    nn.Linear(self.input_size, 256, bias=True),
        #    nn.BatchNorm1d(256),
        #    nn.LeakyReLU(inplace=True),
        #    nn.Linear(256, 256, bias=True),
        #    nn.BatchNorm1d(256),
        #    nn.LeakyReLU(inplace=True),
        #    nn.Linear(256, embedding_size),
        #    #nn.BatchNorm1d(embedding_dim)
        #) for _ in range(self.height)])

        #self.nn_model = nn.ModuleList([nn.Sequential(
        #    nn.Linear(self.input_size, 256, bias=True),
        #    nn.BatchNorm1d(256),
        #    nn.LeakyReLU(inplace=True),
        #    nn.Linear(256, 16),
        #    #nn.BatchNorm1d(embedding_dim)
        #) for _ in range(self.height)])

        self.nn_model = nn.Sequential(
            nn.Linear(self.height * self.width * self.input_size, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.height * self.width * self.embedding_size),
            #nn.BatchNorm1d(embedding_dim)
        )



    def forward(self, input, hx=None):

        batch_size = input.size(0)
        sentence_length = input.size(1)

        inputDeep = input.movedim(-3,-1)

        #embedded1 = [inputDeep[:,:,i,:,:] for i in range(self.height)]
        #embedded = torch.Tensor(batch_size, sentence_length, 1, self.width, 16)
        #for i in range(self.height):
        #    embedded1[i] = (self.nn_model[i](embedded1[i].reshape((batch_size*sentence_length*self.width, self.input_size))))
        #    embedded1[i] = embedded1[i].reshape((batch_size, sentence_length, 1, self.width, 16))
        #    if (i == 0):
        #        embedded = embedded1[i]
        #    else:
        #        embedded = torch.cat((embedded, embedded1[i]), dim=2)

        #embedded = embedded.reshape((batch_size*sentence_length*self.height*self.width, 16))
        #embedded = (self.nn_model2(embedded)).reshape((batch_size, sentence_length, self.height, self.width, self.embedding_size))
        
        inputDeep1 = inputDeep.reshape((batch_size*sentence_length, self.height*self.width*self.input_size))
        embedded = self.nn_model(inputDeep1)
        embedded = embedded.reshape((batch_size, sentence_length, self.height, self.width, self.embedding_size))

        #inputDeep1 = inputDeep.reshape((batch_size*sentence_length*self.height*self.width, self.input_size))
        #print(inputDeep.shape, inputDeep1.shape)
        #embedded = self.nn_model(inputDeep1)
        #embedded = embedded.reshape((batch_size, sentence_length, self.height, self.width, self.embedding_size))
        embedded = embedded.movedim(-1,-3)
        #print(embedded.shape, input.shape)


        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, sentence_length, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, self.height, self.width).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, self.height, self.width))
        else:
             h0 = hx

        #outs = torch.Tensor(batch_size, input.size(1), self.output_size, self.height, self.width)

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer], h0[layer]))

        
        result_tensor = torch.Tensor(batch_size, 1, self.output_size, self.height, self.width)
        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        embedded[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            #print(hidden_l[0].shape)
            newTensor = self.conv1(hidden_l[0])
            #print (newTensor.shape)
            #newTensor = self.pool1(newTensor)
            #print (newTensor.shape)
            newTensorI = newTensor.squeeze().unsqueeze(1).unsqueeze(2)
            if (t == 0):
                result_tensor = newTensorI
            else:
                result_tensor = torch.cat((result_tensor, newTensorI), dim=1)
            #print(hidden_l[0].shape,newTensor.shape, result_tensor.shape)

        #torch.stack(outs)
        #torch.cat(outs,)


        #out = outs.squeeze()

        #out = self.conv(out)

        #print("a", result_tensor.shape)

        return result_tensor

