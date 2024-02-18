import torch.nn as nn
import torch
#import torch_geometric
from torch_geometric.nn import ChebConv, GCNConv, GCN
from torch_geometric.nn.inits import glorot, zeros
#import torch.nn.functional as F
#from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.jit import ignore

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


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        """
        x: input features (N x in_features), N is the number of nodes
        adj: adjacency matrix (N x N), sparse or dense
        """
        support = torch.matmul(x, self.weight)  # XW
        output = torch.matmul(adj, support) + self.bias  # AXW + b
        return output
    
class GraphConvolutionNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super(GraphConvolutionNLayer, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.sizes = [self.in_channels]
        for i in range(self.num_layers-1):
            self.sizes.append(self.hidden_channels)
        if (self.num_layers > 1):
            self.sizes.append(self.out_channels)

        self.gcn_cell_list = nn.ModuleList([GraphConvolution(self.sizes[i],self.sizes[i+1]) for i in range(self.num_layers)])

    def forward(self, x, edge_index, edge_weight):
        """
        x: input features (N x in_features), N is the number of nodes
        adj: adjacency matrix (N x N), sparse or dense
        """
        adjacency_matrix = Variable(torch.zeros(x.size(1), x.size(1)))
        #print(edge_index)
        #print(edge_index[0])
        adjacency_matrix[edge_index[0], edge_index[1]] = 1
        #hidden = list()
        #hidden.append(x)
        for layer in range(self.num_layers):
            x = self.gcn_cell_list[layer](x,adjacency_matrix)
            #hidden.append()
        return x


class MyGCNConv(GCNConv):
    propagate_type = {'x': torch.Tensor, 'edge_index': torch.Tensor, 'edge_weight': torch.Tensor }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)


class GConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True, normalization = "sym"):
        super(GConvLSTMCell, self).__init__()

        self.in_channels = input_size[0]
        self.n_nodes = input_size[1]
        self.hidden_channels = hidden_size
        self.K = kernel_size
        self.normalization = normalization

        self.bias = bias

        #self.Wc = nn.Parameter(torch.zeros((1, self.hidden_size * 3, input_size[1], input_size[2])))
        #self.reset_parameters()
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv(in_channels=self.in_channels, out_channels=self.hidden_channels, K=self.K, normalization=self.normalization, bias=self.bias )
        #self.conv_x_i = GCNConv(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)
        self.conv_h_i = ChebConv(in_channels=self.hidden_channels, out_channels=self.hidden_channels, K=self.K, normalization=self.normalization, bias=self.bias )
        #self.conv_h_i = GCNConv(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)

        #self.conv_x_i = GraphConvolutionNLayer(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)
        #self.conv_h_i = GraphConvolutionNLayer(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)

        self.w_c_i = nn.Parameter(torch.Tensor(1, self.hidden_channels))
        self.b_i = nn.Parameter(torch.Tensor(1, self.hidden_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv(in_channels=self.in_channels, out_channels=self.hidden_channels, K=self.K, normalization=self.normalization, bias=self.bias )
        #self.conv_x_f = GCNConv(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)
        self.conv_h_f = ChebConv(in_channels=self.hidden_channels, out_channels=self.hidden_channels, K=self.K, normalization=self.normalization, bias=self.bias )
        #self.conv_h_f = GCNConv(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)

        #self.conv_x_f = GraphConvolutionNLayer(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)
        #self.conv_h_f = GraphConvolutionNLayer(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)

        self.w_c_f = nn.Parameter(torch.Tensor(1, self.hidden_channels))
        self.b_f = nn.Parameter(torch.Tensor(1, self.hidden_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv(in_channels=self.in_channels, out_channels=self.hidden_channels, K=self.K, normalization=self.normalization, bias=self.bias )
        #self.conv_x_c = GCNConv(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)
        self.conv_h_c = ChebConv(in_channels=self.hidden_channels, out_channels=self.hidden_channels, K=self.K, normalization=self.normalization, bias=self.bias )
        #self.conv_h_c = GCNConv(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)

        #self.conv_x_c = GraphConvolutionNLayer(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)
        #self.conv_h_c = GraphConvolutionNLayer(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)

        self.b_c = nn.Parameter(torch.Tensor(1, self.hidden_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv(in_channels=self.in_channels, out_channels=self.hidden_channels, K=self.K, normalization=self.normalization, bias=self.bias )
        #self.conv_x_o = GCNConv(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)
        self.conv_h_o = ChebConv(in_channels=self.hidden_channels, out_channels=self.hidden_channels, K=self.K, normalization=self.normalization, bias=self.bias )
        #self.conv_h_o = GCNConv(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)

        #self.conv_x_o = GraphConvolutionNLayer(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)
        #self.conv_h_o = GraphConvolutionNLayer(in_channels=self.hidden_channels, hidden_channels=self.hidden_channels, num_layers=self.K, out_channels=self.hidden_channels)

        self.w_c_o = nn.Parameter(torch.Tensor(1, self.hidden_channels))
        self.b_o = nn.Parameter(torch.Tensor(1, self.hidden_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max,batch=X.shape[0])
        I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max,batch=H.shape[0])
        #I = self.conv_x_i(X, edge_index, edge_weight)
        #I = I + self.conv_h_i(H, edge_index, edge_weight)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max,batch=X.shape[0])
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max,batch=H.shape[0])
        #F = self.conv_x_f(X, edge_index, edge_weight)
        #F = F + self.conv_h_f(H, edge_index, edge_weight)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max,batch=X.shape[0])
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max,batch=H.shape[0])
        #T = self.conv_x_c(X, edge_index, edge_weight)
        #T = T + self.conv_h_c(H, edge_index, edge_weight)
        T = T + self.b_c 
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max, batch=X.shape[0])
        O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max, batch=H.shape[0])
        #O = self.conv_x_o(X, edge_index, edge_weight)
        #O = O + self.conv_h_o(H, edge_index, edge_weight)
        O = O + (self.w_c_o * C)
        O = O + self.b_o    
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H
        

    def forward(self, input, edge_index, edge_weight, hx=None, lambda_max=None):

        # Inputs:
        #       input: of shape (batch_size, nodes, input_size)
        #       hx: of shape (2, batch_size, nodes, hidden_size)
        # Outputs:
        #       hy: of shape (batch_size, nodes, hidden_size)
        #       cy: of shape (batch_size, nodes, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.n_nodes, self.hidden_channels))
            hx = (hx, hx)
        hx, cx = hx

        #H = self._set_hidden_state(input, H)
        #C = self._set_cell_state(input, C)
        I = self._calculate_input_gate(input, edge_index, edge_weight, hx, cx, lambda_max)
        F = self._calculate_forget_gate(input, edge_index, edge_weight, hx, cx, lambda_max)
        C = self._calculate_cell_state(input, edge_index, edge_weight, hx, cx, I, F, lambda_max)
        O = self._calculate_output_gate(input, edge_index, edge_weight, hx, cx, lambda_max)
        H = self._calculate_hidden_state(O, C)

        return (H,C)


class GCNLSTM(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, kernel_size, num_layers):
        super().__init__()

        #self.e_a = e_a
        #self.e_i = e_i

        self.e_i = torch.LongTensor(np.array(
            [[ 0,  0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 12, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 18, 18, 19, 20, 21, 22],
             [ 1,  5,  6,  2,  7,  3,  8,  4,  9,  5, 10, 11,  7, 11,  8, 13,  9, 14, 10, 15, 11, 16, 17, 13, 17, 18, 14, 19, 15, 20, 16, 21, 17, 22, 23, 19, 23, 20, 21, 22, 23]]))
        self.e_a = torch.Tensor(np.array(
              [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))

        self.input_size = input_size[0]
        self.nodes = input_size[1]
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.padding = (3 // 2, 3 // 2)
        self.conv1 = nn.Conv2d(in_channels=self.embedding_size, out_channels=self.hidden_size, kernel_size=(3, 3), padding=self.padding, bias=True)

        self.gcn_cell_list = nn.ModuleList([GConvLSTMCell((self.embedding_size,self.nodes),
                                            self.hidden_size,
                                            self.kernel_size) for _ in range(self.num_layers)])
        #self.gnn = GCN(in_channels=self.embedding_size, hidden_channels=self.hidden_size, num_layers=kernel_size)

        self.linear = torch.nn.Linear(hidden_size, 1)

        self.intermediate_size = 8

        self.nn_model = nn.ModuleList([nn.Sequential(
            nn.Linear(self.intermediate_size, 256, bias=True),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.embedding_size),
            #nn.BatchNorm1d(self.intermediate_size)
        ) for _ in range(self.nodes)])

        self.nn_model2 = nn.Sequential(
            #nn.Linear(self.intermediate_size, 256, bias=True),
            nn.Linear(self.input_size, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.intermediate_size),
            nn.BatchNorm1d(self.intermediate_size)
        )

    def forward(self, input, hx = None):
        # Shapes:
        #   in:  (batch, sentence, nodes, features(inp))
        #   out: (batch, sentence, nodes)
        #   emb: (batch, sentence, nodes, features(emb))

        batch_size = input.size(0)
        sentence_length = input.size(1)
        inputDeep = input.movedim(2,3)
        #print(batch_size)
        #print(sentence_length)
        #print(input.shape)

        # Big FC with the same coefficients for all nodes to get general dependencies
        inputDeep = inputDeep.reshape((batch_size*sentence_length*self.nodes, self.input_size))
        inputDeep = (self.nn_model2(inputDeep)).reshape((batch_size, sentence_length, self.nodes, self.intermediate_size))

        # Small FCs for each node to account for the differences in nodes
        embedded1 = torch.split(inputDeep, 1, dim=2)
        #embedded1 = [inputDeep[:,:,i,:] for i in range(self.nodes)]
        #embedded = nn.Parameter(torch.zeros(batch_size, sentence_length, self.nodes, self.embedding_size))
        output_list = []
        for i in range(self.nodes):
            embedded11 = self.nn_model[i](embedded1[i].reshape((batch_size*sentence_length, self.intermediate_size)))
            embedded11 = embedded11.reshape((batch_size, sentence_length, 1, self.embedding_size))
            #print(embedded[:,:,i,:].shape)
            #print(embedded1[i].shape)

            output_list.append(embedded11)
            #embedded[:,:,i,:] = embedded1[i]
            #else:
            #    embedded = torch.cat((embedded, embedded1[i]), dim=2)
        embedded = torch.cat(output_list, dim=2)
        #shape: batch, sentence, nodes, embedding

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, batch_size, self.nodes, self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, batch_size, self.nodes, self.hidden_size))
        else:
            h0 = hx
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer], h0[layer]))
        output_list1 = []
        #result_tensor = Variable(torch.zeros(batch_size, sentence_length, self.nodes))


        #No GCN: just convolution for each word in a sentence (reshape nodes->4v6 just in this particular case)
        """
        #embeddedt = embedded.movedim(2,3)
        conv_results = torch.empty_like(embedded)  # Create a new tensor for convolution results
        for i in range(sentence_length):
            #conv_input = embeddedt[:, i, :, :].reshape(batch_size, self.embedding_size, 6, 4)
            #conv_results[:, i, :, :] = self.conv1(conv_input).reshape(batch_size, self.hidden_size, self.nodes)
            conv_results[:, i, :, :] = self.gnn(x = embedded[:,i,:,:],edge_index=self.e_i,edge_weight=self.e_a).reshape(batch_size, self.nodes, self.hidden_size)
        embeddedtt = conv_results#.movedim(3, 2)

        #No GCN: just lstm for each node separately
        lstm_results = torch.empty_like(embeddedtt)
        for i in range(self.nodes):
            if torch.cuda.is_available():
                hiddentt = (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
                cellsttt = (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
            else:
                hiddentt = (torch.zeros(self.num_layers, batch_size, self.hidden_size))
                cellsttt = (torch.zeros(self.num_layers, batch_size, self.hidden_size))
            lstm_input = embeddedtt[:, :, i, :]
            lstm_results[:,:,i,:], (hiddentt,cellsttt) = self.rnn(lstm_input, (hiddentt,cellsttt))
        tempResult = self.linear(embeddedtt.reshape((batch_size*sentence_length*self.nodes, self.hidden_size))).reshape((batch_size, sentence_length, self.nodes))
        """


        # LSTM + GCN going "column-by-column" through the sentence
        
        for t in range(sentence_length):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.gcn_cell_list[layer](
                        embedded[:, t, :, :],
                        self.e_i, self.e_a,
                        (hidden[layer][0],hidden[layer][1])
                        )
                    #hidden_l = embedded[:, t, :, :]
                else:
                    hidden_l = self.gcn_cell_list[layer](
                        hidden[layer - 1][0],
                        self.e_i, self.e_a,
                        (hidden[layer][0], hidden[layer][1])
                        )
                    #hidden_l = hidden[layer - 1]

                hidden[layer] = hidden_l

            #print(hidden_l[0].shape)
            newTensor = self.linear(hidden_l[0].reshape((batch_size*self.nodes, self.hidden_size))).reshape((batch_size, 1, self.nodes))
            #print (newTensor.shape)
            #newTensorI = newTensor.unsqueeze(1)
            #result_tensor[:,t,:] = newTensor
            output_list1.append(newTensor)
            #if (t == 0):
            #    result_tensor = newTensorI
            #else:
            #    result_tensor = torch.cat((result_tensor, newTensorI), dim=1)
            #print(hidden_l[0].shape,newTensor.shape, result_tensor.shape)
        result_tensor = torch.cat(output_list1, dim=1)
        
        newTensor11 = self.linear(embedded.reshape((batch_size*sentence_length,self.nodes, self.hidden_size))).reshape((batch_size, sentence_length, self.nodes))

        #print(e_i.shape)
        #print(e_a.shape)
        print(result_tensor.shape)
        #print(tempResult.shape)

        # result shape: (batch,sentence,nodes)
        return newTensor11



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

        #self.conv = nn.Conv2d(in_channels=self.hidden_size,
        #                     out_channels=self.output_size,
        #                     kernel_size=self.kernel_size,
        #                     padding=self.padding,
        #                     bias=self.bias)

        self.conv1 = nn.Conv2d(in_channels=self.hidden_size, out_channels=1, kernel_size=(1, 1))

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

        self.intermediate_size = 8

        self.nn_model = nn.ModuleList([nn.Sequential(
            nn.Linear(self.intermediate_size, 256, bias=True),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.embedding_size),
            #nn.BatchNorm1d(self.intermediate_size)
        ) for _ in range(self.height)])

        self.nn_model2 = nn.Sequential(
            #nn.Linear(self.intermediate_size, 256, bias=True),
            nn.Linear(self.input_size, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.intermediate_size),
            nn.BatchNorm1d(self.intermediate_size)
        )



    def forward(self, input, hx=None):

        batch_size = input.size(0)
        sentence_length = input.size(1)

        inputDeep = input.movedim(-3,-1)

        inputDeep = inputDeep.reshape((batch_size*sentence_length*self.height*self.width, self.input_size))
        inputDeep = (self.nn_model2(inputDeep)).reshape((batch_size, sentence_length, self.height, self.width, self.intermediate_size))

        embedded1 = [inputDeep[:,:,i,:,:] for i in range(self.height)]
        embedded = torch.Tensor(batch_size, sentence_length, 1, self.width, self.embedding_size)
        for i in range(self.height):
            embedded1[i] = (self.nn_model[i](embedded1[i].reshape((batch_size*sentence_length*self.width, self.intermediate_size))))
            embedded1[i] = embedded1[i].reshape((batch_size, sentence_length, 1, self.width, self.embedding_size))
            if (i == 0):
                embedded = embedded1[i]
            else:
                embedded = torch.cat((embedded, embedded1[i]), dim=2)

        #embedded = embedded.reshape((batch_size*sentence_length*self.height*self.width, self.intermediate_size))
        #embedded = inputDeep.reshape((batch_size*sentence_length*self.height*self.width, self.input_size))
        #embedded = (self.nn_model2(embedded)).reshape((batch_size, sentence_length, self.height, self.width, self.embedding_size))
        
        #inputDeep1 = inputDeep.reshape((batch_size*sentence_length, self.height*self.width*self.input_size))
        #embedded = self.nn_model(inputDeep1)
        #embedded = embedded.reshape((batch_size, sentence_length, self.height, self.width, self.embedding_size))

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

        
        result_tensor = torch.Tensor(batch_size, 1, self.height, self.width)
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
            newTensorI = newTensor.squeeze().unsqueeze(1)
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

