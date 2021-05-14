from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.device = device

        # self.g = nn.Linear(input_dim+hidden_dim,hidden_dim)
        # self.i = nn.Linear(input_dim+hidden_dim,hidden_dim)
        # self.o = nn.Linear(input_dim+hidden_dim,hidden_dim)
        # self.f = nn.Linear(input_dim+hidden_dim,hidden_dim)
        # self.output = nn.Linear(hidden_dim,output_dim)
        self.gx = nn.Linear(input_dim, hidden_dim)
        self.gh = nn.Linear(hidden_dim, hidden_dim)
        self.ix = nn.Linear(input_dim, hidden_dim)
        self.ih = nn.Linear(hidden_dim, hidden_dim)
        self.fx = nn.Linear(input_dim, hidden_dim)
        self.fh = nn.Linear(hidden_dim, hidden_dim)
        self.ox = nn.Linear(input_dim, hidden_dim)
        self.oh = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = x.unsqueeze(2)
        # Implementation here ...
        device_comp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # y_lst = torch.zeros(self.batch_size, self.output_dim,device=device_comp)
        h_last =torch.zeros(self.hidden_dim,self.hidden_dim,device=device_comp)
        c_last=torch.zeros(self.hidden_dim,self.hidden_dim,device=device_comp)
        self.bg = torch.zeros(self.batch_size, self.batch_size,device=device_comp)
        self.bi = torch.zeros(self.batch_size, self.batch_size,device=device_comp)
        self.bf = torch.zeros(self.batch_size, self.batch_size,device=device_comp)
        self.bo = torch.zeros(self.batch_size, self.batch_size,device=device_comp)
        self.bp = torch.zeros(self.batch_size, self.output_dim,device=device_comp)


        for j in range(self.seq_length):
            n = x[:,j]
            xh = torch.cat((n,h_last),dim=1)
            g = self.tanh(self.gx(x[:,j])+self.gh(h_last)+self.bg)
            i = self.sigmoid(self.ix(x[:,j])+self.ih(h_last)+self.bi)
            f = self.sigmoid(self.fx(x[:,j])+self.fh(h_last)+self.bf)
            o = self.sigmoid(self.ox(x[:,j])+self.oh(h_last)+self.bo)
            c_last = g*i+c_last*f # c now
            h_last = self.tanh(c_last)*o # h now


        # score on output_dim size of classes
        y = self.output(h_last)+self.bp
        return y
        
    # add more methods here if needed