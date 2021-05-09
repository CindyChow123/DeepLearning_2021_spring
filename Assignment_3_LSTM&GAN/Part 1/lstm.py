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

        self.g = nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.i = nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.o = nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.f = nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.output = nn.Linear(hidden_dim,output_dim)

    def forward(self, x):
        # Implementation here ...
        y_lst = torch.zeros(self.batch_size, self.output_dim)
        for index in range(len(x)):
            seq=x[index]
            h_last=torch.unsqueeze(torch.zeros(self.hidden_dim),0)
            c_last=torch.unsqueeze(torch.zeros(self.hidden_dim),0)
            for j in range(len(seq)):
                n = torch.unsqueeze(torch.tensor([seq[j]]),0)
                xh = torch.cat((n,h_last),dim=1)
                g = self.tanh(self.g(xh))
                i = self.sigmoid(self.i(xh))
                f = self.sigmoid(self.f(xh))
                o = self.sigmoid(self.o(xh))
                c_last = g*i+c_last*f # c now
                h_last = self.tanh(c_last)*o # h now

                if j == len(seq)-1:
                    # score on output_dim size of classes
                    y = self.output(h_last)
                    y_lst[index]=y
        return y_lst
        
    # add more methods here if needed