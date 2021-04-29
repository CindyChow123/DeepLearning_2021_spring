from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.u = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(hidden_dim, output_dim)
        # self.sm = nn.Softmax()

    def forward(self, x):
        # Implementation here ...
        # every batch, init hidden, small input ok(<=5), batch item=one word
        y_lst = torch.zeros(self.batch_size,self.output_dim)
        for index in range(len(x)):
            seq=x[index]
            s_last=torch.unsqueeze(torch.zeros(self.hidden_dim),0)
            for i in range(len(seq)):
                s = self.tanh(self.u(torch.unsqueeze(seq[i],0)) + self.w(s_last))
                s_last = s
                if i == len(seq)-1:
                    y = self.v(s)
                    # y = self.sm(v)
                    y_lst[index]=y
        return y_lst

    # add more methods here if needed
