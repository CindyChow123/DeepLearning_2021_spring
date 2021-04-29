from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn
import torch

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.relu = nn.ReLU()
        self.linear1=nn.Linear(n_inputs,n_hidden[0])
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.linear_mid = nn.ModuleList([nn.Linear(n_hidden[i-1],n_hidden[i]) for i in range(1,len(n_hidden))])
        for i in range(len(self.linear_mid)):
            torch.nn.init.xavier_uniform_(self.linear_mid[i].weight)
        self.linear2 = nn.Linear(n_hidden[-1],n_classes)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        # self.softmax=nn.Softmax(dim=0)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        x=torch.from_numpy(x).float()
        x=x.view(-1,self.n_inputs)
        x=self.relu(self.linear1(x))
        for i in range(len(self.linear_mid)):
            x=self.linear_mid[i](x)
            x=self.relu(x)
        out=self.linear2(x)
        # out=self.softmax(x)
        return out
