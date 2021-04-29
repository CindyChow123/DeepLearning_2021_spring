from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.layers = []
        for i in range(len(n_hidden)):
            if i == 0:
                self.layers.append([Linear(n_inputs,n_hidden[0]),ReLU()])
            else:
                self.layers.append([Linear(n_hidden[i-1],n_hidden[i]),ReLU()])
        self.output_layer = [Linear(n_hidden[len(n_hidden)-1],n_classes),SoftMax()]

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for layer in self.layers:
            x = layer[0].forward(x)
            x = layer[1].forward(x)
        x = self.output_layer[0].forward(x)
        out = self.output_layer[1].forward(x)
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        dou = self.output_layer[1].backward(dout)
        dou = self.output_layer[0].backward(dou)
        for i in range(len(self.layers)):
            dou = self.layers[-i-1][1].backward(dou)
            dou = self.layers[-i-1][0].backward(dou)
        return
