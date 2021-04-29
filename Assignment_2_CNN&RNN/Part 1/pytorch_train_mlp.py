from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from torch.nn import CrossEntropyLoss
from pytorch_mlp import MLP
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import cross_entropy


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i][targets[i]] == 1:
            accuracy += 1
    accuracy = accuracy / len(predictions)
    return accuracy

def prob_to_class(prob_lst):
    label_lst=[]
    for prob in prob_lst:
        if prob[1] > prob[0]:
            label_lst.append([0,1])
        else:
            label_lst.append([1,0])
    return label_lst

def train(mode):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    # Generate training data
    X, Y = make_moons(1000, random_state=25)
    # Y_one_hot = np.eye(2)[Y] # CrossEntropy will automatically turn a label into one-hot encoder
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Init MLP
    n_hidden = DNN_HIDDEN_UNITS_DEFAULT.split(',')
    n_hidden = [int(x) for x in n_hidden]
    # criterion = CrossEntropyLoss()
    # criterion = cross_entropy()
    mlp=MLP(2,n_hidden=n_hidden,n_classes=2)
    if mode=='sgd':
        acc_test=[]
        x_axis=[]
        optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001, momentum=0)
        for i in range(0,MAX_EPOCHS_DEFAULT):
            print('---epoch '+str(i)+' running---')
            for j in range(len(X_train)):
                xi=np.expand_dims(X_train[j],0)
                yi=torch.tensor([y_train[j]])
                train_predict = mlp.forward(xi)
                loss = cross_entropy(train_predict, yi)
                # print(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i%EVAL_FREQ_DEFAULT==0:
                with torch.no_grad():
                    mlp.eval()
                    x_axis.append(i)
                    test_predict=mlp.forward(X_test)
                    label_lst = prob_to_class(test_predict)
                    acc_test.append(accuracy(label_lst,y_test))
                    print('---Accuracy='+str(acc_test[-1]))
                    mlp.train()
    else:
        acc_test = []
        x_axis = []
        optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
        for i in range(0, MAX_EPOCHS_DEFAULT):
            print('---epoch '+str(i)+' running---')
            # zero grad for every batch
            train_predict = mlp.forward(X_train)
            loss = cross_entropy(train_predict, torch.tensor(y_train))
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update
            with torch.no_grad():
                for param in mlp.parameters():
                    param -= LEARNING_RATE_DEFAULT * param.grad



            if i % EVAL_FREQ_DEFAULT == 0:
                x_axis.append(i)
                test_predict = mlp.forward(X_test)
                label_lst = prob_to_class(test_predict)
                acc_test.append(accuracy(label_lst, y_test))
                print('---Accuracy=' + str(acc_test[-1]))

        return acc_test,x_axis




def main():
    """
    Main function
    """
    acc_test,x_axis = train('sgd')
    # acc_test,x_axis = train('batch')
    print(acc_test)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()