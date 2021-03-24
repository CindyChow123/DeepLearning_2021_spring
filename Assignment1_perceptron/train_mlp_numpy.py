from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.datasets import make_moons

import argparse
import numpy as np
import os

from sklearn.model_selection import train_test_split

from mlp_numpy import MLP
from modules import CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
METHOD = 'batch' #'sgd' or 'batch'

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
        if predictions[i][0] == targets[i][0] or predictions[i][1] == targets[i][1]:
            accuracy += 1
    accuracy = accuracy/len(predictions)
    return accuracy

def prob_to_class(prob):
    if prob[1] > prob[0]:
        return [0,1]
    else:
        return [1,0]

def train(method):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    # Generate training data
    X,Y = make_moons(1000,random_state=25)
    Y_one_hot = np.eye(2)[Y]
    X_train,X_test,y_train,y_test = train_test_split(X,Y_one_hot,test_size=0.2,random_state=42)
    # Init MLP
    n_hidden = DNN_HIDDEN_UNITS_DEFAULT.split(',')
    n_hidden = [int(x) for x in n_hidden]
    loss_layer = CrossEntropy()
    mlp = MLP(2,n_hidden,2)
    test_accu = []
    train_accu = []
    x_axis =[]
    # train loop
    for j in range(MAX_EPOCHS_DEFAULT):
        # print('---epoch',j,'---')
        if method == 'sgd':
            for i in range(len(X_train)):
                predict = mlp.forward(X_train[i])
                # loss = loss_layer.forward(predict,y_train)
                dout = loss_layer.backward(predict,y_train[i])
                dx = mlp.backward(dout)

                # update the weight
                mlp.output_layer[0].params['weight'] = mlp.output_layer[0].params['weight'] - LEARNING_RATE_DEFAULT * mlp.output_layer[0].grads['weight']
                mlp.output_layer[0].params['bias'] = mlp.output_layer[0].params['bias'] - LEARNING_RATE_DEFAULT * mlp.output_layer[0].grads['bias']
                for layer in mlp.layers:
                    layer[0].params['weight'] = layer[0].params['weight'] - LEARNING_RATE_DEFAULT * \
                                                           layer[0].grads['weight']
                    layer[0].params['bias'] = layer[0].params['bias'] - LEARNING_RATE_DEFAULT * \
                                                         layer[0].grads['bias']
        else:
            dw_out = 0
            db_out = 0
            dw_hid = dict()
            db_hid = dict()
            for i in range(len(X_train)):
                predict = mlp.forward(X_train[i])
                # loss = loss_layer.forward(predict,y_train)
                dout = loss_layer.backward(predict, y_train[i])
                dx = mlp.backward(dout)

                # store the grad of weight and bias
                dw_out += mlp.output_layer[0].grads['weight']
                db_out += mlp.output_layer[0].grads['bias']
                for i in range(len(mlp.layers)):
                    if i not in dw_hid.keys():
                        dw_hid[i] = mlp.layers[i][0].grads['weight']
                    else:
                        dw_hid[i] += mlp.layers[i][0].grads['weight']
                    if i not in db_hid.keys():
                        db_hid[i] = mlp.layers[i][0].grads['bias']
                    else:
                        db_hid[i] += mlp.layers[i][0].grads['bias']
            # update the grad in a batch
            mlp.output_layer[0].params['weight'] = mlp.output_layer[0].params['weight'] - LEARNING_RATE_DEFAULT * dw_out / 800
            mlp.output_layer[0].params['bias'] = mlp.output_layer[0].params['bias'] - LEARNING_RATE_DEFAULT * db_out / 800
            for i in range(len(mlp.layers)):
                mlp.layers[i][0].params['weight'] = mlp.layers[i][0].params['weight'] - LEARNING_RATE_DEFAULT * dw_hid[i]
                mlp.layers[i][0].params['bias'] = mlp.layers[i][0].params['bias'] - LEARNING_RATE_DEFAULT * db_hid[i]

        # evaluation
        if j % EVAL_FREQ_DEFAULT == 0:
            x_axis.append(j)
            plabels = np.empty(y_test.shape)
            for i in range(len(X_test)):
                eval_predict = mlp.forward(X_test[i])
                c = prob_to_class(eval_predict)
                plabels[i]=c
            # eval_loss = loss_layer.forward(plabels,y_test)
            # eval_losses.append(eval_loss)
            a = accuracy(plabels,y_test)
            test_accu.append(a)
            # print('---test accuracy',a,' ---')

            plabels = np.empty(y_train.shape)
            for i in range(len(X_train)):
                eval_predict = mlp.forward(X_train[i])
                c = prob_to_class(eval_predict)
                plabels[i] = c
            # eval_loss = loss_layer.forward(plabels,y_test)
            # eval_losses.append(eval_loss)
            a = accuracy(plabels, y_train)
            train_accu.append(a)
            # print('---train accuracy', a, ' ---')

    return x_axis,test_accu,train_accu

def main(method):
    """
    Main function
    """
    x_axis, test_acc, train_acc = train(method)
    return x_axis, test_acc,train_acc

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
    main(METHOD)