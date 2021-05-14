from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import PalindromeDataset
from lstm import LSTM
# from lstm_E import LSTM

def get_correct(predicts,targets):
    _,pred_label=torch.max(predicts.data,1)
    is_true=(pred_label==targets).data
    return is_true.sum().item()

def train(config):

    gpu = torch.cuda.is_available()
    # Initialize the model that we are going to use
    # print("E")
    model = LSTM(seq_length=config.input_length,input_dim=config.input_dim,hidden_dim=config.num_hidden,output_dim=config.num_classes,
                 batch_size=config.batch_size)
    if gpu:
        model.cuda()

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # Adjust learning rate
    lrs = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.96)

    with tqdm(total=config.batch_size*config.train_steps) as pbar:
        x_lst = []
        acc_lst = []
        loss_lst = []
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            if gpu:
                batch_inputs, batch_targets = batch_inputs.cuda().float(), batch_targets.cuda()
            optimizer.zero_grad()
            predicts = model(batch_inputs)
            loss = criterion(predicts, batch_targets)
            loss.backward()
            optimizer.step()
            lrs.step()
            # the following line is to deal with exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

            # Add more code here ...
            accuracy = get_correct(predicts, batch_targets) / config.batch_size
            pbar.update(config.batch_size)

            if step % 10 == 0:
                # print acuracy/loss here
                x_lst.append(step)
                acc_lst.append(accuracy)
                loss_lst.append(loss.item())

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')
    return x_lst,loss_lst,acc_lst,model

def train_with_model(config,model):
    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    if torch.cuda.is_available():
        model = model.cuda()

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0,
                                    momentum=0, centered=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    with tqdm(total=config.batch_size * config.train_steps) as pbar:
        x_lst = []
        acc_lst = []
        loss_lst = []
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                batch_inputs = batch_inputs.cuda()
                batch_targets = batch_targets.cuda()
            optimizer.zero_grad()
            predicts = model(batch_inputs)
            # the following line is to deal with exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

            # Add more code here ...

            loss = criterion(predicts, batch_targets)
            loss.backward()
            optimizer.step()
            accuracy = get_correct(predicts, batch_targets) / config.batch_size
            pbar.update(config.batch_size)
            scheduler.step()

            if step % 10 == 0:
                # print acuracy/loss here
                x_lst.append(step)
                acc_lst.append(accuracy)
                loss_lst.append(loss.item())

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')
    return x_lst, loss_lst, acc_lst, model

def show_accu(x_axis1,y_axis1,y_axis2,x_label,y_label):
    # %matplotlib inline
    plt.plot(x_axis1,y_axis1,'g--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    plt.plot(x_axis1,y_axis2,'b--')
    plt.xlabel(x_label)
    plt.ylabel("accuracy")
    plt.show()

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=1500, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # model = LSTM(seq_length=config.input_length, input_dim=config.input_dim, hidden_dim=config.num_hidden,
    #              output_dim=config.num_classes,
    #              batch_size=config.batch_size)
    x_lst,loss_lst,acc_lst,model = train(config)
    show_accu(x_axis1=x_lst,y_axis1=loss_lst,y_axis2=acc_lst,x_label="train step",y_label="loss")