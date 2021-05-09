from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN

def get_correct(predicts,targets):
    _,pred_label=torch.max(predicts.data,1)
    is_true=(pred_label==targets).data
    return is_true.sum().item()

def train(config):

    # Initialize the model that we are going to use
    model = VanillaRNN(seq_length=config.input_length,input_dim=config.input_dim,hidden_dim=config.num_hidden,output_dim=config.num_classes,batch_size=config.batch_size)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    with tqdm(total=config.batch_size*config.train_steps) as pbar:
        x_lst=[]
        acc_lst=[]
        loss_lst=[]
        pbar.set_description('model training:')
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            optimizer.zero_grad()
            predicts=model(batch_inputs)
            # the following line is to deal with exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

            loss = criterion(predicts,batch_targets)
            loss.backward()
            optimizer.step()
            accuracy = get_correct(predicts,batch_targets)/config.batch_size
            pbar.update(config.batch_size)
            if step % 10 == 0:
                # print acuracy/loss here
                # print("\n---accuracy",accuracy," ---")
                # print("***loss",loss.item()," ***")
                x_lst.append(step)
                acc_lst.append(accuracy)
                loss_lst.append(loss.item())

            if step == config.train_steps:
                break
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655

    print('Done training.')
    return x_lst,loss_lst,acc_lst

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence') # 10
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')# 10
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps') #10000
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config)
    print(type(config))