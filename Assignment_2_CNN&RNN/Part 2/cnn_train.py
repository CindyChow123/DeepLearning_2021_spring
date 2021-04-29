from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import datasets, transforms,utils
from tqdm import tqdm

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
# MAX_EPOCHS_DEFAULT = 2
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = '../data'
TEST_SIZE=10000

FLAGS = None

def get_cifar10(batch_size):
    # transform PIL image to tensors(normalize not support pil), normalize - dataloader needs
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # load train and test img set
    train_set = datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
    # iterate
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    # classes
    classes = ('airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader,test_loader,classes


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
    _, class_label = torch.max(predictions.data, 1)
    # tensor [true,false]; tensor[number]
    is_true = (class_label == targets).data
    return is_true.sum().item()

def train(net,batch_size=BATCH_SIZE_DEFAULT,max_epoch=MAX_EPOCHS_DEFAULT,eval_freq=EVAL_FREQ_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    train_loader, test_loader, classes = get_cifar10(batch_size)
    # model definition
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    #train
    avg_loss = 0
    x_axis = []
    x_axis_test = []
    loss_lst = []
    loss_lst_test = []
    accu_lst = []
    with tqdm(total=50000 * max_epoch) as pbar:
        pbar.set_description('model training:')
        for epoch in range(max_epoch):
            for index, data in enumerate(train_loader, 0):
                imgs, labels = data

                optimizer.zero_grad()

                results = net(imgs)
                loss = criterion(results, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                if index % eval_freq == eval_freq - 1:  # print every mini-batch
                    # print('[epoch %d, total %5d] loss: %.3f' %
                    #       (epoch + 1, (index+1)*batch_size+epoch*50000, avg_loss / EVAL_FREQ_DEFAULT))
                    # training loss
                    x_axis.append((index + 1) * batch_size + epoch * 50000)
                    loss_lst.append(avg_loss / eval_freq)
                    avg_loss = 0.0
                pbar.update(batch_size)
                    # test loss
                # print("---index",index," finish---")
            with torch.no_grad():
                test_avg_loss = 0
                correct=0
                for data in test_loader:
                    images, labels = data
                    outputs = net(images)
                    # get loss
                    test_loss = criterion(outputs, labels)
                    test_avg_loss += test_loss.item()
                    # get classification accuracy
                    correct += accuracy(outputs,labels)
                test_freq=TEST_SIZE/batch_size
                # print('test loss: %.3f' % (test_avg_loss / test_freq))
                loss_lst_test.append(test_avg_loss / test_freq)  # TEST SIZE/BATCH=sum of how many loss
                x_axis_test.append(50000 * (epoch + 1))
                accu_lst.append(correct/TEST_SIZE)

    print("Training finish\n")
    return net, x_axis, loss_lst, x_axis_test, loss_lst_test,accu_lst


def main():
    """
    Main function
    """
    net, x_axis, loss_lst, x_axis_test, loss_lst_test,accu_lst=train(max_epoch=2)

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()