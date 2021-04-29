from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object.

    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super().__init__()
    self.n_channels=n_channels
    self.n_classes=n_classes
    self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
    self.bn1=nn.BatchNorm2d(num_features=64)
    self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
    self.bn2=nn.BatchNorm2d(num_features=128)
    self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
    self.bn3=nn.BatchNorm2d(num_features=256)
    self.conv4=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
    self.bn4=nn.BatchNorm2d(num_features=256)
    self.conv5=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
    self.bn5=nn.BatchNorm2d(num_features=512)
    self.conv6=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
    self.bn6=nn.BatchNorm2d(num_features=512)
    self.conv7=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
    self.bn7=nn.BatchNorm2d(num_features=512)
    self.conv8=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
    self.bn8=nn.BatchNorm2d(num_features=512)
    self.linear=nn.Linear(in_features=512,out_features=10)
    self.relu=nn.ReLU()



  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    x=self.max_pool(self.relu(self.bn1(self.conv1(x))))

    x=self.max_pool(self.relu(self.bn2(self.conv2(x))))

    x=self.relu(self.bn3(self.conv3(x)))
    x=self.max_pool(self.relu(self.bn4(self.conv4(x))))

    x = self.relu(self.bn5(self.conv5(x)))
    x = self.max_pool(self.relu(self.bn6(self.conv6(x))))

    x = self.relu(self.bn7(self.conv7(x)))
    x = self.max_pool(self.relu(self.bn8(self.conv8(x))))

    x=x.view(-1,512)
    out=self.linear(x)




    return out
