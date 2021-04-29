import torch.nn as nn
import torch.nn.functional as F


class MC(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(3*32*32,120)
        self.linear2=nn.Linear(120,84)
        self.linear3=nn.Linear(84,10)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x):
        x=x.view(-1,3*32*32)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x