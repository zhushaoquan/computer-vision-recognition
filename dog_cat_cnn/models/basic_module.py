# coding:utf8
import torch
from torch import nn
import time
import os


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)

    # def get_optimizer(self, lr, weight_decay):
    #     return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
