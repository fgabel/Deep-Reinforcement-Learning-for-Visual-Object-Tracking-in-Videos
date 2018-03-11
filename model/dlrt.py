#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init





class DLRTnet(nn.Module):
    """DLRTnet is based on https://arxiv.org/abs/1701.08936
    # write something about the architecture
    
    """
    def __init__(self, *args):
        """args:
            - a
            - b
            - c 
        """
        
    @staticmethod
    def weight_init(m):
        """ call this for weight initialisation 
        """
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        """ call this for weight reset in each of the modules
        # we might not need this
        """
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):

        return x