import torch
import torch.nn as nn
import numpy as np
from dragon.search_space.cells import Brick
from dragon.utils.tools import logger

from dragon.search_space.cells import Brick


class Conv2d(Brick):
    def __init__(self, input_shape, out_channels, kernel_size, stride=None, padding=None):
        super(Conv2d, self).__init__(input_shape)
        h, w, in_channels = input_shape
        self.kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        if stride is None:
            stride = 1
        if padding is not None:
            if padding == "same":
                stride = 1
        else:
            padding = "valid"
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=stride, padding=padding)

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        if (self.kernel_size[0]>X.shape[-2]) or (self.kernel_size[1] > X.shape[-1]):
            X = self.pad(X)
        X = self.cnn(X)
        X = X.permute(0, 2, 3, 1)
        return X
    
    def pad(self,X):
        diff0 = self.kernel_size[0] - X.shape[-2]
        if diff0<0:
            diff0 = 0
        diff1 =self.kernel_size[1] - X.shape[-1]
        if diff1<0:
            diff1 = 0
        pad = (int(np.ceil(np.abs(diff1)/2)), int(np.floor(np.abs(diff1))/2), int(np.ceil(np.abs(diff0)/2)), int(np.floor(np.abs(diff0))/2))
        return nn.functional.pad(X, pad)
    
    def modify_operation(self, input_shape):
        h,w,d_in = input_shape
        diff = d_in - self.in_channels
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (0,0,0,0,int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.cnn.weight.data = nn.functional.pad(self.cnn.weight, pad)
        self.in_channels = d_in

    def load_state_dict(self, state_dict, **kwargs):
        input_shape = state_dict['cnn.weight'].shape[1]
        self.modify_operation(input_shape)
        super(Conv2d, self).load_state_dict(state_dict, **kwargs)
        




class Conv1d(Brick):
    def __init__(self, input_shape, kernel_size, out_channels, padding="same"):
        super(Conv1d, self).__init__(input_shape)
        _, d_in = input_shape
        self.kernel_size = kernel_size
        self.in_channels = d_in
        self.cnn = nn.Conv1d(in_channels=d_in, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1)

    def forward(self, X):
        X = X.permute(0,2,1)        
        if self.kernel_size>X.shape[-1]:
            X = self.pad(X)
        X = self.cnn(X)
        X = X.permute(0,2,1)
        return X
    
    def modify_operation(self, input_shape):
        _, d_in = input_shape
        diff = d_in - self.in_channels
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (0,0,int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.cnn.weight.data = nn.functional.pad(self.cnn.weight, pad)
        self.in_channels = d_in

    def pad(self,X): 
        if len(X.shape) > 2:
            bs, c, d_in = X.shape
        else:
            bs, d_in = X.shape
        diff = self.kernel_size - d_in
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        return nn.functional.pad(X, pad)
    
