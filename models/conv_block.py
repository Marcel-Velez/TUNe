####
# Convolution block like also used in the SampleCNN model from https://github.com/Spijkervet/CLMR
# This block is used in every variant of the TUNe architecture
####

import torch.nn as nn

def initialize(m):
    if isinstance(m, (nn.Conv1d)):

        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, stride, padding):
        super(ConvBlock, self).__init__()

        self.layers = []
        self.layers.append(nn.Conv1d(in_chan, out_chan, kernel_size=kernel, stride=stride, padding=padding))
        self.layers.append(nn.BatchNorm1d(out_chan))
        self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)

        self.layers.apply(initialize)

    def forward(self, x):

        out = self.layers(x)

        return out