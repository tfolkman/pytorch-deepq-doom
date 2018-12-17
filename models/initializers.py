from torch import nn


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal(m.weight)