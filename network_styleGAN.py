import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super(ApplyNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def fowra


class G_style(nn.Module):

class StyleGenerator(nn.Module):
    def __init__(self,):

class StyleDiscriminator(nn.Module):
    def __init__(self,):
        