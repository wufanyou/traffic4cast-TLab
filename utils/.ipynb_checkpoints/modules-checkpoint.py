import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self,x):
        x = x * F.sigmoid(x)
        return x
    
class MeanThreshold(nn.Module):
    def __init__(self,threshold=0,ids=0):
        super(MeanThreshold, self).__init__()
        self.threshold = threshold
        self.ids = ids
    def forward(self, x):
        x = x[:,self.ids:,:,:]
        x = x.mean(1,keepdim=True)
        x = (x>self.threshold).type(x.dtype)
        return x
    