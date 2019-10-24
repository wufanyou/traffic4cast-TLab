import torch
import torch.nn as nn
import torch.nn.functional as F

def mapActivation(name):
    if name == 'relu':
        activation = torch.nn.ReLU()
    elif name == 'swish':
        activation = Swish()
    return activation
#  (GMT+2)
# 30  4:30-4:45
# 69  7:45-8:00
# 126 12:30-12:45
# 186 17:15-17:30
# 234 21:15-21.30

# （GMT+3）
# 57  7:45-8:00
# 114 12:30-12:45
# 174 17:15-17:30
# 222 21:15-21.30
# 258 +1 0:30-0:45

def getPredictIndex(city):
    if city == 'Berlin':
        index = [30, 69, 126, 186, 234]
    elif city in ['Istanbul','Moscow']:
        index = [57, 114, 174,222, 258]
    return index