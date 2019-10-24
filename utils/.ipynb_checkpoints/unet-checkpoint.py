# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
import torch.nn.functional as F

from .unet_parts import *
from .modules import MeanThreshold
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,embed=10):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.dense1 = nn.Linear(embed,32,bias=True)
        self.dense2 = nn.Linear(32,65,bias=False)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=False)
        self.up2 = up(512, 128, bilinear=False)
        self.up3 = up(256, 64, bilinear=False)
        self.up4 = up(128, 64, bilinear=False)
        self.outc = outconv(65, n_classes)
        self.MeanThreshold = MeanThreshold(-1,0) 
    def forward(self, x,e):
        m = self.MeanThreshold(x)
        #e = F.relu(self.dense1(e))
        #e = self.dense2(e)
        x1 = self.inc(x)#+e.reshape(-1,64,1,1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([x,m],1)
        x = x  #e.reshape(-1,65,1,1)
        x = self.outc(x)
        x = torch.tanh(x)
        return x