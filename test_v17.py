# %load valid_v6.py
from utils import *
#from utils import DatasetFolderV15 as DatasetFolder
import numpy as np
from fastprogress import master_bar,progress_bar
import time
import h5py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np

valueMap=np.vectorize(lambda x:{0:0,1:1,2:85,3:170,4:255}[x])
  
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='./processed/')
    parser.add_argument('-o', '--output_dir', type=str, default='./')
    parser.add_argument('-m', '--model_dir', type=str, default='/data/data20180901/weights/v17')
    parser.add_argument('-c', '--city', type=str, default='Moscow')
    parser.add_argument('-ch', '--channel', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--step', type=int, default=12)
    parser.add_argument('--version', type=str, default='v17')
    parser.add_argument('--suffix', type=str, default='best')
    parser.add_argument('--rename', type=str, default='0')
    args = parser.parse_args()
    return args

args = getArgs()
IN_PATH = args.input_dir
OUT_PATH = args.output_dir
MODEL_PATH = args.model_dir
CITY = args.city
DEVICE = f'cuda:{args.device}'
VERSION = args.version
STEP = 12

CROP_DICT = [(0,299,0,299),(0,299,137,None),(196,None,137,None),(196,None,0,299),(98,397,68,367)]
SUFFIX = args.suffix
CHANNEL = args.channel
OUTVERSION = VERSION if args.rename=='0' else args.rename

VERSION_MAP={2:{'Moscow':[0,1,2,3,4],'Istanbul':[0,1,2,3,4],'Berlin':[0,1,2,3,4]},
             1:{'Moscow':[0,1,2,3,4],'Istanbul':[0,1,2,3,4],'Berlin':[0,1,2,3,4]},
            }

if __name__=='__main__':
    total_loss=0
    CROP_RESULT=[]
    for CROP in range(5):
        CROP_SIZE = CROP_DICT[CROP]
        index = getPredictIndex(CITY)#[0:1]
        
        if CROP in VERSION_MAP[CHANNEL][CITY]:
            LEAK_STEP = 65 
            model = UNet(STEP*3+LEAK_STEP,3,20).to(DEVICE)
            try:
                model.load_state_dict(torch.load(f'{MODEL_PATH}/{VERSION}_{CITY}_{CHANNEL}_{CROP}_{SUFFIX}.pth'))
            except:
                model.load_state_dict(torch.load(f'{MODEL_PATH}/../v16/v16_{CITY}_{CHANNEL}_{CROP}_{SUFFIX}.pth'))
            
            folder = DatasetFolderV17(IN_PATH,CITY,'test',index,STEP,CHANNEL,is_transform=False,predict_length=3,skip=0,crop=CROP_SIZE)
        else:
            LEAK_STEP = 33 
            model = UNet(STEP*3+LEAK_STEP,3,20).to(DEVICE)
            try:
                model.load_state_dict(torch.load(f'{MODEL_PATH}/../v15/v15_{CITY}_{CHANNEL}_{CROP}_{SUFFIX}.pth'))
            except:
                model.load_state_dict(torch.load(f'{MODEL_PATH}/../v15/v15_{CITY}_{CHANNEL}_0_{SUFFIX}.pth'))
            folder = DatasetFolderV15(IN_PATH,CITY,'test',index,STEP,CHANNEL,is_transform=False,predict_length=3,skip=0,crop=CROP_SIZE)
        count = 0
        #ch_loss = 0
        model = model.eval()
        mean_zero=0
        RESULT={}
        for X,Y in folder:
            X = X.to(DEVICE)[None,:]
            meta=folder.Y[count]
            #MAP = torch.tensor(np.load(f'{IN_PATH}/{CITY}/day_zero/{meta[0]}/{CHANNEL}.npy')).reshape([1,1,495,436])
            #MAP = torch.cat([MAP,MAP,MAP],1).to(DEVICE)
            #MAP = MAP[:,:,CROP_SIZE[0]:CROP_SIZE[1],CROP_SIZE[2]:CROP_SIZE[3]]
            count+=1
            with torch.set_grad_enabled(False):
                out = (model(X*2/255-1,None)+1)/2*255
                out = torch.round(out)
                out = torch.clamp(out,0,255)
                #out[MAP]=0
            RESULT[(meta[0],meta[1][0])]=(out.detach().cpu())
        print(f"[{CITY}] [{CHANNEL}] {CROP_SIZE} finish")
        CROP_RESULT.append(RESULT.copy())
    folder = DatasetFolderV16(IN_PATH,CITY,'test',index,STEP,CHANNEL,is_transform=False,predict_length=3,skip=0,crop=(0,None,0,None))
    
    for count in range(len(folder)):
        meta=folder.Y[count]
        X=torch.zeros([1,3,495,436])
        C=torch.zeros([1,3,495,436])
        for CROP in range(5):
            Z1=torch.zeros([1,3,495,436])
            Z2=torch.zeros([1,3,495,436])
            idx = CROP_DICT[CROP]
            Z1[:,:,idx[0]:idx[1],idx[2]:idx[3]]=CROP_RESULT[CROP][(meta[0], meta[1][0])]
            Z2[:,:,idx[0]:idx[1],idx[2]:idx[3]]=1
            X+=Z1
            C+=Z2
        X /= C
        X = torch.round(X)
        X = torch.clamp(X,0,255)
        
        try:
            os.makedirs(f'{OUT_PATH}/result/numpy/{OUTVERSION}/{CITY}/{meta[0]}/{CHANNEL}')
        except:
            pass
        
        output=X.permute(1,2,3,0).numpy().astype(np.uint8)
        np.save(f'{OUT_PATH}/result/numpy/{OUTVERSION}/{CITY}/{meta[0]}/{CHANNEL}/{meta[1][0]}.npy',output)