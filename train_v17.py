from utils import *
from utils import DatasetFolderV17 as DatasetFolder
import numpy as np
from fastprogress import master_bar,progress_bar
import time
#import h5py
import os
import sys
import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='./processed/')
    parser.add_argument('-o', '--output_dir', type=str, default='/data/data20180901/ly_v17/')
    parser.add_argument('-c', '--city', type=str, default='Moscow')
    parser.add_argument('-ch', '--channel', type=int, default=0)
    parser.add_argument('-w','--windows',type=int,default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--init_lr', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='0,1,2,3')
    parser.add_argument('--step', type=int, default=12)
    parser.add_argument('--version', type=str, default=0)
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--in_suffix', type=str, default='best')
    parser.add_argument('--out_suffix', type=str, default='')
    args = parser.parse_args()
    return args

#setting
args = getArgs()
IN_PATH = args.input_dir
OUT_PATH = args.output_dir
CITY = args.city
EPOCHS = args.epochs 
BATCH_SIZE = args.batch_size
INIT_LR = args.init_lr
CHANNEL = args.channel
STEP = args.step
VERSION = sys.argv[0].split('_')[1].split('.')[0] if args.version==0 else args.version
WINDOWS = args.windows
LEAK_STEP = 65 #if IS_LEAK else 0
DISTRIBUTED_TRAINING=True
BATCH_SIZE = BATCH_SIZE*len(args.device.split(',')) if DISTRIBUTED_TRAINING else BATCH_SIZE
CROP = args.crop
CROP_SIZE = [(0,299,0,299),(0,299,137,None),(196,None,137,None),(196,None,0,299),(98,397,68,367)][CROP]
IN_SUFFIX = args.in_suffix
OUT_SUFFIX = args.out_suffix

os.environ["CUDA_VISIBLE_DEVICES"]=args.device
import torch
import torch.nn as nn
import torch.nn.functional as F
        
def getPredictIndex(city):
    if city == 'Berlin':
        index = [30, 69, 126, 186, 234]
    elif city in ['Istanbul','Moscow']:
        index = [57, 114, 174,222, 258]
    return index

if __name__=="__main__":
    valid_index = getPredictIndex(CITY)
    #valid_index = [i+j for i in valid_index for j in range(3)]
    #train_index = [x+j for x in valid_index for j in range(-1*STEP,STEP+1)]
    train_index = set([x+j for x in valid_index for j in range(-1*WINDOWS,WINDOWS+1)])
    model = UNet(STEP*3+LEAK_STEP,3,20)
    #try:
    try:
        model.load_state_dict(torch.load(f'{OUT_PATH}/{VERSION}_{CITY}_{CHANNEL}_{CROP}_best.pth'))
        print('load best')
    except:
        try:
            pass
            print ('load raw')
        except:
            print('load fail')
    #model.load_state_dict(torch.load(f'{OUT_PATH}/{CITY}_raw_{CHANNEL}.pth'))    
    #except:
        #print('load fail')
        
    if DISTRIBUTED_TRAINING:
        model=nn.DataParallel(model)
        model=model.cuda()
    else:
        model = model.cuda()
        
    train_folder = DatasetFolder(IN_PATH,CITY,'train',train_index,STEP,CHANNEL,3,skip=0,is_transform=True,crop=CROP_SIZE)
    valid_folder = DatasetFolder(IN_PATH,CITY,'val',valid_index,STEP,CHANNEL,3,skip=0,is_transform=True,crop=CROP_SIZE)
    train=torch.utils.data.DataLoader(train_folder,batch_size=BATCH_SIZE,shuffle=True,num_workers=12)
    valid=torch.utils.data.DataLoader(valid_folder,batch_size=BATCH_SIZE,shuffle=True,num_workers=12)
    lr = INIT_LR
    #loss = torch.nn.MSELoss(reduction='mean')
    loss = torch.nn.MSELoss(reduction='mean')
    mb=master_bar(range(EPOCHS))
    best_loss = float('inf')
    valid_loss = float('inf')
    
    count=0
    sum_loss=0
    model=model.eval()
    for X,Y in progress_bar(valid,parent=mb, txt_len=100):
        X = X.cuda()#*2/255-1
            #E = E.cuda()
        Y = Y.cuda()#*2/255-1
        with torch.set_grad_enabled(False):
            out = model(X,None)
            l = loss(out,Y)*X.size()[0]
            count+=X.size()[0]
            sum_loss+=l
    best_loss = (sum_loss/count)/4
    valid_loss = best_loss
    
    for e in mb:
        optimizer=torch.optim.SGD(model.parameters(),lr=lr)
        lr = 0.99*lr
        model=model.train()
        for X,Y in progress_bar(train,parent=mb, txt_len=100):
            X = X.cuda()#*2/255-1
            #E = E.cuda()
            Y = Y.cuda()#*2/255-1
            optimizer.zero_grad()
            out = model(X,None)
            l = loss(out,Y)
            l.backward()
            optimizer.step()
            mb.child.comment = f'[{CHANNEL}][{best_loss:.7f}][{valid_loss:.7f}][{l/4:.7f}]'
            
        count=0
        sum_loss=0
        model=model.eval()
        for X,Y in progress_bar(valid,parent=mb, txt_len=100):
            X = X.cuda()#*2/255-1
            #E = E.cuda()
            Y = Y.cuda()#*2/255-1
            with torch.set_grad_enabled(False):
                out = model(X,None)
                l = loss(out,Y)*X.size()[0]
                count+=X.size()[0]
                sum_loss+=l
        valid_loss=(sum_loss/count)/4
        mb.write(f'[{e}][{STEP}] valid_loss {valid_loss:.7f}')
        
        model=model.train()
        if valid_loss<=best_loss:
            if DISTRIBUTED_TRAINING:
                torch.save(model.module.state_dict(),f'{OUT_PATH}/{VERSION}_{CITY}_{CHANNEL}_{CROP}{OUT_SUFFIX}_best.pth')
            else:
                torch.save(model.state_dict(),f'{OUT_PATH}/{VERSION}_{CITY}_{CHANNEL}_{CROP}{OUT_SUFFIX}_best.pth')
            best_loss=valid_loss
    torch.save(model.module.state_dict(),f'{OUT_PATH}/{VERSION}_{CITY}_{CHANNEL}_{CROP}{OUT_SUFFIX}_final.pth')