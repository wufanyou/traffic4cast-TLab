from utils import *
from utils import DatasetFolderV12 as DatasetFolder
import numpy as np
from fastprogress import master_bar,progress_bar
import time
import h5py
import os
import argparse
   
def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='./processed/')
    parser.add_argument('-o', '--output_dir', type=str, default='./')
    parser.add_argument('-m', '--model_dir', type=str, default='./')
    parser.add_argument('-c', '--city', type=str, default='Berlin')
    #parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--step', type=int, default=12)
    parser.add_argument('--version', type=str, default='v17')
    #parser.add_argument('-nl','--no_leak',action='store_false')
    #parser.add_argument('--activation',type=str,default='relu')
    args = parser.parse_args()
    return args

#IN_PATH = '/data/data20180901/processed/'
#OUT_PATH = './'
#CITY = 'Berlin'
#DEVICE = 'cuda:0'
#STEP = 3
#VERSION = '0'

args = getArgs()
IN_PATH = args.input_dir
OUT_PATH = args.output_dir
MODEL_PATH = args.model_dir
CITY = args.city
#DEVICE = f'cuda:{args.device}'
STEP = args.step
VERSION = args.version
#IS_LEAK = args.no_leak
#LEAK_STEP = 18 if IS_LEAK else 0
#ACTIVATION = args.activation
VERSION_MAP={
    'Moscow':{0:'v13',1:VERSION,2:VERSION},
    'Berlin':{0:'v13',1:VERSION,2:VERSION},
    'Istanbul':{0:'v13',1:VERSION,2:VERSION},
}
if __name__=='__main__':
    index = getPredictIndex(CITY)
    #index = [i+j for i in index for j in range(3)]
    print(index)
    folder = DatasetFolder(IN_PATH,CITY,'test',index,STEP,0,is_transform=False,predict_length=1,skip=0)    
    for DATE in folder.meta:
        d_arr=[]
        #CHANNEL = 0
        for CHANNEL in [0,1,2]: 
            arr = []
            for ids in index:
                arr.append(np.load(f'{OUT_PATH}/result/numpy/{VERSION_MAP[CITY][CHANNEL]}/{CITY}/{DATE}/{CHANNEL}/{ids}.npy')[None,:])
            arr = np.concatenate(arr)
            #print(arr.shape)
            d_arr.append(arr)
            
        """
        for CHANNEL in [2]:
            arr = []
            for ids in index:
                t_arr=[]
                for i in range(3):
                    t_arr.append(np.load(f'{OUT_PATH}/result/numpy/{VERSION_MAP[CITY][CHANNEL]}/{CITY}/{DATE}/{CHANNEL}/{ids+i}.npy')[None,:])
                t_arr = np.concatenate(t_arr,1)
                arr.append(t_arr)
            arr = np.concatenate(arr)
            #print(arr.shape)
            d_arr.append(arr)
        """
        
        d_arr = np.concatenate(d_arr,-1)
        #print(d_arr.shape)
        try:
            os.makedirs(f'{OUT_PATH}/result/output/{VERSION}/{CITY}/{CITY}_test/')
        except:
            pass
        filename = f'{OUT_PATH}/result/output/{VERSION}/{CITY}/{CITY}_test/{DATE}_100m_bins.h5' 
        write_data(d_arr,filename)
