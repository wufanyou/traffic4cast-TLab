import h5py
import numpy as np
import os
from fastprogress import master_bar,progress_bar
from tqdm import tqdm
import pandas as pd
import argparse
from utils.functions import getPredictIndex

parser = argparse.ArgumentParser()
parser.add_argument('--city')
parser.add_argument('-i','--in_path',type=str, default='./')
args = parser.parse_args()

path = args.in_path
city = args.city
out_path = './processed/'

# expand h5 data to npy data in order to increase IO-capacity.
meta = []
mb = master_bar(['test','training','validation'])
for set in mb:
    for file in progress_bar(os.listdir(f'{path}/{city}/{city}_{set}'),parent=mb):
        date = file.split('_')[0]
        meta.append((city,set,date))
        f = h5py.File(f'{path}/{city}/{city}_{set}/{file}')
        f = list(f['array'])
        for time in range(288):
            t_f = f[time]
            time = str(time).zfill(3)
            try:
                os.makedirs(f'{out_path}/{city}/{date}/{time}')
            except:
                pass
            for channel in range(3):
            	arr = t_f[:,:,channel]
            	np.save(f'{out_path}/{city}/{date}/{time}/{channel}.npy',arr)

#---------------------------------------------------
# Month and Week Feats
meta = pd.read_csv(f'{out_path}/meta.csv')
meta=meta[meta['set']!='test']
meta=meta[[city,'set']].sort_values(city)
meta['date']=pd.to_datetime(meta[city],format='%Y%m%d')
meta['month']=meta.date.dt.month
meta['weekday']=meta.date.dt.weekday
for channel in range(3):
    for m,df in tqdm(meta.groupby('month')):
        d_dict={}
        for d in df[city]:
            t_arr=[]
            for t in tqdm(range(288)):
                t=str(t).zfill(3)
                t_arr.append(np.load(f'{out_path}/{city}/{d}/{t}/{channel}.npy').reshape(1,495,436))
            t_arr=np.concatenate(t_arr).reshape(1,288,495,436)
            d_dict[d]=t_arr
        a=list(d_dict.values())
        a=np.concatenate(a)
        a=a.mean(0)
        for t in tqdm(range(288)):
            try:
                os.makedirs(f'{out_path}/{city}/Month/{m}/{str(t).zfill(3)}/')
            except:
                pass
            np.save(f'{out_path}/{city}/Month/{m}/{str(t).zfill(3)}/{channel}.npy',a[t])
        for w,ddf in tqdm(df.groupby('weekday')):
            a=[]
            for t in ddf[city]:
                a.append(d_dict[t])
            a = np.concatenate(a)
            a = a.mean(0)
            for t in tqdm(range(288)):
                try:
                    os.makedirs(f'{out_path}/{city}/Week/{m}/{w}/{str(t).zfill(3)}/')
                except:
                    pass
                np.save(f'{out_path}/{city}/Week/{m}/{w}/{str(t).zfill(3)}/{channel}.npy',a[t])

#-------------------------------------------------------
# hour_moving_winodw feats.
meta = pd.read_csv(f'{out_path}/meta.csv')
for d in tqdm(meta[city]):
    try:
        os.makedirs(f'{out_path}/{city}/hour_moving_window/{d}')
    except:
        pass
    w_arr=[]
    for windows in range(288):
        c_arr =[]
        for channel in range(3):
            c_arr.append(np.load(f'{out_path}/{city}/{d}/{str(windows).zfill(3)}/{channel}.npy').reshape(1,495,436))
        c_arr=np.concatenate(c_arr).reshape(1,3,495,436)
        #c_arr=c_arr.reshape(1,495,436)
        w_arr.append(c_arr)
        if len(w_arr)==13:
            np.save(f'{out_path}/{city}/hour_moving_window/{d}/{windows-6}.npy',np.concatenate(w_arr).mean(0))
            w_arr.pop(0)

#------------------------------------------------------
# day_avg_period feats.
index = getPredictIndex(city)
meta = pd.read_csv(f'{out_path}/meta.csv')
meta=meta[[city,'set']]
index = getPredictIndex(city)
for d in tqdm(meta[city]):
    try:
        os.makedirs(f'{out_path}/{city}/day_avg_period/')
    except:
        pass
    d_arr=[]
    for idx in index:
        i_arr = []
        for i in range(-12,0):
            c_arr=[]
            for channel in range(3):
                c_arr.append(np.load(f'{out_path}/{city}/{d}/{str(idx+i).zfill(3)}/{channel}.npy')[None,None])
            c_arr=np.concatenate(c_arr,1)
            i_arr.append(c_arr)
        i_arr = np.concatenate(i_arr,0).mean(0)[None]
        d_arr.append(i_arr)
    d_arr=np.concatenate(d_arr,0)
    np.save(f'{out_path}/{city}/day_avg_period/{d}.npy')