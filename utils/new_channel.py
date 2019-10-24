import pandas as pd
import os
import numpy as np
from fastprogress import master_bar,progress_bar
from tqdm import tqdm
import argparse
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/data/data20180901/processed/')
    parser.add_argument('-c', '--city', type=str, default='Berlin')
    parser.add_argument('-ch', '--channel', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = getArgs()
    root = args.input_dir
    city = args.city
    channel = args.channel
    meta = pd.read_csv(f'{root}/meta.csv')
    meta=meta[meta['set']!='test']
    meta=meta[[city,'set']].sort_values(city)
    meta['date']=pd.to_datetime(meta[city],format='%Y%m%d')
    meta['month']=meta.date.dt.month
    meta['weekday']=meta.date.dt.weekday
    
    for m,df in tqdm(meta.groupby('month')):
        d_dict={}
        for d in df[city]:
            t_arr=[]
            for t in tqdm(range(288)):
                t=str(t).zfill(3)
                t_arr.append(np.load(f'{root}/{city}/{d}/{t}/{channel}.npy').reshape(1,495,436))
            t_arr=np.concatenate(t_arr).reshape(1,288,495,436)
            d_dict[d]=t_arr
        a=list(d_dict.values())
        a=np.concatenate(a)
        a=a.mean(0)
        for t in tqdm(range(288)):
            try:
                os.makedirs(f'{root}/{city}/Month/{m}/{str(t).zfill(3)}/')
            except:
                pass
            np.save(f'{root}/{city}/Month/{m}/{str(t).zfill(3)}/{channel}.npy',a[t])
        for w,ddf in tqdm(df.groupby('weekday')):
            a=[]
            for t in ddf[city]:
                a.append(d_dict[t])
            a = np.concatenate(a)
            a = a.mean(0)
            for t in tqdm(range(288)):
                try:
                    os.makedirs(f'{root}/{city}/Week/{m}/{w}/{str(t).zfill(3)}/')
                except:
                    pass
                np.save(f'{root}/{city}/Week/{m}/{w}/{str(t).zfill(3)}/{channel}.npy',a[t])