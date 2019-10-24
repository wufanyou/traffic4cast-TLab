import h5py
import numpy as np
import os
from fastprogress import master_bar,progress_bar
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--city')
args = parser.parse_args()
import pandas as pd

#valueMap = np.vectorize(lambda x:{0:0,1:1,85:2,170:3,255:4}[x])

path = '/data/data20180901/data'
out_path = '/data/data20180901/processed/'
channel = 2
meta = []
city= args.city
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
            #for channel in range(3):
            arr = t_f[:,:,channel]
                #if channel == 2:
            #arr = valueMap(arr)
            #arr = arr.astype(np.uint8)
            np.save(f'{out_path}/{city}/{date}/{time}/{channel}.npy',arr)
            
meta=pd.DataFrame(meta)
meta.columns=['city','set','date']
meta.to_csv(f'{out_path}/{city}_meta.csv',index=False)