import torch
import datetime
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import torch.nn.functional as F


def dateTruncate(date):
    #20180403
    date= min(date,pd.to_datetime('20181231',format='%Y%m%d'))
    date= max(date,pd.to_datetime('20180101',format='%Y%m%d'))
    if int(datetimeToDate(date))==20180403:
        date = pd.to_datetime('20180404',format='%Y%m%d')
    return date

def dateToMonthWeek(date):
    date = pd.to_datetime(date,format='%Y%m%d')
    month = date.month
    weekday = date.weekday()
    return month,weekday

def datetimeToDate(date):
    return int(date.strftime('%Y%m%d'))

def timeIndexShift(date,index,hours):
    new_date = dateTruncate(pd.to_datetime(str(date),format='%Y%m%d')+pd.Timedelta(minutes=index*5))
    new_date = new_date+pd.Timedelta(hours=hours)
    new_index=(new_date.hour*60+new_date.minute)//5
    new_index = max(6,new_index)
    new_index = min(281,new_index)
    new_date = datetimeToDate(dateTruncate(new_date))
    return new_date,new_index



def toTensor(x,dtype=torch.float32,transform=True):
    x = torch.tensor(x,dtype=dtype)
    x = x*2/255-1 if transform else x
    return x

def read_fileV6(path,channel,indexs,is_leak=True):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
    else:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{j}.npy") for index in indexs for j in range(3)]
        
    arr_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1]
    arr_1 = np.concatenate(arr_1)
    
    if not is_leak:
        arr = arr_1
    else:
        # D-14,D-7,D-3,...,D+3,D+7,D+14
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')#-pd.Timedelta(days=7)
        dates = [datetimeToDate(dateTruncate(date+pd.Timedelta(days=d))) for d in [14,-7,-5,-6,-4,-3,-2,-1,1,2,3,4,5,6,7,14]]
        arr_2 = [np.load(f"{path}/{d}/{str(indexs[-1]).zfill(3)}/{channel}.npy") for d in dates]
        arr_2 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)
                     
        arr = np.concatenate([arr_1,arr_2,arr_3])
    return arr      
class DatasetFolderV6(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=12,predict=0,predict_length=3,is_leak=True,is_transform=True):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step,i)) for i in label_time_index]
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.max_day = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        self.is_leak = is_leak
        self.is_transform =is_transform
        data = pd.read_csv(f'{root}/{city}_weather_p.csv',index_col=0)
        data['weather']=data.weather.astype('category').cat.codes
        data['wind_direnction_from']=data['wind_direnction_from'].astype('category').cat.codes
        data['wind_direnction_to']=data['wind_direnction_to'].astype('category').cat.codes
        self.weather = data
        data = pd.read_csv(f'{root}/holidays.csv')
        data = data[data['City']==self.city]
        self.holiday = set(data['Date'])
        
    def __getitem__(self, index):

        X = toTensor(read_fileV6(f"{self.root}/{self.city}/",self.predict,self.X[index],True),torch.float32,self.is_transform)
        date = self.X[index][0]
        month = int(str(date)[4:6])
        day = int(str(date)[6:])
        normal_day = day/self.max_day[month]
        sin_normal_day = np.sin(normal_day*np.pi)
        cos_normal_day = np.cos(normal_day*np.pi)
        
        time = self.Y[index][1][0]
        sin_time=np.sin(time/287*np.pi)
        cos_time=np.cos(time/287*np.pi)
        
        weekday= datetime.date(2018,month,day).weekday()
        sin_weekday=np.sin(weekday/6*np.pi)
        cos_weekday=np.cos(weekday/6*np.pi)
        
        holiday = 1 if date in self.holiday else 0
        
        time_zone = 2 if self.city=='Berlin' else 3
        new_date = dateTruncate(pd.to_datetime(str(date),format='%Y%m%d')+pd.Timedelta(minutes=self.X[index][0]*5+time_zone*60))
        data = self.weather[self.weather.date==date]
        if self.city!='Moscow':
            data = data[data.hour==new_date.hour].reset_index(drop=True)
            #print(data)
            data['min'] = data['min']-20
            data=data[data['min']==((new_date.minute//30)*30)]
        else:
            hour=(new_date.hour//3)*3
            data = data[data.hour==hour]
        
        E = torch.tensor([month,day,sin_normal_day,cos_normal_day,time,sin_time,cos_time,weekday,sin_weekday,cos_weekday,holiday],dtype=torch.float32)
        E2 = torch.tensor((data.drop(['date','hour','min'],axis=1)).to_numpy().reshape(-1),dtype=torch.float32)
        if len(E2)==0:
            E2 = torch.zeros([9],dtype=torch.float32)
            #print(new_date)
        E = torch.cat([E,E2])
        # LABEL
        Y = toTensor(read_fileV6(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        return (X,E,Y)
    def __len__(self): 
        return len(self.X)
    
def read_fileV10(path,channel,indexs,is_leak=True):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
    else:
        arr_1 = []
        for i in range(3):
            arr = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{i}.npy") for index in indexs]
            arr = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr]
            arr = np.concatenate(arr)
            arr = arr.reshape(1,arr.shape[0],arr.shape[1],arr.shape[2])
            arr_1.append(arr.copy())
        arr_1 = np.concatenate(arr_1)
    arr_2 = None
    
    if is_leak:
        # D-14,D-7,D-3,...,D+3,D+7,D+14
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')#-pd.Timedelta(days=7)
        dates = [datetimeToDate(dateTruncate(date+pd.Timedelta(days=d))) for d in [14,-7,-5,-6,-4,-3,-2,-1,1,2,3,4,5,6,7,14]]
        arr_2 = [np.load(f"{path}/{d}/{str(indexs[-1]).zfill(3)}/{channel}.npy") for d in dates]
        arr_2 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)        
        arr_2 = np.concatenate([arr_2,arr_3])
        
    return arr_1,arr_2      


class DatasetFolderV12(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=12,predict=0,predict_length=1,is_leak=True,is_transform=True,skip=0):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.max_day = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        self.is_leak = is_leak
        self.is_transform =is_transform
        data = pd.read_csv(f'{root}/{city}_weather_p.csv',index_col=0)
        data['weather']=data.weather.astype('category').cat.codes
        data['wind_direnction_from']=data['wind_direnction_from'].astype('category').cat.codes
        data['wind_direnction_to']=data['wind_direnction_to'].astype('category').cat.codes
        self.weather = data
        data = pd.read_csv(f'{root}/holidays.csv')
        data = data[data['City']==self.city]
        self.holiday = set(data['Date'])
        
    def __getitem__(self, index):

        X = toTensor(read_fileV6(f"{self.root}/{self.city}/",self.predict,self.X[index],True),torch.float32,self.is_transform)
        date = self.X[index][0]
        month = int(str(date)[4:6])
        day = int(str(date)[6:])
        normal_day = day/self.max_day[month]
        sin_normal_day = np.sin(normal_day*np.pi)
        cos_normal_day = np.cos(normal_day*np.pi)
        
        time = self.Y[index][1][0]
        sin_time=np.sin(time/287*np.pi)
        cos_time=np.cos(time/287*np.pi)
        
        weekday= datetime.date(2018,month,day).weekday()
        sin_weekday=np.sin(weekday/6*np.pi)
        cos_weekday=np.cos(weekday/6*np.pi)
        
        holiday = 1 if date in self.holiday else 0
        
        time_zone = 2 if self.city=='Berlin' else 3
        new_date = dateTruncate(pd.to_datetime(str(date),format='%Y%m%d')+pd.Timedelta(minutes=self.X[index][0]*5+time_zone*60))
        data = self.weather[self.weather.date==date]
        if self.city!='Moscow':
            data = data[data.hour==new_date.hour].reset_index(drop=True)
            #print(data)
            data['min'] = data['min']-20
            data=data[data['min']==((new_date.minute//30)*30)]
        else:
            hour=(new_date.hour//3)*3
            data = data[data.hour==hour]
        
        E = torch.tensor([month,day,sin_normal_day,cos_normal_day,time,sin_time,cos_time,weekday,sin_weekday,cos_weekday,holiday],dtype=torch.float32)
        E2 = torch.tensor((data.drop(['date','hour','min'],axis=1)).to_numpy().reshape(-1),dtype=torch.float32)
        if len(E2)==0:
            E2 = torch.zeros([9],dtype=torch.float32)
            #print(new_date)
        E = torch.cat([E,E2])
        # LABEL
        Y = toTensor(read_fileV6(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        return (X,E,Y)
    def __len__(self): 
        return len(self.X)
    
def read_fileV13(path,channel,indexs,is_leak=True):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
    else:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{j}.npy") for index in indexs for j in range(3)]
        
    arr_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1]
    arr_1 = np.concatenate(arr_1)
    
    if not is_leak:
        arr = arr_1
    else:
        # D-14,D-7,D-3,...,D+3,D+7,D+14
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')#-pd.Timedelta(days=7)
        dates = [datetimeToDate(dateTruncate(date+pd.Timedelta(days=d))) for d in [14,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,14]]
        arr_2 = [np.load(f"{path}/{d}/{str(indexs[-1]).zfill(3)}/{channel}.npy") for d in dates]
        arr_2 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)
                     
        arr = np.concatenate([arr_1,arr_2,arr_3])
    return arr

class DatasetFolderV13(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=12,predict=0,predict_length=1,is_leak=True,is_transform=True,skip=0,crop=(0,299,0,299)):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        self.crop = crop
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.max_day = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        self.is_leak = is_leak
        self.is_transform =is_transform
        data = pd.read_csv(f'{root}/{city}_weather_p.csv',index_col=0)
        data['weather']=data.weather.astype('category').cat.codes
        data['wind_direnction_from']=data['wind_direnction_from'].astype('category').cat.codes
        data['wind_direnction_to']=data['wind_direnction_to'].astype('category').cat.codes
        self.weather = data
        data = pd.read_csv(f'{root}/holidays.csv')
        data = data[data['City']==self.city]
        self.holiday = set(data['Date'])
        
    def __getitem__(self, index):

        X = toTensor(read_fileV13(f"{self.root}/{self.city}/",self.predict,self.X[index],True),torch.float32,self.is_transform)
        X = X[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        date = self.X[index][0]
        month = int(str(date)[4:6])
        day = int(str(date)[6:])
        normal_day = day/self.max_day[month]
        sin_normal_day = np.sin(normal_day*np.pi)
        cos_normal_day = np.cos(normal_day*np.pi)
        
        time = self.Y[index][1][0]
        sin_time=np.sin(time/287*np.pi)
        cos_time=np.cos(time/287*np.pi)
        
        weekday= datetime.date(2018,month,day).weekday()
        sin_weekday=np.sin(weekday/6*np.pi)
        cos_weekday=np.cos(weekday/6*np.pi)
        
        holiday = 1 if date in self.holiday else 0
        
        time_zone = 2 if self.city=='Berlin' else 3
        new_date = dateTruncate(pd.to_datetime(str(date),format='%Y%m%d')+pd.Timedelta(minutes=self.X[index][0]*5+time_zone*60))
        data = self.weather[self.weather.date==date]
        if self.city!='Moscow':
            data = data[data.hour==new_date.hour].reset_index(drop=True)
            #print(data)
            data['min'] = data['min']-20
            data=data[data['min']==((new_date.minute//30)*30)]
        else:
            hour=(new_date.hour//3)*3
            data = data[data.hour==hour]
        
        E = torch.tensor([month,day,sin_normal_day,cos_normal_day,time,sin_time,cos_time,weekday,sin_weekday,cos_weekday,holiday],dtype=torch.float32)
        E2 = torch.tensor((data.drop(['date','hour','min'],axis=1)).to_numpy().reshape(-1),dtype=torch.float32)
        if len(E2)==0:
            E2 = torch.zeros([9],dtype=torch.float32)
            #print(new_date)
        E = torch.cat([E,E2])
        # LABEL
        Y = toTensor(read_fileV13(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        Y = Y[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        return (X,E,Y)
    def __len__(self): 
        return len(self.X)
    
def read_fileV15(path,channel,indexs,is_leak=True):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
    else:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{j}.npy") for index in indexs for j in range(3)]
        
    arr_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1]
    arr_1 = np.concatenate(arr_1)
    
    if not is_leak:
        arr = arr_1
    else:
        # D-14,D-7,D-3,...,D+3,D+7,D+14
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')#-pd.Timedelta(days=7)
        dates = [datetimeToDate(dateTruncate(date+pd.Timedelta(days=d))) for d in [-14,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,14]]
        arr_2 = [np.load(f"{path}/{d}/{str(indexs[-1]).zfill(3)}/{channel}.npy") for d in dates]
        arr_2 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)
        
        arr_4 = np.load(f"{path}/day_avg_period/{raw_date}.npy")
        arr_4 = arr_4.reshape(-1,arr_4.shape[2],arr_4.shape[3])
        
        arr = np.concatenate([arr_1,arr_2,arr_3,arr_4])
    return arr

class DatasetFolderV15(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=12,predict=0,predict_length=3,is_leak=True,is_transform=True,skip=0,crop=(0,299,0,299)):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        self.crop = crop
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.is_leak = is_leak
        self.is_transform =is_transform
        
    def __getitem__(self, index):

        X = toTensor(read_fileV15(f"{self.root}/{self.city}/",self.predict,self.X[index],True),torch.float32,self.is_transform)
        X = X[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        
        # LABEL
        Y = toTensor(read_fileV15(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        Y = Y[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        return (X,Y)
    def __len__(self): 
        return len(self.X)
    
    
class DatasetFolderOL(object):
    def __init__(self,date, root,city="Berlin",label_time_index=range(3,287),step=12,predict=0,predict_length=3,is_leak=True,is_transform=True,skip=0,crop=(0,299,0,299)):
        self.root = root
        self.city = city
        self.date = date
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        self.crop = crop
        #meta = pd.read_csv(f'{root}/meta.csv')
        #self.meta = meta[meta.set==set_type][city].tolist()
        self.meta =[date] if type(date)==int else date
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.is_leak = is_leak
        self.is_transform = is_transform
        
    def __getitem__(self, index):

        X = toTensor(read_fileV17(f"{self.root}/{self.city}/",self.predict,self.X[index],True),torch.float32,self.is_transform)
        X = X[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        
        # LABEL
        Y = toTensor(read_fileV17(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        Y = Y[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        return (X,Y)
    def __len__(self): 
        return len(self.X)
    
def read_fileV16(path,channel,indexs,is_leak=True):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
    else:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{j}.npy") for index in indexs for j in range(3)]
        
    arr_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1]
    arr_1 = np.concatenate(arr_1)
    
    if not is_leak:
        arr = arr_1
    else:
        # D-14,D-7,D-3,...,D+3,D+7,D+14
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')#-pd.Timedelta(days=7)
        dates = [datetimeToDate(dateTruncate(date+pd.Timedelta(days=d))) for d in [-14,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,14]]
        arr_2 = [np.load(f"{path}/hour_avg/{d}/{indexs[-1]//12}.npy") for d in dates]
        arr_2 = [x.reshape(-1,x.shape[1],x.shape[2]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)
        
        arr_4 = np.load(f"{path}/day_avg_period/{raw_date}.npy")
        arr_4 = arr_4.reshape(-1,arr_4.shape[2],arr_4.shape[3])
        
        arr = np.concatenate([arr_1,arr_2,arr_3,arr_4])
    return arr

class DatasetFolderV16(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=12,predict=0,predict_length=3,is_leak=True,is_transform=True,skip=0,crop=(0,299,0,299)):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        self.crop = crop
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.is_leak = is_leak
        self.is_transform =is_transform
        
    def __getitem__(self, index):

        X = toTensor(read_fileV16(f"{self.root}/{self.city}/",self.predict,self.X[index],True),torch.float32,self.is_transform)
        X = X[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        
        # LABEL
        Y = toTensor(read_fileV16(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        Y = Y[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        return (X,Y)
    def __len__(self): 
        return len(self.X)
    
def read_fileV17(path,channel,indexs,is_leak=True,skip=0):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
    else:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{j}.npy") for index in indexs for j in range(3)]
        
    arr_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1]
    arr_1 = np.concatenate(arr_1)
    
    if not is_leak:
        arr = arr_1
    else:
        # D-14,D-7,D-3,...,D+3,D+7,D+14
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')#-pd.Timedelta(days=7)
        dates = [datetimeToDate(dateTruncate(date+pd.Timedelta(days=d))) for d in [14,7,6,5,4,3,2,1,-1,-2,-3,-4,-5,-6,-7,-14]]
        arr_2 = [np.load(f"{path}/hour_moving_window/{d}/{indexs[-1]+1+skip}.npy") for d in dates]
        arr_2 = [x.reshape(-1,x.shape[1],x.shape[2]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)
        
        arr_4 = np.load(f"{path}/day_avg_period/{raw_date}.npy")
        arr_4 = arr_4.reshape(-1,arr_4.shape[2],arr_4.shape[3])
        
        arr = np.concatenate([arr_1,arr_2,arr_3,arr_4])
    return arr

class DatasetFolderV17(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=12,predict=0,predict_length=3,is_leak=True,is_transform=True,skip=0,crop=(0,299,0,299)):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        self.crop = crop
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.is_leak = is_leak
        self.is_transform =is_transform
        
    def __getitem__(self, index):

        X = toTensor(read_fileV17(f"{self.root}/{self.city}/",self.predict,self.X[index],True,self.skip),torch.float32,self.is_transform)
        X = X[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        
        # LABEL
        Y = toTensor(read_fileV17(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        Y = Y[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        return (X,Y)
    def __len__(self): 
        return len(self.X)

def read_fileV18(path,channel,indexs,is_leak=True,skip=0):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
    else:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{j}.npy") for index in indexs for j in range(3)]
        
    arr_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1]
    arr_1 = np.concatenate(arr_1)
    
    if not is_leak:
        arr = arr_1
    else:
        # hour 
        infos = [timeIndexShift(raw_date,indexs[-1]+1+skip,hour) for hour in [-336, -168, -73, -49, -25, -72, -48,-24, 24,48,72,25,49,73,168,336]]
        arr_2 = [np.load(f"{path}/hour_moving_window/{x[0]}/{x[1]}.npy") for x in infos]
        arr_2 = [x.reshape(-1,x.shape[1],x.shape[2]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)
        
        arr_4 = np.load(f"{path}/day_avg_period/{raw_date}.npy")
        arr_4 = arr_4.reshape(-1,arr_4.shape[2],arr_4.shape[3])
        
        arr = np.concatenate([arr_1,arr_2,arr_3,arr_4])
    return arr

class DatasetFolderV18(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=12,predict=0,predict_length=3,is_leak=True,is_transform=True,skip=0,crop=(0,299,0,299)):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        self.crop = crop
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.is_leak = is_leak
        self.is_transform =is_transform
        
    def __getitem__(self, index):

        X = toTensor(read_fileV18(f"{self.root}/{self.city}/",self.predict,self.X[index],True,self.skip),torch.float32,self.is_transform)
        X = X[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        
        # LABEL
        Y = toTensor(read_fileV18(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        Y = Y[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        return (X,Y)
    def __len__(self): 
        return len(self.X)
    
    
def read_fileV19(path,channel,indexs,is_leak=True,skip=0):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
        arr_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1]
        arr_1 = np.concatenate(arr_1)
    else:
        arr_1_0 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/0.npy") for index in indexs]
        arr_1_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/1.npy") for index in indexs]
        arr_1_2 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/2.npy") for index in indexs]
        
        arr_1_0 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1_0]
        arr_1_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1_1]
        arr_1_2 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1_2]
        arr_1_0 = np.concatenate(arr_1_0)
        arr_1_1 = np.concatenate(arr_1_1)#加两个通道
        arr_1_2 = np.concatenate(arr_1_2)
        #arr_avg_0=arr_1_0.mean(0).reshape(1,495, 436)
        arr_avg_1=arr_1_1.mean(0).reshape(1,495, 436)
        arr_avg_2=arr_1_2.mean(0).reshape(1,495, 436)
        
        arr_1 = np.concatenate([arr_1_0,arr_1_1,arr_1_2,arr_avg_1,arr_avg_2])#
        
    
    if not is_leak:
        arr = arr_1
    else:
        # D-14,D-7,D-3,...,D+3,D+7,D+14
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')#-pd.Timedelta(days=7)
        dates = [datetimeToDate(dateTruncate(date+pd.Timedelta(days=d))) for d in [-14,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,14]]#加两个通道
        arr_2 = [np.load(f"{path}/hour_moving_window/{d}/{indexs[-1]+1+skip}.npy") for d in dates]
        arr_2 = [x.reshape(-1,x.shape[1],x.shape[2]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/0.npy"))##加4个通道
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/0.npy"))
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/1.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/1.npy"))
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/2.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/2.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)
        
        arr_4 = np.load(f"{path}/day_avg_period/{raw_date}.npy")
        arr_4 = arr_4.reshape(-1,arr_4.shape[2],arr_4.shape[3])
        
        arr = np.concatenate([arr_1,arr_2,arr_3,arr_4])
    return arr

class DatasetFolderV19(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=10,predict=0,predict_length=3,is_leak=True,is_transform=True,skip=0,crop=(0,299,0,299)):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        self.crop = crop
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]##step 10 释放6个通道
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.is_leak = is_leak
        self.is_transform =is_transform
        
    def __getitem__(self, index):

        X = toTensor(read_fileV19(f"{self.root}/{self.city}/",self.predict,self.X[index],True,self.skip),torch.float32,self.is_transform)
        X = X[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        
        # LABEL
        Y = toTensor(read_fileV19(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        Y = Y[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        return (X,Y)
    def __len__(self): 
        return len(self.X)


def read_fileV20(path,channel,indexs,is_leak=True,skip=0):
    raw_date,indexs = indexs
    
    #T-N,...,T-1
    if not is_leak:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{channel}.npy") for index in indexs]
    else:
        arr_1 = [ np.load(f"{path}/{raw_date}/{str(index).zfill(3)}/{j}.npy") for index in indexs for j in range(3)]
        
    arr_1 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_1]
    arr_1 = np.concatenate(arr_1)
    
    if not is_leak:
        arr = arr_1
    else:
        # D-14,D-7,D-3,...,D+3,D+7,D+14
        date = pd.to_datetime(str(raw_date),format='%Y%m%d')#-pd.Timedelta(days=7)
        dates = [datetimeToDate(dateTruncate(date+pd.Timedelta(days=d))) for d in [-21,-14,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,14,21]]
        arr_2 = [np.load(f"{path}/hour_moving_window/{d}/{indexs[-1]+1+skip}.npy") for d in dates]
        arr_2 = [x.reshape(-1,x.shape[1],x.shape[2]) for x in arr_2]
        arr_2 = np.concatenate(arr_2)
        
        #Month & Week
        arr_3 = []
        arr_3.append(np.load(f"{path}/Month/{date.month}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3.append(np.load(f"{path}/Week/{date.month}/{date.weekday()}/{str(indexs[-1]).zfill(3)}/{channel}.npy"))
        arr_3 = [x.reshape(1,x.shape[0],x.shape[1]) for x in arr_3]
        arr_3 = np.concatenate(arr_3)
        
        arr_4 = np.load(f"{path}/day_avg_period/{raw_date}.npy")
        arr_4 = arr_4.reshape(-1,arr_4.shape[2],arr_4.shape[3])
        
        arr = np.concatenate([arr_1,arr_2,arr_3,arr_4])
    return arr
class DatasetFolderV20(object):
    def __init__(self, root,city="Berlin",set_type="train",label_time_index=range(3,287),step=10,predict=0,predict_length=3,is_leak=True,is_transform=True,skip=0,crop=(0,299,0,299)):
        self.root = root
        self.city = city
        self.set_type = set_type
        self.label_time_index = label_time_index
        self.step = step
        self.skip = skip
        self.crop = crop
        meta = pd.read_csv(f'{root}/meta.csv')
        self.meta = meta[meta.set==set_type][city].tolist()
        self.predict = predict
        X = [list(range(i-self.step-self.skip,i-self.skip)) for i in label_time_index]##step 10 释放6个通道
        self.X = [(i,j) for i in self.meta for j in X]
        Y = [list(range(i,i+predict_length)) for i in label_time_index]
        self.Y = [(i,j) for i in self.meta for j in Y]
        self.is_leak = is_leak
        self.is_transform =is_transform
        
    def __getitem__(self, index):

        X = toTensor(read_fileV20(f"{self.root}/{self.city}/",self.predict,self.X[index],True,self.skip),torch.float32,self.is_transform)
        X = X[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        
        # LABEL
        Y = toTensor(read_fileV20(f"{self.root}/{self.city}/",self.predict,self.Y[index],False),torch.float32,self.is_transform)
        Y = Y[:,self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]]
        return (X,Y)
    def __len__(self): 
        return len(self.X)