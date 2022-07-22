import datetime
import pandas as pd
from statsforecast.adapters.prophet import AutoARIMAProphet
#from pathlib import Path
import numpy as np
#from multiprocessing import Pool
import itertools
#import os.path
#from fractions import gcd
#import pdb

def daterange(start_date, end_date):
    for n in range(0,int((end_date - start_date).days),15):
        yield start_date + datetime.timedelta(n)

# def compute(ip,dates):#df will not be the input, i,j will be
#     # ip = ip_asc
#     sorted_dates = sorted_datetimes
#     df = {'ds':dates,'y':ip}
#     df = pd.DataFrame(df)
#     df = df[df['y']!=0.0]
#     m =  AutoARIMAProphet()
#     m.fit(df)
#     future = m.make_future_dataframe(periods=1)
#     forecast = m.predict(future)
#     return [forecast['yhat'].values[0]]

if __name__ == "__main__":
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2020-01-01')
    #global end
    df = pd.read_pickle('./namelist/s2_08.pkl')
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    ip = np.load('./namelist/b08.npy')
    #ip = np.swapaxes(ip,0,2)
    ip4532 = ip[:, 45, 32]
    ip2516 = ip[:, 25, 16]
    df4532 = {'ds':df['start_datetime'],'y':ip4532}
    df4532 = pd.DataFrame(df4532)
    df4532 = df4532[df4532['y']!=0.0]
    df2516 = {'ds':df['start_datetime'],'y':ip2516}
    df2516 = pd.DataFrame(df2516)
    df2516 = df2516[df2516['y']!=0.0]
    op4532 = []
    op2516 = []
    for single_date in daterange(start_date, end_date):
        start = single_date
        end = start + datetime.timedelta(days=85)
        #stack_df = df[np.logical_and(df['start_datetime'] >= np.datetime64(start), df['start_datetime'] < np.datetime64(end))]
        df = df4532[np.logical_and(df4532['ds']>=np.datetime64(start),df4532['ds']<np.datetime64(end))]
        m = AutoARIMAProphet()
        m.fit(df)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        op4532.append([end,forecast['yhat'].values[0]])
        df = df2516[np.logical_and(df2516['ds']>=np.datetime64(start),df2516['ds']<np.datetime64(end))]
        m = AutoARIMAProphet()
        m.fit(df)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        op2516.append([end,forecast['yhat'].values[0]])
        #global sorted_datetimes
        # sorted_datetimes = sorted(list(set(
        # stack_df['start_datetime'].values)))
        # sorted_datetimes = pd.DataFrame(sorted_datetimes)
        # sorted_datetimes.columns =['start_time']
        # sorted_datetimes['start_time'] = pd.to_datetime(sorted_datetimes['start_time'])     
    
    np.save('./autoarima/b084532.npy',op4532)
    np.save('./autoarima/b082516.npy',op2516)