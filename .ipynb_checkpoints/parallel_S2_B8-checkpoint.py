import datetime
import pandas as pd
from statsforecast.adapters.prophet import AutoARIMAProphet
from pathlib import Path
from mantle_utils.alg.raster import Raster
from rasterio.windows import Window
import numpy as np
from multiprocessing import Pool
import itertools
import os.path
#from fractions import gcd
#import pdb

def daterange(start_date, end_date):
    for n in range(0,int((end_date - start_date).days),15):
        yield start_date + datetime.timedelta(n)

def compute(idx0,idx1):#df will not be the input, i,j will be
    # ip = ip_asc
    sorted_dates = sorted_datetimes
    df = {'ds':sorted_dates['start_time'],'y':ip[:,idx0,idx1]}
    df = pd.DataFrame(df)
    df = df[df['y']!=0.0]
    m =  AutoARIMAProphet()
    m.fit(df)
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)
    # if (((idx0+5)%10==0)&(idx1%16==0)):
    #     save_path_t = './s2b08/trends/'
    #     save_path_c = './s2b08/comp/'
    #     fig1 = m.plot(forecast)
    #     fig2 = m.plot_components(forecast)
    #     fig1.savefig(os.path.join(save_path_t,'%s_%s_%s.svg'%(str(end.date()),str(idx0),str(idx1))))
    #     fig2.savefig(os.path.join(save_path_c,'%s_%s_%s.svg'%(str(end.date()),str(idx0),str(idx1))))
    
    return [forecast['yhat'].values[0],idx0,idx1]

if __name__ == "__main__":
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2020-01-01')
    global end
    df = pd.read_pickle('./namelist/s2_08.pkl')
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    for single_date in daterange(start_date, end_date):
        start = single_date
        end = start + datetime.timedelta(days=85)
        stack_df = df[np.logical_and(df['start_datetime'] >= np.datetime64(start), df['start_datetime'] < np.datetime64(end))]
        global sorted_datetimes
        sorted_datetimes = sorted(list(set(
        stack_df['start_datetime'].values)))
        # global ip_asc
        # global ip_dsc 
        # ip_asc = []
        # ip_dsc = []
        global ip
        ip = []
        size = [80,50]
        window = Window(col_off=100, row_off=200, width=size[0], height=size[1])
        for start_datetime_np in sorted_datetimes:
            geotiff_path = stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values[0]
            partial_raster = Raster(geotiff_path, window=window)
            temp = partial_raster.array.reshape(size[1],size[0])
            ip.append(temp)
        sorted_datetimes = pd.DataFrame(sorted_datetimes)
        sorted_datetimes.columns =['start_time']
        sorted_datetimes['start_time'] = pd.to_datetime(sorted_datetimes['start_time'])
        ip = np.array(ip)
        out = np.zeros(size[::-1])
        i = range(size[1])
        j = range(size[0])
        paramlist = list(itertools.product(i,j))
        with Pool(14) as pool:
        #Distribute the parameter sets evenly across the cores
            res = pool.starmap(compute,paramlist)
            for k in range(len(paramlist)):
                out[res[k][1]][res[k][2]] = res[k][0]
        np.save(os.path.join('./s2b08/','%s.npy'%str(end.date()),out))