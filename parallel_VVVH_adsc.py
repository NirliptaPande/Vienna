import datetime
import pandas as pd
from prophet import Prophet
from pathlib import Path
from mantle_utils.alg.raster import Raster
from rasterio.windows import Window
import numpy as np
from multiprocessing import Pool
import itertools
import os.path
def daterange(start_date, end_date):
    for n in range(0,int((end_date - start_date).days),15):
        yield start_date + datetime.timedelta(n)

def compute(idx0,idx1):#df will not be the input, i,j will be
    ip = ip_asc
    sorted_dates = sorted_datetimes_asc
    save_path_t = './single_op_asc_dsc/asc/trends_partial/'
    save_path_c = './single_op_asc_dsc/asc/comp_partial/'
    if(decide & 0b1 == 1):
        ip = ip_dsc
        save_path_t = './single_op_asc_dsc/dsc/trends_partial/'
        save_path_c = './single_op_asc_dsc/dsc/comp_partial/'
        sorted_dates = sorted_datetimes_dsc
    # df is the  final dataset, 1D in this case, with 2 columns, time and Y values
    df = {'ds':sorted_dates['start_time'],'y':ip[:,idx0,idx1]}
    df = pd.DataFrame(df)
    df = df[df['y']!=255]
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)
    if (((idx0+5)%10==0)&(idx1%16==0)):
        fig1 = m.plot(forecast)
        fig2 = m.plot_components(forecast)
        fig1.savefig(os.path.join(save_path_t,'%s_%s_%s.svg'%(str(end),str(idx0),str(idx1))))
        fig2.savefig(os.path.join(save_path_c,'%s_%s_%s.svg'%(str(end),str(idx0),str(idx1))))
    
    return [forecast['yhat'].values[0],idx0,idx1]
if __name__ == "__main__":
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2020-01-01')
    global end
    global sorted_datetimes_asc
    global sorted_datetimes_dsc
    df = pd.read_pickle('./namelist/s1_alltime.pkl')
    df_asc = pd.read_pickle('./namelist/s1_dates_a.pkl')
    df_dsc = pd.read_pickle('./namelist/s1_dates_d.pkl')
    df_asc.columns =['start_time']
    df_dsc.columns =['start_time']
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df_asc['start_time'] = pd.to_datetime(df_asc['start_time'])
    df_dsc['start_time'] = pd.to_datetime(df_dsc['start_time'])
    for single_date in daterange(start_date, end_date):
        start = single_date
        end = start + datetime.timedelta(days=85)
        stack_df = df[np.logical_and(df['start_datetime'] >= np.datetime64(start), df['start_datetime'] < np.datetime64(end))]
        sorted_datetimes_asc = df_asc[np.logical_and(df_asc['start_time'] >= np.datetime64(start), df_asc['start_time'] < np.datetime64(end))]
        sorted_datetimes_dsc = df_dsc[np.logical_and(df_dsc['start_time'] >= np.datetime64(start), df_dsc['start_time'] < np.datetime64(end))]
        sorted_datetimes_dsc['start_time'] = pd.to_datetime(sorted_datetimes_dsc['start_time'])
        sorted_datetimes_asc['start_time'] = pd.to_datetime(sorted_datetimes_asc['start_time'])
        sorted_datetimes = sorted(list(set(
        stack_df['start_datetime'].values)))
        global ip_asc
        global ip_dsc 
        ip_asc = []
        ip_dsc = []
        size = [80,50]
        window = Window(col_off=100, row_off=200, width=size[0], height=size[1])
        for start_datetime_np in sorted_datetimes:
            geotiff_path = stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values[0]
            partial_raster = Raster(geotiff_path, window=window)
            temp = partial_raster.array.reshape(size[1],size[0])
            if (start_datetime_np in sorted_datetimes_asc.values):
                ip_asc.append(temp)
            if (start_datetime_np in sorted_datetimes_dsc.values):
                ip_dsc.append(temp)
        global decide
        decide = 0b0
        ip_asc = np.array(ip_asc)
        ip_dsc = np.array(ip_dsc)  
        out_a = np.zeros(size[::-1])
        out_d = np.zeros(size[::-1])
        i = range(size[1])
        j = range(size[0])
        paramlist = list(itertools.product(i,j))
        with Pool(14) as pool:
            res = pool.starmap(compute,paramlist)
            for k in range(len(paramlist)):
                out_a[res[k][1]][res[k][2]] = res[k][0]
        np.save('./namelist/prediction_asc_%s.npy'%str(end),out_a)
        check = 0b1
        with Pool(14) as pool:
            res = pool.starmap(compute,paramlist)
            for k in range(len(paramlist)):
                out_d[res[k][1]][res[k][2]] = res[k][0]
        np.save('./namelist/prediction_dsc_%s.npy'%str(end),out_d)