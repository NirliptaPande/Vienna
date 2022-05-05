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
#import pdb
def compute(idx0,idx1):#df will not be the input, i,j will be
    # ip = ip_asc
    sorted_dates = sorted_datetimes
    # if(decide & 0b1 == 1):
    #     ip = ip_dsc
    #     save_path_t = './single_op_asc_dsc/dsc/trends_partial/'
    #     save_path_c = './single_op_asc_dsc/dsc/comp_partial/'
    #     sorted_dates = sorted_datetimes_dsc
    df = {'ds':sorted_dates['start_time'],'y':ip[:,idx0,idx1]}
    df = pd.DataFrame(df)
    df = df[df['y']!=255]
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)
    if (((idx0+5)%10==0)&(idx1%16==0)):
        save_path_t = './single_op/trends/'
        save_path_c = './single_op/comp/'
        fig1 = m.plot(forecast)
        fig2 = m.plot_components(forecast)
        fig1.savefig(os.path.join(save_path_t,'%s_%s.svg'%(str(idx0),str(idx1))))
        fig2.savefig(os.path.join(save_path_c,'%s_%s.svg'%(str(idx0),str(idx1))))
    
    return [forecast['yhat'].values[0],idx0,idx1]

if __name__ == "__main__":
    start = '2019-05-01'
    end = '2019-08-01'
    stack_df = pd.read_pickle('./namelist/s1_alltime.pkl')
    #stack_df = pd.DataFrame(stack_df)
    stack_df['start_datetime'] = pd.to_datetime(stack_df['start_datetime'])
    stack_df = stack_df[np.logical_and(stack_df['start_datetime'] >= np.datetime64(start), stack_df['start_datetime'] < np.datetime64(end))]
    # global sorted_datetimes_asc
    # global sorted_datetimes_dsc
    # sorted_datetimes_asc = pd.read_pickle('./namelist/s1_dates_a.pkl')
    # sorted_datetimes_dsc = pd.read_pickle('./namelist/s1_dates_d.pkl')
    # sorted_datetimes_asc.columns =['start_time']
    # sorted_datetimes_dsc.columns =['start_time']
    # sorted_datetimes_asc = sorted_datetimes_asc[np.logical_and(sorted_datetimes_asc['start_time'] >= np.datetime64(start), sorted_datetimes_asc['start_time'] < np.datetime64(end))]
    # sorted_datetimes_dsc = sorted_datetimes_dsc[np.logical_and(sorted_datetimes_dsc['start_time'] >= np.datetime64(start), sorted_datetimes_dsc['start_time'] < np.datetime64(end))]
    # sorted_datetimes_dsc['start_time'] = pd.to_datetime(sorted_datetimes_dsc['start_time'])
    # sorted_datetimes_asc['start_time'] = pd.to_datetime(sorted_datetimes_asc['start_time'])
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
        # if (start_datetime_np in sorted_datetimes_asc.values):
        #     ip_asc.append(temp)
        # if (start_datetime_np in sorted_datetimes_dsc.values):
        #     ip_dsc.append(temp)
    # global decide
    # decide = 0b0
    # ip_asc = np.array(ip_asc)
    # ip_dsc = np.array(ip_dsc) 
    # df is the  final dataset, 1D in this case, with 2 columns, time and Y values
    # out_a = np.zeros(size[::-1])
    # out_d = np.zeros(size[::-1])
    sorted_datetimes = pd.DataFrame(sorted_datetimes)
    sorted_datetimes.columns =['start_time']
    sorted_datetimes['start_time'] = pd.to_datetime(sorted_datetimes['start_time'])
    ip = np.array(ip)
    out = np.zeros(size[::-1])
    i = range(size[1])
    j = range(size[0])
    paramlist = list(itertools.product(i,j))
    #pdb.set_trace()
    #Generate processes equal to the number of cores
    # with Pool(16) as pool:
    # #Distribute the parameter sets evenly across the cores
    #     res = pool.starmap(compute,paramlist)
    #     for k in range(len(paramlist)):
    #         out_a[res[k][1]][res[k][2]] = res[k][0]
    # np.save('./namelist/prediction_asc.npy',out_a)
    # decide = 0b1
    # with Pool(12) as pool:
    # #Distribute the parameter sets evenly across the cores
    #     res = pool.starmap(compute,paramlist)
    #     for k in range(len(paramlist)):
    #         out_d[res[k][1]][res[k][2]] = res[k][0]
    # np.save('./namelist/prediction_dsc.npy',out_d)
    with Pool(14) as pool:
    #Distribute the parameter sets evenly across the cores
        res = pool.starmap(compute,paramlist)
        for k in range(len(paramlist)):
            out[res[k][1]][res[k][2]] = res[k][0]
    np.save('./namelist/prediction.npy',out)