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
#import pbd
def compute(idx0,idx1):#df will not be the input, i,j will be
    #ip = ip_'%s'sc %(order)
    ip = ip_asc
    save_path_t = './single_op_asc_dsc/asc/trends/'
    save_path_c = './single_op_asc_dsc/asc/comp/'
    sorted_dates = sorted_datetimes_asc
    if(check & 0b1 == 1):
        ip = ip_dsc
        save_path_t = './single_op_asc_dsc/dsc/trends/'
        save_path_c = './single_op_asc_dsc/dsc/comp/'
        sorted_dates = sorted_datetimes_dsc
    df = {'ds':sorted_dates,'y':ip[:,idx0,idx1]}
    df = pd.DataFrame(df)
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)
    #out[i][j]=

    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    fig1.savefig(os.path.join(save_path_t,'%s_%s.svg'%(str(idx0),str(idx1))))
    fig2.savefig(os.path.join(save_path_c,'%s_%s.svg'%(str(idx0),str(idx1))))
    #np.save('prediction.npy',out)
    return [forecast['yhat'].values[0],idx0,idx1]

if __name__ == "__main__":
    stack_df = pd.read_feather('./namelist/s1_alltime.ftr')
    stack_df = pd.DataFrame(stack_df)
    stack_df = stack_df[np.logical_and(stack_df['start_datetime'] >= np.datetime64('2019-01-01'), stack_df['start_datetime'] < np.datetime64('2019-04-01'))]
    global sorted_datetimes_asc
    global sorted_datetimes_dsc
    sorted_datetimes_asc = pd.read_feather('./namelist/s1_dates_a.ftr')
    sorted_datetimes_dsc = pd.read_feather('./namelist/s1_dates_d.ftr')
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
        if (start_datetime_np in np.datetime64(sorted_datetimes_asc.values)):
            ip_asc.append(temp)
        if (start_datetime_np in np.datetime64(sorted_datetimes_dsc.values)):
            ip_dsc.append(temp)
    
    global check
    check = 0b0
    ip_asc = np.array(ip_asc)
    ip_dsc = np.array(ip_dsc) 
    # df is the  final dataset, 1D in this case, with 2 columns, time and Y values
    out_a = np.zeros(size[::-1])
    out_d = np.zeros(size[::-1])
    i = range(size[1])
    j = range(size[0])
    paramlist = list(itertools.product(i,j))
    #Generate processes equal to the number of cores
    with Pool(16) as pool:
    #Distribute the parameter sets evenly across the cores
        res = pool.starmap(compute,paramlist)
        for k in range(len(paramlist)):
            out_a[res[k][1]][res[k][2]] = res[k][0]
    check = 0b1
    with Pool(16) as pool:
    #Distribute the parameter sets evenly across the cores
        res = pool.starmap(compute,paramlist)
        for k in range(len(paramlist)):
            out_d[res[k][1]][res[k][2]] = res[k][0]
        

