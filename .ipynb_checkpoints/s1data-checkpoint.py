from rasterio.windows import Window
from scipy.ndimage import median_filter
from mantle_utils.alg.raster import Raster
import pandas as pd
import numpy as np
from multiprocessing import Pool
def prep_data(name):
    df = pd.read_pickle('./namelist/s1_alltime.pkl')
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df.sort_index(inplace=True)
    df_asc = pd.read_pickle('./namelist/%s.pkl'%name)
    df_asc['start_datetime'] = pd.to_datetime(df_asc[0])
    df_asc.sort_index(inplace=True)
    df = df.merge(df_asc,how='inner', on='start_datetime')
    date = sorted(list(set(df['start_datetime'].values)))
    size = [80,50]
    window = Window(col_off=100, row_off=200, width=size[0], height=size[1])
    ip = []
    for start_datetime_np in date:
        geotiff_path = df.loc[df['start_datetime'] == start_datetime_np]['raster_path'].values[0]
        partial_raster = Raster(geotiff_path, window=window)
        temp = partial_raster.array.reshape(size[1],size[0])
        temp = median_filter(temp, size=3)
        ip.append(temp)
    ip = np.array(ip)
    #ip = ip[ip != 0]
    np.save('./namelist/%s.npy'%name,ip)
if __name__ == "__main__":
    with Pool() as pool:
        pool.map(prep_data,['s1_dates_d','s1_dates_a'])