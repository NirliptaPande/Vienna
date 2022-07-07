#Drop NA
#Smooth data
#Create 80X50 boxes
from rasterio.windows import Window
from scipy.ndimage import median_filter
from mantle_utils.alg.raster import Raster
import pandas as pd
import numpy as np
from multiprocessing import Pool
def prep_data(band):
    df = pd.read_pickle('./namelist/s2_%s.pkl'%band)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
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
    np.save('./namelist/b%s.npy'%band,ip)
if __name__ == "__main__":
    with Pool() as pool:
        pool.map(prep_data,['05','06','07','11','12'])