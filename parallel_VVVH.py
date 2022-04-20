from mantle_utils.workflow.enqueue import get_raster_stack_paths_and_metadata
from mantle_utils.workflow.process import sanitize_raster_stack_paths_and_metadata
import pandas as pd
from prophet import Prophet
from pathlib import Path
from mantle_utils.alg.raster import Raster
from rasterio.windows import Window
import numpy as np
from multiprocessing import Pool
def compute(idx,ip, sorted_datetimes):#df will not be the input, i,j will be
    df = {'ds':sorted_datetimes,'y':ip[:,idx[0],idx[1]]}
    df = pd.DataFrame(df)
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)
    #out[i][j]=
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    fig1.savefig('trend%s%s.svg'%(str(idx[0]),str(idx[1]))
    fig2.savefig('comp%s%s.svg'%(str(idx[0]),str(idx[1]))
    #np.save('prediction.npy',out)
    return [forecast['yhat'].values[0],i,j]

if __name__ == "__main__":
    stack_raster_paths, stack_metadata = \
    get_raster_stack_paths_and_metadata(
        env_name='production',
        stacking_source_run_name='nirlipta_30UXB_stack',
        tile_idx='30UXB',
        raise_on_inconsistent_stack=False)

    stack_df = sanitize_raster_stack_paths_and_metadata(
    raster_paths=stack_raster_paths,
    stack_metadata=stack_metadata,
    sanitize_paths=False)
    size = [80,50]
    window = Window(col_off=100, row_off=200, width=size[0], height=size[1])
    #loop through time do it in a continous manner???
    stack_df = stack_df[np.logical_and(stack_df['start_datetime'] >= np.datetime64('2019-01-01'), stack_df['start_datetime'] < np.datetime64('2019-04-01'))]
    stack_df = stack_df.loc[stack_df['source_artefact_name'] == 'VVVH']
    sorted_datetimes = sorted(list(set(
    stack_df['start_datetime'].values)))
    ip = []
    for start_datetime_np in sorted_datetimes:
       # print(stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values)
        geotiff_path = stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values[0]
        partial_raster = Raster(geotiff_path, window=window)
        temp = partial_raster.array.reshape(size[1],size[0])
        ip.append(temp)
    #pdb.set_trace()
    ip = np.array(ip) 
    # df is the  final dataset, 1D in this case, with 2 columns, time and Y values
    out = np.zeros(size[::-1])
    i = range(size[1])
    j = range(size[0])
    paramlist = list(itertools.product(i,j))
    #Generate processes equal to the number of cores
    with Pool() as pool:
    #Distribute the parameter sets evenly across the cores
        res = pool.starmap(compute,zip(paramlist,ip,sorted_datetimes))
        for _ in range(len(paramlist)):
            out[res[_][1]][res[_][2]] = res[_][0]
        

