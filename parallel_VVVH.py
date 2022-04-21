import datetime
from mantle_utils.workflow.enqueue import get_raster_stack_paths_and_metadata
from mantle_utils.workflow.process import sanitize_raster_stack_paths_and_metadata
from mantle_utils.workflow.metadatadb.product import ProductCollection
from mantle_utils.workflow.metadatadb.product import create_s2_tile_idx_search_dict
from mantle_utils.workflow.metadatadb.product import create_datetime_search_dict
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
    save_path_t=''
    save_path_c=''
    if(check & 0b1 == 0):
        #ip = ip_asc
        save_path_t = './single_op_asc_dsc/asc/trends/'
        save_path_c = './single_op_asc_dsc/asc/comp/'
    if(check & 0b1 == 1):
        ip = ip_dsc
        save_path_t = './single_op_asc_dsc/dsc/trends/'
        save_path_c = './single_op_asc_dsc/dsc/comp/'
    df = {'ds':sorted_datetimes,'y':ip[:,idx0,idx1]}
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
    stacking_source_run_name = 'nirlipta_30UXB_stack'
    tile_idx = '30UXB'

    stack_raster_paths, stack_metadata = \
        get_raster_stack_paths_and_metadata(
            env_name='production',
            stacking_source_run_name=stacking_source_run_name,
            tile_idx=tile_idx,
            raise_on_inconsistent_stack=False)
       
    stack_df = sanitize_raster_stack_paths_and_metadata(
        raster_paths=stack_raster_paths,
        stack_metadata=stack_metadata,
        sanitize_paths=False)

    product_collection = ProductCollection()

    product_tile_search_dict = \
        create_s2_tile_idx_search_dict(tile_idx=tile_idx)

    stack_df = stack_df[stack_df['source_workflow_name'] == 's1-generate_composites']
    stack_df = stack_df[np.logical_and(stack_df['start_datetime'] >= np.datetime64('2019-01-01'), stack_df['start_datetime'] < np.datetime64('2019-04-01'))]
    global sorted_datetimes
    sorted_datetimes = sorted(list(set(
    stack_df['start_datetime'].values)))
    global ip_asc
    global ip_dsc 
    ip_asc = []
    ip_dsc = []
    size = [80,50]
    window = Window(col_off=100, row_off=200, width=size[0], height=size[1])
    for start_datetime_np in sorted_datetimes:

        # cf. https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64  # NOQA:501
        start_datetime = datetime.datetime.utcfromtimestamp(
            start_datetime_np.astype(int) * 1e-9)

        product_datetime_search_dict = \
            create_datetime_search_dict(
                datetime=start_datetime)
        
        search_dict = {
            'product_metadata.year': start_datetime.year,
            'product_metadata.month': start_datetime.month,
            'product_metadata.day': start_datetime.day,
            **product_tile_search_dict}

        products = product_collection.find_in_db(
            env_name='production',
            projection_dict={
                '_id': 0,
                'product_metadata.pass': 1},
            workflow_name='s1-generate_s2tiles',
            run_name='v1.0-v1.0',
            search_dict=search_dict)
        geotiff_path = stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values[0]
        partial_raster = Raster(geotiff_path, window=window)
        temp = partial_raster.array.reshape(size[1],size[0])
        #pdb.set_trace()

        if products[0]['product_metadata']['pass'] == 'ASCENDING':
            ip_asc.append(temp)
        if products[0]['product_metadata']['pass'] == 'DESCENDING':
            ip_dsc.append(temp)  


    # here you do whatever you want with your DESCENDING product

    #loop through time do it in a continous manner???
    
    #stack_df = stack_df.loc[stack_df['source_artefact_name'] == 'VVVH']
    
    global check
    check = 0b0
    
    #for start_datetime_np in sorted_datetimes:
       # print(stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values)
        
    #pdb.set_trace()
    ip_asc = np.array(ip_asc)
    ip_dsc = np.array(ip_dsc) 
    # df is the  final dataset, 1D in this case, with 2 columns, time and Y values
    out_a = np.zeros(size[::-1])
    out_d = np.zeros(size[::-1])
    i = range(size[1])
    j = range(size[0])
    paramlist = list(itertools.product(i,j))
    #Generate processes equal to the number of cores
    with Pool(12) as pool:
    #Distribute the parameter sets evenly across the cores
        res = pool.starmap(compute,paramlist)
        for k in range(len(paramlist)):
            out_a[res[k][1]][res[k][2]] = res[k][0]
    check = 0b1
    with Pool(12) as pool:
    #Distribute the parameter sets evenly across the cores
        res = pool.starmap(compute,paramlist)
        for k in range(len(paramlist)):
            out_d[res[k][1]][res[k][2]] = res[k][0]
        

