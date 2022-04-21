import datetime
from mantle_utils.workflow.enqueue import get_raster_stack_paths_and_metadata
from mantle_utils.workflow.process import sanitize_raster_stack_paths_and_metadata
from mantle_utils.workflow.metadatadb.product import ProductCollection
from mantle_utils.workflow.metadatadb.product import create_s2_tile_idx_search_dict
from mantle_utils.workflow.metadatadb.product import create_datetime_search_dict
import pandas as pd

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
#global sorted_datetimes_asc
#global sorted_datetimes_dsc
sorted_datetimes_asc =[]
sorted_datetimes_dsc =[]
sorted_datetimes = sorted(list(set(
stack_df['start_datetime'].values)))
#global ip_asc
#global ip_dsc 
#ip_asc = []
#ip_dsc = []
#size = [80,50]
#window = Window(col_off=100, row_off=200, width=size[0], height=size[1])
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
    # geotiff_path = stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values[0]
    # partial_raster = Raster(geotiff_path, window=window)
    # temp = partial_raster.array.reshape(size[1],size[0])
    # #pdb.set_trace()

    if products[0]['product_metadata']['pass'] == 'ASCENDING':
        # ip_asc.append(temp)
        sorted_datetimes_asc.append(start_datetime_np)
    if products[0]['product_metadata']['pass'] == 'DESCENDING':
        # ip_dsc.append(temp)  
        sorted_datetimes_dsc.append(start_datetime_np)
stack_df.to_feather('./namelist/s1_alltime.ftr')
dt_sort = pd.DataFrame(sorted_datetimes)
dt_sort_a = pd.DataFrame(sorted_datetimes_asc)
dt_sort_d = pd.DataFrame(sorted_datetimes_dsc)
stack_df.to_feather('./namelist/s1_all_a.ftr')
stack_df.to_feather('./namelist/s1_all_d.ftr')
