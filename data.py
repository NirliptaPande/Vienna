import numpy as np
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
sorted_datetimes_asc =[]
sorted_datetimes_dsc =[]
sorted_datetimes = sorted(list(set(
stack_df['start_datetime'].values)))
for start_datetime_np in sorted_datetimes:
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

    if products[0]['product_metadata']['pass'] == 'ASCENDING':
        sorted_datetimes_asc.append(start_datetime_np)
    if products[0]['product_metadata']['pass'] == 'DESCENDING':
        sorted_datetimes_dsc.append(start_datetime_np)
stack_df.to_feather('./namelist/s1_alltime.ftr')
dt_sort_a = pd.DataFrame(sorted_datetimes_asc)
dt_sort_d = pd.DataFrame(sorted_datetimes_dsc)
dt_sort_a.to_feather('./namelist/s1_dates_a.ftr')
dt_sort_d.to_feather('./namelist/s1_dates_d.ftr')
