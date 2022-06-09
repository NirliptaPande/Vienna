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

band08 = stack_df[stack_df['source_artefact_name'] == 'B08']
band05 = stack_df[stack_df['source_artefact_name'] == 'B05']
band06 = stack_df[stack_df['source_artefact_name'] == 'B06']
sorted_datetimes = sorted(list(set(
stack_df['start_datetime'].values)))
band08 = pd.DataFrame(band08)
band05 = pd.DataFrame(band05)
band06 = pd.DataFrame(band06)
band08.to_pickle('./namelist/s2_08.pkl')
band05.to_pickle('./namelist/s2_05.pkl')
band06.to_pickle('./namelist/s2_06.pkl')
