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
band02 = stack_df[stack_df['source_artefact_name'] == 'B02']
band03 = stack_df[stack_df['source_artefact_name'] == 'B03']
band04 = stack_df[stack_df['source_artefact_name'] == 'B04']
band08 = stack_df[stack_df['source_artefact_name'] == 'B08']
band05 = stack_df[stack_df['source_artefact_name'] == 'B05']
band06 = stack_df[stack_df['source_artefact_name'] == 'B06']
band07 = stack_df[stack_df['source_artefact_name'] == 'B07']
band11 = stack_df[stack_df['source_artefact_name'] == 'B11']
band12 = stack_df[stack_df['source_artefact_name'] == 'B12']
band08 = pd.DataFrame(band08)
band05 = pd.DataFrame(band05)
band06 = pd.DataFrame(band06)
band02 = pd.DataFrame(band02)
band03 = pd.DataFrame(band03)
band04 = pd.DataFrame(band04)
band07 = pd.DataFrame(band07)
band11 = pd.DataFrame(band11)
band12 = pd.DataFrame(band12)
band07.to_pickle('./namelist/s2_07.pkl')
band11.to_pickle('./namelist/s2_11.pkl')
band12.to_pickle('./namelist/s2_12.pkl')
band02.to_pickle('./namelist/s2_02.pkl')
band03.to_pickle('./namelist/s2_03.pkl')
band04.to_pickle('./namelist/s2_04.pkl')
band08.to_pickle('./namelist/s2_08.pkl')
band05.to_pickle('./namelist/s2_05.pkl')
band06.to_pickle('./namelist/s2_06.pkl')
