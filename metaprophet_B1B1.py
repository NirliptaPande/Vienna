from mantle_utils.workflow.enqueue import get_raster_stack_paths_and_metadata
from mantle_utils.workflow.process import sanitize_raster_stack_paths_and_metadata

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
#stack_df has ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12'] bands from S2 and VVVH polarity from S1
#the data is from 2019 01 Jan to 2021 Dec 31 frequency is still unkown
#extract data of 3 months, for past 'n' years
#I am not sure if March '19 can follow Jan '20
#I was thinking of filling up the missing data after cloud cover, but first thing is to get the system up and running
stack_df[np.logical_and(stack_df['start_datetime'] > np.datetime64('2019-01-04'), stack_df['start_datetime'] < np.datetime64('2019-01-10'))]
# okay, so this is how it is going to work,
#1. take a starting point
#2. Extract all S2 tiles 8 weeks from the starting point
#2.1Extract a smaller 5X5 section, let's say, upper left corner
#3. Take the next date to be the target, after 8th week to be the target
#4. test1: I/P = B1 O/P = B1
#5. test2: I/P = S2 O/P = B1
#6. test2: I/P = S2+S1 O/P = B1
#7. O/P should contain average missing pixels per image, median missing pixels per image, error 
#In this system, spatial correlation is not considered, O/P band and I/P bands are the same
