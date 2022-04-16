import pdb
from mantle_utils.workflow.enqueue import get_raster_stack_paths_and_metadata
from mantle_utils.workflow.process import sanitize_raster_stack_paths_and_metadata
import pandas as pd
from prophet import Prophet
from pathlib import Path
from mantle_utils.alg.raster import Raster
from rasterio.windows import Window
import numpy as np

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
#the data is from 2019 01 Jan to 2021 Dec 31 frequency is still unknown
#extract data of 3 months, for past 'n' years
#I am not sure if March '19 can follow Jan '20
#I was thinking of filling up the missing data after cloud cover, but first thing is to get the system up and running
# okay, so this is how it is going to work,
#1. take a starting point ()()()
#2. Extract all S2 tiles 8 weeks from the starting point((Did this for S1)()())
#2.1Extract a smaller 5X5 section, let's say, upper left corner((This too)()())
#3. Take the next date to be the target, after 8th week to be the target((This is also done)()())
#4. test1: I/P = B1 O/P = B1
#5. test2: I/P = S2 O/P = B1
#6. test2: I/P = S2+S1 O/P = B1
#7. O/P should contain average missing pixels per image, median missing pixels per image, error 
#In this system, spatial correlation is not considered, O/P band and I/P bands are the same
# target = stack_df[np.logical_and(source_artefact_name = 'VVVH',stack_df['start_datetime'] >= np.datetime64('2019-01-05'))]
# target = target.iloc[0]
window = Window(col_off=100, row_off=200, width=80, height=50)
# target = Raster(Path(target[raster_path]), window=window)
# target = pd.Dataframe(target)#target converted to dataframe,output mini raster
stack_df = stack_df[np.logical_and(stack_df['start_datetime'] >= np.datetime64('2019-01-01'), stack_df['start_datetime'] < np.datetime64('2019-04-01'))]
#now stack_df has paths of both S1 and S2 b/w Jan beginning to end of April
stack_df = stack_df.loc[stack_df['source_artefact_name'] == 'VVVH'] #sorted out the S1 data
#stack_df = stack_df[source_artefact_name = 'B02']#sorts out S1 B02
# a potential problem is loading 5 month data across time with 6 day repeat cycle for S1, 5 day for S2
#load the data using raster path
#how to parallize it
sorted_datetimes = sorted(list(set(
    stack_df['start_datetime'].values)))
ip = []
for start_datetime_np in sorted_datetimes:
   # print(stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values)
    geotiff_path = stack_df.loc[stack_df['start_datetime'] == start_datetime_np]['raster_path'].values[0]
    partial_raster = Raster(geotiff_path, window=window)
    temp = partial_raster.array.reshape(50,80)
    ip.append(temp)
#pdb.set_trace()
ip = np.array(ip) 
# df is the  final dataset, 1D in this case, with 2 columns, time and Y values
out = np.zeros([50,80])
for i in range(50):
    for j in range(80):
        df = {'ds':sorted_datetimes,'y':ip[:,i,j]}
        df = pd.DataFrame(df)
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        out[i][j]=forecast['yhat'].values[0]

np.save('prediction.npy',out)
# m.plot_components(forecast).savefig('01010105_comp.svg')#saves trend, idt very useful as data is of a weekly frequency
# fig = m.plot(forecast)
# fig.savefig('01010105.svg')
#make a 2D array from all the fig
#scale it up, as in this works on only 1 section, 1 time period =, put it in a loop and make the final variable names practical
#Interpretable error, idk how to, RMSE? I mean, idk what kind of values we are dealing with, so no idea
#How to compare it to a S1 tile
