import datetime
import pandas as pd
from prophet import Prophet
import numpy as np
import itertools

def daterange(start_date, end_date):
    for n in range(0,int((end_date - start_date).days),15):
        yield start_date + datetime.timedelta(n)

if __name__ == "__main__":
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2020-01-01')
    #global end
    df = pd.read_pickle('./namelist/s1_dates_a.pkl')
    df['start_datetime'] = pd.to_datetime(df[0])
    df.sort_index(inplace=True)
    #df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    date = sorted(list(set(df['start_datetime'].values)))
    #date = pd.DataFrame(date)
    ip = np.load('./namelist/s1_dates_a.npy')
    #ip = np.swapaxes(ip,0,2)
    ip4532 = ip[:, 45, 32]
    ip2516 = ip[:, 25, 16]
    df4532 = {'ds':date,'y':ip4532}
    df4532 = pd.DataFrame(df4532)
    df4532 = df4532[df4532['y']!=255]
    df2516 = {'ds':date,'y':ip2516}
    df2516 = pd.DataFrame(df2516)
    df2516 = df2516[df2516['y']!=255]
    op4532 = []
    op2516 = []
    for single_date in daterange(start_date, end_date):
        start = single_date
        end = start + datetime.timedelta(days=85)
        #stack_df = df[np.logical_and(df['start_datetime'] >= np.datetime64(start), df['start_datetime'] < np.datetime64(end))]
        df = df4532[np.logical_and(df4532['ds']>=np.datetime64(start),df4532['ds']<np.datetime64(end))]
        m =  Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        op4532.append([end,forecast['yhat'].values[0]])
        df = df2516[np.logical_and(df2516['ds']>=np.datetime64(start),df2516['ds']<np.datetime64(end))]
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        op2516.append([end,forecast['yhat'].values[0]])
    np.save('./prophet/s1asc_4532.npy',op4532)
    np.save('./prophet/s1asc_2516.npy',op2516)