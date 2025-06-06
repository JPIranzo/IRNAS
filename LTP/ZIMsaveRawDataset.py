import pandas as pd
import numpy as np
from datetime import time
import os
import sys
#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl


year_datas=["2014","2015","2016","2019"]
#year_datas=["2014"]
sufix="rht"

meteodata=True
normalizePerDay=False

#pd.set_option('display.max_rows', None)

# create a blank dataframe to save characteristics data (X) and another for class data (Y)
savedfX = pd.DataFrame()
savedfY = pd.DataFrame()

for year_data in year_datas:
    # Load prediction data
    tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

    if meteodata:
        # add meteo to ltpP
        ltpP=ltpP.join(meteoP)

    # delete all NaN columns from ltp
    ltpP = ltpP.dropna(axis=1,how='all')
    # fill NaN values in ltp with the previous value
    ltpP = ltpP.ffill()
    # fill NaN values in ltp with the next value
    ltpP = ltpP.bfill()

    if normalizePerDay:
        # get the mean value of ltp for each day
        ltp_medioP = ltpP.groupby(ltpP.index.date).mean()

        # get the standard deviation of ltp for each day
        ltp_stdP = ltpP.groupby(ltpP.index.date).std()

        # change the index to datetime
        ltp_medioP.index = pd.to_datetime(ltp_medioP.index)
        ltp_stdP.index = pd.to_datetime(ltp_stdP.index)

        # resample ltp_medio and ltp_std to minutes
        ltp_medioP = ltp_medioP.resample('T').pad()
        ltp_stdP = ltp_stdP.resample('T').pad()

        # ltp normalized per day
        ltpP = (ltpP - ltp_medioP) / ltp_stdP
    # create two columns Fecha and Hora with the date and time of the index as string
    ltpP['Fecha'] = ltpP.index.date.astype(str)
    ltpP['Hora'] = ltpP.index.time.astype(str)

    # substitute the index with the columns Fecha and Hora
    ltpP = ltpP.set_index(['Fecha','Hora'])

    # valdata index as string
    valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')

    # unstack valdatapd
    valdatapd = valdatapd.unstack()

    # unstack the second level of the index of ltpP
    ltpP = ltpP.unstack(level=1)
    #stack the first level of the index of ltpP
    ltpP = ltpP.stack(level=0)

    # swap the levels of the index of ltpP
    ltpP = ltpP.swaplevel()

    # sort the index of ltpP
    ltpP = ltpP.sort_index()

    # sort the index of valdatapd
    valdatapd = valdatapd.sort_index()

    if meteodata:
        # separate the meteo data once again from ltpP as the index that doesn't start with LTP
        meteoT = ltpP.loc[ltpP.index.get_level_values(0).str.startswith('LTP')==False,:]
        # separate the ltp data from ltpP as the index that starts with LTP
        ltpP = ltpP.loc[ltpP.index.get_level_values(0).str.startswith('LTP')==True,:]

        # add a column level to ltpP with the name 'LTP'
        ltpP.columns = pd.MultiIndex.from_product([['LTP'],ltpP.columns])

        # unstack the first level of the index of meteoT
        meteoT = meteoT.unstack(level=0)
        # swap the levels of the columns of meteoT
        meteoT = meteoT.swaplevel(axis=1)
        # sort the columns of meteoT
        meteoT = meteoT.sort_index(axis=1)

        # merge meteo into ltp by the second level of the index in ltp and the only level of the index in meteoT
        ltpP = ltpP.join(meteoT)

    # get the intersection index of valdatapd and ltpP
    valdatapd_ltp = valdatapd.index.intersection(ltpP.index)

    # remove the values of ltp that are not in valdatapd
    ltpv = ltpP.loc[valdatapd_ltp,:]
    
    # print("ltpt")
    # print(ltpt)
    # print("meteo_ltp")
    # print(meteo_ltp)
    # print("meteoT_norm")
    # print(meteoT_norm)
    # print("ltpt_col")
    # print(ltpt_col)

    Xv=ltpv
    Yv=valdatapd

    # Xv as dataframe with the index of Yv
    Xv=pd.DataFrame(Xv,index=Yv.index)

    # add Xv and Yv to their respective dataframes
    savedfX=pd.concat([savedfX,Xv])
    savedfY=pd.concat([savedfY,Yv])

    print("Year: ", year_data)
    print("X shape: ", Xv.shape)
    print("Y shape: ", Yv.shape)

# add savedfY to savedfX as a column named 'Y', subtracting 1 so that the values of Y are between 0 and 2 (as is the usual case when numbering classes)
savedfX['Y']=savedfY-1

# create a string with the last two digits of each year in year_datas
year_datas_str = ''.join(year_datas[-2:] for year_datas in year_datas)
if meteodata:
    year_datas_str = year_datas_str + 'meteo'

print(savedfX.columns)

#drop first and last level of columns names
savedfX.columns = savedfX.columns.droplevel(0)
print(savedfX)

# rename the last column (unnamed) to 'Y'
savedfX = savedfX.rename(columns={savedfX.columns[-1]: 'Y'})
print(savedfX)

# store savedfX in a csv with a name composed of 'db' followed by the last two digits of each year in year_datas
savedfX.to_csv('db'+year_datas_str+'raw.csv')