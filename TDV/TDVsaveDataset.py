import sys
from time import time
import matplotlib
from matplotlib import patches as mpatches
from matplotlib.markers import MarkerStyle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import isadoralib as isl
import sklearn.discriminant_analysis as skda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp

years=["2014","2015","2016","2019"]
stress_mismatch=[-1,0,0,0]

# Create an empty dataframe to save the processed data
savedf=pd.DataFrame()
# For each year in the list
for i in range(0,len(years),1):
    # Get the year
    year=years[i]
    # Get the stress mismatch
    mismatch=stress_mismatch[i]

    # Load the data
    tdv,ltp,meteo,stress=isl.cargaDatosTDV(year,"")

    # Remove the nan values
    tdv=tdv.dropna()
    ltp=ltp.dropna()
    meteo=meteo.dropna()
    stress=stress.dropna()

    # Get the values of tdv between 5 and 8 of each day
    tdv_5_8 = tdv.between_time(time(5,0),time(8,0))
    # Get the maximum of tdv between 5 and 8 of each day
    tdv_5_8_max = tdv_5_8.groupby(tdv_5_8.index.date).max()
    # Get the difference between the maximum of tdv between 5 and 8 of each day
    tdv_5_8_max_diff = tdv_5_8_max.diff(periods=1).dropna()
    # Get the sign of the difference between the maximum of tdv between 5 and 8 of each day
    tdv_5_8_max_diff_sign = tdv_5_8_max_diff.apply(np.sign)
    # Change the negative values to 0
    tdv_5_8_max_diff_sign[tdv_5_8_max_diff_sign<0]=0
    # Create a dataframe that is 1 when tdv_5_8_max_diff_sign is 0 and 0 when it is 1
    tdv_5_8_max_diff_sign_inv=tdv_5_8_max_diff_sign.apply(lambda x: 1-x)

    # Create two dataframes with the size of tdv_5_8_max_diff_sign and values 0
    pk0=pd.DataFrame(np.zeros(tdv_5_8_max_diff_sign.shape),index=tdv_5_8_max_diff_sign.index,columns=tdv_5_8_max_diff_sign.columns)
    pk1=pd.DataFrame(np.zeros(tdv_5_8_max_diff_sign.shape),index=tdv_5_8_max_diff_sign.index,columns=tdv_5_8_max_diff_sign.columns)
    # For each day in tdv_5_8_max_diff_sign
    for i in tdv_5_8_max_diff_sign.index:
        # If it is the first row
        if i==tdv_5_8_max_diff_sign.index[0]:
            # Add to pk0 the value of tdv_5_8_max_diff_sign_inv
            pk0.loc[i]=tdv_5_8_max_diff_sign_inv.loc[i]
            # Add to pk1 the value of tdv_5_8_max_diff_sign
            pk1.loc[i]=tdv_5_8_max_diff_sign.loc[i]
        # If it is not the first row
        else:
            # Get the previous index by subtracting one day
            i_ant=i-pd.Timedelta(days=1)
            # Add to pk0 the value of the previous row of pk0 plus the value of the row of tdv_5_8_max_diff_sign_inv, multiplied by the value of the row of tdv_5_8_max_diff_sign_inv
            pk0.loc[i]=(pk0.loc[i_ant]+tdv_5_8_max_diff_sign_inv.loc[i])*tdv_5_8_max_diff_sign_inv.loc[i]
            # Add to pk1 the value of the previous row of pk1 plus the value of the row of tdv_5_8_max_diff_sign, multiplied by the value of the row of tdv_5_8_max_diff_sign
            pk1.loc[i]=(pk1.loc[i_ant]+tdv_5_8_max_diff_sign.loc[i])*tdv_5_8_max_diff_sign.loc[i]
    # Substract pk0 from pk1 to get the number of days since the last change of trend, with the sign of the current trend
    pk=pk1-pk0

    # create a new dataframe fk with the exponentially weighted moving average of tdv_5_8_max_diff_sign
    bk=tdv_5_8_max_diff_sign.ewm(alpha=0.5).mean()
    
    # Get the current trend into a new dataframe
    trend=tdv_5_8_max_diff_sign.copy()
    
    # Remove the nan values
    bk=bk.dropna()
    trend=trend.dropna()

    # Create a dataframe with diff tdv_5_8_max_diff_sign that represents the changes of trend
    ctend=pd.DataFrame(tdv_5_8_max_diff_sign.diff(periods=1).dropna())

    # Equal to 1 the non-null values
    ctend[ctend!=0]=1

    # Get the values of the maximum in the time slot when there is a change of trend
    max_ctend=tdv_5_8_max[ctend!=0]
    # Fill the null values with the previous value
    max_ctend=max_ctend.fillna(method='ffill')
    # When there is no previous value, fill with 0
    max_ctend=max_ctend.fillna(0)
    # Add one day to the date
    max_ctend.index = max_ctend.index + pd.Timedelta(days=1)
    # Get the difference between the maximum of tdv between the current maximum and the maximum in the last change of trend
    max_ctend_diff=tdv_5_8_max-max_ctend
    # Remove the nan values
    max_ctend_diff=max_ctend_diff.dropna()

    # Apply a mismatch offset to valdatapd
    stress.index = stress.index + pd.Timedelta(days=mismatch)

    # Convert the indices of tdv_5_8_max, pk, bk, max_ctend_diff and valdatapd to datetime
    tdv_5_8_max.index = pd.to_datetime(tdv_5_8_max.index)
    pk.index = pd.to_datetime(pk.index)
    bk.index = pd.to_datetime(bk.index)
    trend.index = pd.to_datetime(trend.index)
    max_ctend_diff.index = pd.to_datetime(max_ctend_diff.index)
    stress.index = pd.to_datetime(stress.index)

    # Crop the dataframes tdv_5_8_max, pk, bk, bk1, max_ctend_diff and valdatapd to have the same size and indices
    common_index = tdv_5_8_max.index.intersection(pk.index).intersection(bk.index).intersection(max_ctend_diff.index).intersection(stress.index).intersection(trend.index)
    tdv_5_8_max = tdv_5_8_max.loc[common_index]
    pk = pk.loc[common_index]
    bk = bk.loc[common_index]
    trend = trend.loc[common_index]
    max_ctend_diff = max_ctend_diff.loc[common_index]
    stress = stress.loc[common_index]

    # Stack the dataframes
    tdv_max_stack=tdv_5_8_max.stack()
    pk_stack=pk.stack()
    bk_stack=bk.stack()
    bk1_stack=trend.stack()
    ctend_stack=max_ctend_diff.stack()
    data_stack_val=stress.stack()

    # Create a dataframe with the values of tdv_max_stack, pk_stack, bk_stack and ctend_stack as columns
    #data_val=pd.DataFrame({'tdv_max':tdv_max_stack.copy(),'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy()})
    data_val=pd.DataFrame({'pk':pk_stack.copy(),'bk':bk_stack.copy(),'ctend':ctend_stack.copy(),'bk1':bk1_stack.copy()})

    # Crop data_val to the indices of data_stack_val
    data_val=data_val.loc[data_stack_val.index]

    # Add to savedf the values of data_val preserving the same columns
    savedf=pd.concat([savedf,data_val],axis=0)

    # Add to savedf the values of stress as a column "Y", substrating 1 to the values so that the clases are 0, 1 and 2
    savedf["Y"]=data_stack_val-1
    # Remove the rows with nan values
    savedf=savedf.dropna()

# Swap the index levels
savedf=savedf.swaplevel()

# create a string with the last two digits of each year in years
year_datas_str = ''.join(year[-2:] for year in years)

# store savedf in a csv with a name composed of 'TDVdb' followed by the last two digits of each year in year_datas
savedf.to_csv('db\TDVdb'+year_datas_str+'.csv')