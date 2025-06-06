import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Add the implementation/ZIM directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..','implementation/ZIM'))

import functions as f  # Now you can import functions.py

#openmeteo has no data available for 2014 to 2019. I'll use the available data from IRNAS-CSIC to emulate it
# to simplify the script, I'll use the meteoraw database to get the data

#load the data
dataFile='db/ZIMdb14151619meteoraw.csv'

#ZIM
# doPCAs=[False,True,True]
# ns_components=[0,13,15]


# where to save the classifier
classifierFolder='classifiers/trainAll'
classifierFile='classifierTest.clas'

# years to be used as training data
#years_train=[['2014'], ['2015'], ['2016'], ['2019']]
years_train=['2014', '2015', '2016', '2019']

# create the save folder
if not os.path.exists(classifierFolder):
    os.makedirs(classifierFolder)

# Read data file. use the first two columns as index.
origdata = pd.read_csv(dataFile, sep=',', decimal='.', index_col=[0,1])

# crop the data to the years of interest
# extract the year from the second level of the index by separating the string by '-' and taking the first element and place it in a temporary column
origdata['year'] = origdata.index.get_level_values(1).str.split('-').str[0]

# select the years of interest
origdata = origdata.loc[origdata['year'].isin(years_train)]

# drop the temporary column
origdata = origdata.drop(columns='year')

# add '.0' to the column names
origdata.columns = origdata.columns.astype(str) + '.0'
# Split the column names on '.' and create a MultiIndex
origdata.columns = pd.MultiIndex.from_tuples([tuple(col.split('.')) for col in origdata.columns])

#drop the third level of the column index
origdata.columns = origdata.columns.droplevel(2)

# ZIM data has second column index as 0, extract into a separate dataframe
zimdata = origdata.xs('0', axis=1, level=1)

# separate the stress level column ('Y') from the data
stress = zimdata['Y']
zimdata = zimdata.drop(columns='Y')

# extract all meteorological data in order: 'humidity','radiation','temperature','VPD'
humidity = origdata.xs('1', axis=1, level=1)
radiation = origdata.xs('2', axis=1, level=1)
temperature = origdata.xs('3', axis=1, level=1)
VPD = origdata.xs('4', axis=1, level=1)

# get the sunrise time for each sample in the data as the hour where the radiation becomes positive for the first time in the day
sunriseTime = radiation.apply(lambda x: x[x>0].index[0], axis=1)

# get the sunset time for each sample in the data as the hour where the radiation is still positive for the last time in the day
sunsetTime = radiation.apply(lambda x: x[x>0].index[-1], axis=1)

# drop the first level of indices of the data
zimdata.index = zimdata.index.droplevel(0)
stress.index = stress.index.droplevel(0)
humidity.index = humidity.index.droplevel(0)
radiation.index = radiation.index.droplevel(0)
temperature.index = temperature.index.droplevel(0)
VPD.index = VPD.index.droplevel(0)

# turn the sunrise and sunset times into a 1-dimensional array
sunriseTime = sunriseTime.values
sunsetTime = sunsetTime.values

# Process the raw ZIM data
processed_data = f.processRawZIMData(zimdata, sunriseTime, sunsetTime)

# Process the raw meteorological data
processed_humidity = f.processRawMeteoData(humidity, sunriseTime, sunsetTime)
processed_radiation = f.processRawMeteoData(radiation, sunriseTime, sunsetTime)
processed_temperature = f.processRawMeteoData(temperature, sunriseTime, sunsetTime)
processed_VPD = f.processRawMeteoData(VPD, sunriseTime, sunsetTime)

# Combine the processed data into a single DataFrame
processed_data = np.concatenate((processed_data, processed_humidity, processed_radiation, processed_temperature, processed_VPD), axis=1)

# create a dataframe from the processed data recovering the original index
processed_data = pd.DataFrame(processed_data, index=zimdata.index)

# add the stress level to the processed data
processed_data['Y'] = stress

# remove samples with missing values
processed_data = processed_data.dropna()

# separate the stress level column from the data
stress = processed_data['Y']
processed_data = processed_data.drop(columns='Y')

#save the processed data to a file
processed_data.to_csv('db/processed_data.csv')

# train the classifier
classifier = f.trainClassifier(processed_data, stress)

# save the classifier
f.saveModel(classifier, classifierFolder+'/'+classifierFile)