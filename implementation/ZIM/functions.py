import pandas as pd
import numpy as np
from sklearn.svm import SVC
from joblib import dump, load

'''Contains functions for ZIM and meteo data processing'''

def trainClassifier(data:np.ndarray, target:np.ndarray) -> SVC:
    '''Train a classifier using the data and target values
    
    Parameters
    ----------
    data : numpy.ndarray
        The data to train the classifier with. It should be a 2-dimensional array
        where each row represents a sample and each column represents a feature.
    target : numpy.ndarray
        The target values for the data. It should be a 1-dimensional array where
        each element corresponds to the target value for the respective sample in `data`.
    
    Returns
    -------
    SVC
        The trained classifier.
    '''
    # Create a support vector classifier
    clf = SVC(kernel="linear", C=0.025)
    # Train the classifier
    clf.fit(data, target)
    # Return the classifier
    return clf
def saveModel(clf:SVC, path:str):
    '''Save the classifier to a file

    Parameters
    -----------
    clf : SVC
        The classifier to save
    path : str
        The path to save the classifier
    
    Returns
    -------
    None
    '''
    # Save the classifier to a file
    dump(clf, path)

def loadModel(path:str) -> SVC:
    '''Load a classifier from a file
    
    Parameters
    ----------
    path (str): The path to load the classifier from
    
    Returns
    -------
    SVC: The loaded classifier
    '''
    # Load the classifier from a file
    return load(path)

def predict(clf:SVC, data:np.ndarray) -> np.ndarray:
    '''Predict the target values for the data
    
    Parameters
    ----------
    clf (SVC)
        The classifier to use for prediction
    data (numpy array)
        The data to predict the target values for. It should be a 2-dimensional array
        where each row represents a sample and each column represents a feature.
    
    Returns
    -------
    numpy array
        The predicted target values. It is a 1-dimensional array where each element
        corresponds to the predicted target value for the respective sample in `data`.
    '''
    # Predict the target values
    return clf.predict(data)

def predict_proba(clf:SVC, data:np.ndarray) -> np.ndarray:
    '''Predict the probabilities of the target values for the data
    
    Parameters
    ----------
    clf (SVC)
        The classifier to use for prediction
    data (numpy array)
        The data to predict the target values for
    
    Returns
    -------
    numpy array
        The predicted probabilities of the target values
    '''
    # Predict the probabilities of the target values
    return clf.predict_proba(data)

def processRawZIMData(data: pd.DataFrame, sunriseTime: np.ndarray, sunsetTime: np.ndarray, nsamples: int = 80, filterWindow: int = 240) -> np.ndarray:
    '''Process the raw ZIM data
    
    Parameters
    ----------
    data (pd.DataFrame)
        The raw ZIM data. It should be a DataFrame where each row represents a sample (day) and
        each column represents a feature (sensor reading), with the index being the date and the
        columns being the time of day.
    sunriseTime (np.ndarray)
        The sunrise time for each sample in the data. It should be a 1-dimensional array
        where each element corresponds to the sunrise time for the respective sample in `data`.
    sunsetTime (np.ndarray)
        The sunset time for each sample in the data. It should be a 1-dimensional array
        where each element corresponds to the sunset time for the respective sample in `data`.
    nsamples (int)
        The number of samples to generate for each day. Default is 80.
    filterWindow (int)
        The window size for the mean filter in minutes. Default is 240.
    
    Returns
    -------
    np.ndarray
        The processed ZIM data. It is a 2-dimensional array where each row represents a sample
        and each column represents a feature.
    '''

    # make sure the column index is a datetimeindex
    data.columns = pd.to_datetime(data.columns, format="%H:%M:%S")

    # Interpolate the missing values in the data
    data = data.interpolate(method="linear", axis=1)

    # Resample the data to 1 minute intervals so the data is consistently sampled
    data = data.T.resample("1min").interpolate(method="linear").T

    # apply a mean filter to the data (horizontally, along the columns) with a window of 4 hours
    data = data.T.rolling(window=filterWindow, center=True).mean().T

    # Normalize the data by subtracting the mean and dividing by the standard deviation
    data = (data - data.mean(axis=1).to_numpy()[:, np.newaxis]) / data.std(axis=1).to_numpy()[:, np.newaxis]

    # Create an empty array to store the processed data, with size (data.shape[0], nsamples)
    processed_data = np.zeros((data.shape[0], nsamples))

    # Generate the samples for each day
    for i in range(data.shape[0]):
        # Get the sunrise and sunset times for the current day
        sunrise = pd.to_datetime(sunriseTime[i], format="%H:%M:%S")
        sunset = pd.to_datetime(sunsetTime[i], format="%H:%M:%S")

        # crop the data to the sunrise-sunset window
        cropped_data = data.iloc[i].between_time(sunrise.time(), sunset.time())

        # create bins for nsamples sections
        bins = pd.cut(cropped_data.index, bins=nsamples, labels=False)

        # calculate the mean for each bin
        binned_data = cropped_data.groupby(bins).mean()

        # fill the processed_data array with the interpolated data
        processed_data[i] = binned_data.to_numpy().flatten()
    
    return processed_data
def processRawMeteoData(data:pd.DataFrame,sunriseTime:np.ndarray,sunsetTime:np.ndarray,nsamples:int=4,filterWindow:int=240) -> np.ndarray:
    '''Process the raw meteo data
    
    Parameters
    ----------
    data (pd.DataFrame)
        The raw meteo data. It should be a DataFrame where each row represents a sample (day) and
        each column represents a feature (sensor reading), with the index being the date and the
        columns being the time of day.
    sunriseTime (np.ndarray)
        The sunrise time for each sample in the data. It should be a 1-dimensional array
        where each element corresponds to the sunrise time for the respective sample in `data`.
    sunsetTime (np.ndarray)
        The sunset time for each sample in the data. It should be a 1-dimensional array
        where each element corresponds to the sunset time for the respective sample in `data`.
    nsamples (int)
        The number of samples to generate for each day. Default is 4.
    filterWindow (int)
        The window size for the mean filter in minutes. Default is 240.
        
    Returns
    -------
    np.ndarray
        The processed meteo data. It is a 2-dimensional array where each row represents a sample
        and each column represents a feature.
    '''
    # make sure the column index is a datetimeindex
    data.columns = pd.to_datetime(data.columns, format="%H:%M:%S")

    # Interpolate the missing values in the data
    data = data.interpolate(method="linear", axis=1)

    # Resample the data to 1 minute intervals so the data is consistently sampled
    data = data.T.resample("1min").interpolate(method="linear").T

    # apply a mean filter to the data (horizontally, along the columns) with a window of 4 hours
    data = data.T.rolling(window=filterWindow, center=True).mean().T

    # Create an empty array to store the processed data, with size (data.shape[0], nsamples)
    processed_data = np.zeros((data.shape[0], nsamples))

    # Generate the samples for each day
    for i in range(data.shape[0]):
        # Get the sunrise and sunset times for the current day
        sunrise = pd.to_datetime(sunriseTime[i], format="%H:%M:%S")
        sunset = pd.to_datetime(sunsetTime[i], format="%H:%M:%S")

        # crop the data to the sunrise-sunset window
        cropped_data = data.iloc[i].between_time(sunrise.time(), sunset.time())

        # create bins for nsamples sections
        bins = pd.cut(cropped_data.index, bins=nsamples, labels=False)

        # calculate the mean for each bin
        binned_data = cropped_data.groupby(bins).mean()

        # fill the processed_data array with the interpolated data
        processed_data[i] = binned_data.to_numpy().flatten()

    return processed_data

def combineZIMMeteoData(*args:np.ndarray) -> np.ndarray:
    '''Combine the ZIM and meteo data
    
    Parameters
    ----------
    *args (np.ndarray)
        The ZIM and meteo data to combine. Each argument should be a 2-dimensional array
        where each row represents a sample and each column represents a feature.
    
    Returns
    -------
    np.ndarray
        The combined data. It is a 2-dimensional array where each row represents a sample
        and each column represents a feature.
    '''
    # Combine the data by concatenating along the columns
    return np.concatenate(args, axis=1)

def addData(*args:np.ndarray) -> np.ndarray:
    '''Add all the data together to form a single dataset
    
    Parameters
    ----------
    *args (np.ndarray)
        The data to add. Each argument should be a 2-dimensional array
        where each row represents a sample and each column represents a feature.
        Can also be used to add together the target values.
    
    Returns
    -------
    np.ndarray
        The added data. It is a 2-dimensional array where each row represents a sample
        and each column represents a feature, or a 1-dimensional array if the input
        arrays are 1-dimensional such as target values.
    '''
    # Add the data by concatenating new rows
    return np.concatenate(args, axis=0)