import pandas as pd
import numpy as np
from datetime import time
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from tensorflow.keras import layers, models
# import keras_tuner as kt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA

# class CNNHypermodel(kt.HyperModel):
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes

#     def build(self, hp):
#         model = models.Sequential()
#         model.add(layers.Input(shape=self.input_shape))

#         input_size = self.input_shape[0]  # Assuming input_shape is a tuple like (length, channels)

#         for i in range(hp.Int('num_conv_layers', 1, 10)):  # 1 to 10 convolutional layers
#             max_kernel_size = min(3 + (10 - i), input_size)  # Adjust max kernel size based on layer index and current input size
            
#             kernel_size = hp.Int('kernel_size_' + str(i), min(3, max_kernel_size), max_kernel_size)
#             filters = hp.Int('filters_' + str(i), 32, 256, step=32)
            
#             if kernel_size > input_size:
#                 break  # Skip adding more layers if the kernel size exceeds the current input size
            
#             model.add(layers.Conv1D(
#                 filters=filters,
#                 kernel_size=kernel_size,
#                 activation='relu',
#                 padding='same'
#             ))
#             # Optional: Add a conditional pooling layer here
#             # Update input_size based on the convolution and pooling operations
#             model.add(layers.MaxPooling1D(1))            
#             input_size = max(1, (input_size - kernel_size) + 1)  # Simplified calculation, adjust if using strides/padding

        
#         model.add(layers.GlobalMaxPooling1D())
#         model.add(layers.Dense(self.num_classes, activation='softmax'))

#         model.compile(
#             optimizer='adam',
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy']
#         )

#         return model


# Databases
#ZIM 'raw','db/ZIMdb14151619raw.csv',
dataFiles=['db/ZIMdb14151619ZIM80MET4.csv']
dataLabels=['ZIM80MET4']
# dataFiles=['db/ZIMdb14151619oldIRNAS.csv']
# dataLabels=['ZIMindicators']
#TDV
# dataFiles=['db/TDVdb14151619raw.csv', 'db/TDVdb14151619meteoraw.csv']
# #dataFiles=['db/TDVdb14151619meteoraw.csv']
# dataLabels=['TDVraw', 'TDVmeteoraw']

# N days of previous data
ndays=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# test on all years or only on test years
testonall=False

#ZIM
# doPCAs=[False,True,True]
# ns_components=[0,13,15]

#TDV
doPCAs=[True]
ns_components=[13]
doPCAbefore=True

verbose=True
confmat=True

# Report file in results/ZIM/ of root directory, the name of the file is the name of the classifier
reportFolder='results/ZIM/Analysis full/'

# years to be used as training data
#years_train=[['2014'], ['2015'], ['2016'], ['2019']]
years_train=[['2014']]
years_valid=['2015']

#years_test=['2016','2019']
years_test=['2014','2015','2016','2019']

# classifiers to be used
classifiers = [
    # "CNN",
    LinearDiscriminantAnalysis(),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025, random_state=42),
    # SVC(gamma=2, C=1, random_state=42),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    # DecisionTreeClassifier(max_depth=5, random_state=42),
    # RandomForestClassifier(n_estimators=1000, random_state=42),
    # MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    # AdaBoostClassifier(algorithm="SAMME", random_state=42),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
]

# classifier labels
clf_labels = [
    # "CNN",
    "LDA",
    # "Nearest Neighbors",
    # "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    # "Decision Tree",
    # "Random Forest",
    # "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA",
    "BEST"
]

# create the report folder if it does not exist
if not os.path.exists(reportFolder):
    os.makedirs(reportFolder)

# create a string to store the confusion matrices report
confusion_matrices_report = ""

#create a dataframe to store the accuracies. The index is the classifier label. The columns names are constructed in each iteration
accpd=pd.DataFrame(index=clf_labels)
# for each database
for idata in range(len(dataFiles)):
    dataFile=dataFiles[idata]
    dataLabel=dataLabels[idata]

    # for each PCA configuration
    for iPCA in range(len(doPCAs)):
        doPCA=doPCAs[iPCA]
        n_components=ns_components[iPCA]

        # for each n days of previous data
        for nday in ndays:

            # for each year in the training data
            for year_train in years_train:
                if verbose:
                    print("Database: ", dataLabel)
                    print("PCA: ", doPCA)
                    print("n_components: ", n_components)
                    print("Training data: ", year_train)
                    print()
                #create a name for the column
                colname=dataLabel
                if doPCA:
                    colname=colname+str(n_components)+"PCA"
                colname=colname+"Train"+year_train[0]

                #add the n days of previous data to the column name
                colname=colname+"ndays"+str(nday)

                #create a list to store the accuracies
                accs=[]

                # best accuracy obtained
                bestacc=0
                # best classifier
                bestclf=''

                # Read data file. use the first two columns as index.
                data = pd.read_csv(dataFile, sep=',', decimal='.', index_col=[0,1])

                #remove nan rows
                data=data.dropna()

                # separate the data in the Y column
                dataY = data["Y"]
                # drop the Y column
                data.drop("Y", axis=1, inplace=True)

                # if you need to use the PCA before the nday shift, do it here
                if doPCA and doPCAbefore:
                    # transform the y column to integer
                    dataY=dataY.astype(int)

                    # add the Y column back to the data
                    data["Y"]=dataY

                    #remove nan rows
                    data=data.dropna()

                    # # 'Fecha' (second index column) is a string with the format 'YYYY-MM-DD', convert it to datetime
                    # data.index.set_levels(pd.to_datetime(data.index.levels[1]), level=1)
                    # print(data.head())

                    # get the year of each data from the second index (yyyy-mm-dd) splitting by "-"
                    data["year"]=data.index.get_level_values(1).str.split("-").str[0]

                    # select the training data as the data in the year_train list
                    dbtrain=data.loc[data["year"].isin(year_train)].copy()
                    # drop the year column
                    dbtrain.drop("year", axis=1, inplace=True)

                    # extract the y column
                    ytrain=dbtrain["Y"]
                    # drop the y column
                    dbtrain.drop("Y", axis=1, inplace=True)
                    
                    # create the PCA object
                    pca = PCA(n_components=n_components)
                    # fit the PCA object to the training data
                    pca.fit(dbtrain)
                    #store the training data indices
                    index_train=dbtrain.index
                    # transform the training data
                    dbtrain = pca.transform(dbtrain)
                    # add the index back to the training data
                    dbtrain=pd.DataFrame(dbtrain, index=index_train)
                    # remove the Y and year columns from the data, storing them temporarily
                    y = data["Y"]
                    year = data["year"]
                    data.drop(["Y", "year"], axis=1, inplace=True)
                    # also store the index of the data
                    index = data.index
                    # transform the remaining data
                    data = pca.transform(data)
                    # add the index back to the data
                    data = pd.DataFrame(data, index=index)
                    # add the Y and year columns back to the data
                    data["Y"] = y
                    data["year"] = year



                # extend the data with the previous nday days
                # create a temporary dataframe to store the shifted data
                dataToJoin=pd.DataFrame()
                # for each day in the range of nday
                for i in range(1,nday):
                    # create a temporary copy of the data and add i to the column names
                    dataToShift=data.copy()
                    dataToShift.columns=[str(col)+str(i) for col in data.columns]

                    dataToJoin=pd.concat([dataToJoin,dataToShift.shift(i)],axis=1)
                # add the shifted data to the original data
                data=pd.concat([data,dataToJoin],axis=1)

                # transform the y column to integer
                dataY=dataY.astype(int)

                # add the Y column back to the data
                data["Y"]=dataY

                #remove nan rows
                data=data.dropna()

                # # 'Fecha' (second index column) is a string with the format 'YYYY-MM-DD', convert it to datetime
                # data.index.set_levels(pd.to_datetime(data.index.levels[1]), level=1)
                # print(data.head())

                # get the year of each data from the second index (yyyy-mm-dd) splitting by "-"
                data["year"]=data.index.get_level_values(1).str.split("-").str[0]

                # select the training data as the data in the year_train list
                dbtrain=data.loc[data["year"].isin(year_train)].copy()
                # drop the year column
                dbtrain.drop("year", axis=1, inplace=True)

                # extract the y column
                ytrain=dbtrain["Y"]
                # drop the y column
                dbtrain.drop("Y", axis=1, inplace=True)

                if doPCA and not doPCAbefore:
                    # create the PCA object
                    pca = PCA(n_components=n_components)
                    # fit the PCA object to the training data
                    pca.fit(dbtrain)
                    #store the training data indices
                    index_train=dbtrain.index
                    # transform the training data
                    dbtrain = pca.transform(dbtrain)
                    # add the index back to the training data
                    dbtrain=pd.DataFrame(dbtrain, index=index_train)
                    # remove the Y and year columns from the data, storing them temporarily
                    y = data["Y"]
                    year = data["year"]
                    data.drop(["Y", "year"], axis=1, inplace=True)
                    # also store the index of the data
                    index = data.index
                    # transform the remaining data
                    data = pca.transform(data)
                    # add the index back to the data
                    data = pd.DataFrame(data, index=index)
                    # add the Y and year columns back to the data
                    data["Y"] = y
                    data["year"] = year

                # generate the validation data
                dbvalid=data.loc[data["year"].isin(years_valid)].copy()
                # drop the year column
                dbvalid.drop("year", axis=1, inplace=True)

                # extract the y column
                yvalid=dbvalid["Y"]
                # drop the y column
                dbvalid.drop("Y", axis=1, inplace=True)

                # train the classifiers
                for clf in classifiers:

                    if verbose:
                        print(clf)

                    # if the classifier is the CNN
                    if clf_labels[classifiers.index(clf)]=="CNN":
                        #update the input shape
                        # input_shape = (dbtrain.shape[1], 1)
                        # num_classes = len(ytrain.unique())
                        # cnnmodel = CNNHypermodel(input_shape=input_shape, num_classes=num_classes)

                        # # Create the tuner
                        # tuner = kt.tuners.RandomSearch(
                        #     hypermodel=cnnmodel,
                        #     objective='val_accuracy',
                        #     max_trials=10,
                        #     directory='results/ZIM/tuner'+dataLabel,
                        #     project_name='CNN'
                        # )
                        # tuner.search(dbtrain, ytrain, epochs=10, validation_data=(dbvalid, yvalid))

                        # # Create the best model
                        # best_hps = tuner.get_best_hyperparameters(num_trials=5)[0]
                        # model = cnnmodel.build(best_hps)
                        # model.fit(dbtrain, ytrain, epochs=10)
                    # else, train the classifier directly
                        pass
                    else:
                        dbtrain.columns = dbtrain.columns.astype(str)
                        dbvalid.columns = dbvalid.columns.astype(str)
                        #dbtest.columns = dbtest.columns.astype(str)
                        clf.fit(dbtrain, ytrain)

                    ytestFull = []
                    ypredFull = []

                    # for each year in the data
                    for year in data["year"].unique():
                        if testonall or (year in years_test):
                            # select the data of the year
                            dbtest=data.loc[data["year"]==year].copy()
                            # drop the year column
                            dbtest.drop("year", axis=1, inplace=True)
                            # extract the y column
                            ytest=dbtest["Y"]
                            # drop the y column
                            dbtest.drop("Y", axis=1, inplace=True)

                            # if the classifier is the CNN, select the predicted class as the one with the highest probability
                            if clf_labels[classifiers.index(clf)]=="CNN":
                                # ypred=np.argmax(model.predict(dbtest), axis=1)
                                # # convert the result into a pandas series with the same index as the ytest
                                # ypred=pd.Series(ypred, index=ytest.index)
                                pass

                            # else, predict the classes directly
                            else:
                                dbtest.columns = dbtest.columns.astype(str)
                                ypred=clf.predict(dbtest)

                            # if it's set to use all years as test data or the year isn't on the training data, save the y values and the predicted y values
                            ytestFull.extend(ytest)
                            ypredFull.extend(ypred)

                            # calculate the confusion matrix
                            if confmat:
                                cm = confusion_matrix(ytest, ypred)

                                # save the confusion matrix to a csv file
                                pd.DataFrame(cm).to_csv(reportFolder+"confusion_matrix_"+dataLabel+"_"+str(year)+"_"+clf_labels[classifiers.index(clf)]+"ndays"+str(nday)+".csv")
                            

                            # show the predicted values and the real values
                            if verbose:
                                print("Year: ", year)
                                print("Real values: ", ytest)
                                print("Predicted values: ", ypred)
                                print()
                        

                    curracc=balanced_accuracy_score(ytestFull, ypredFull)

                    if verbose:
                        cmfull = confusion_matrix(ytestFull, ypredFull)
                        print("confusion matrix: ")
                        print(cmfull)
                        print("Balanced accuracy: ", curracc)
                        print()

                    if curracc>bestacc:
                        bestacc=curracc
                        bestclf=clf_labels[classifiers.index(clf)]
                    
                    accs.append(curracc)

                # add the best accuracy to the list of accuracies
                accs.append(bestacc)

                # add the accuracies to the dataframe
                accpd[colname]=accs
                
                if verbose:
                    print("Best accuracy: ", bestacc)
                    print("Best classifier: ", bestclf)
                    print()
                    print("Accuracies: ")
                    print(accpd)

    # save the accuracies to a csv file
    accpd.to_csv(reportFolder+"accuraciesZIMespanol.csv")

    #save the confusion matrices to a text file
