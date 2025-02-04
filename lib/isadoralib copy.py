import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
import qpsolvers as qp
from scipy.optimize import minimize
from scipy.optimize import linprog

def cargaDatos(year,sufix):
    '''Load data corresponding to a year stored in the files [year][sufix].csv and validacion[year].csv and returns a tuple (tdv,ltp,meteo,hidric stress level).'''
    # Load data
    df = pd.read_csv("rawMinutales"+year+sufix+".csv",na_values='.')
    df.loc[:,"Fecha"]=pd.to_datetime(df.loc[:,"Fecha"])# Date as datetime
    df=df.drop_duplicates(subset="Fecha")
    df.dropna(subset = ["Fecha"], inplace=True)
    df=df.set_index("Fecha")
    df=df.apply(pd.to_numeric, errors='coerce')

    # split dfT into tdv and ltp depending on the beginning of the name of each column and save the rest in meteo
    tdv = df.loc[:,df.columns.str.startswith('TDV')]
    ltp = df.loc[:,df.columns.str.startswith('LTP')]
    meteo = df.drop(df.columns[df.columns.str.startswith('TDV')], axis=1)
    meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

    # Load validation data
    valdatapd=pd.read_csv("validacion"+year+".csv")
    valdatapd.dropna(inplace=True)
    valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
    valdatapd.set_index('Fecha',inplace=True)

    return (tdv,ltp,meteo,valdatapd)

def cargaDatosTDV(year,sufix):
    '''Load data corresponding to a year stored in the files [year][sufix].csv and validacion[year].csv and returns a tuple (tdv,ltp,meteo,hidric stress level).'''
    # Load data
    df = pd.read_csv("rawMinutales"+year+sufix+".csv",na_values='.')
    df.loc[:,"Fecha"]=pd.to_datetime(df.loc[:,"Fecha"])# Date as datetime
    df=df.drop_duplicates(subset="Fecha")
    df.dropna(subset = ["Fecha"], inplace=True)
    df=df.set_index("Fecha")
    df=df.apply(pd.to_numeric, errors='coerce')

    # split dfT into tdv and ltp depending on the beginning of the name of each column and save the rest in meteo
    tdv = df.loc[:,df.columns.str.startswith('TDV')]
    ltp = df.loc[:,df.columns.str.startswith('LTP')]
    meteo = df.drop(df.columns[df.columns.str.startswith('TDV')], axis=1)
    meteo = meteo.drop(meteo.columns[meteo.columns.str.startswith('LTP')], axis=1)

    # Load validation data
    valdatapd=pd.read_csv("validacion"+year+"TDV.csv")
    #valdatapd.dropna(inplace=True)
    valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
    valdatapd.set_index('Fecha',inplace=True)

    return (tdv,ltp,meteo,valdatapd)

def datosADataframe(ltp:pd.DataFrame,meteo:pd.DataFrame,valdatapd:pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    '''Save ltp and meteo data in a dataframe x and valdata in a series y with the proper shape to convert them to numpy arrays for scikit or to continue processing them. X and Y are not reduced to common columns.'''
    ltp['Dia'] = pd.to_datetime(ltp.index).date
    ltp['Delta'] = pd.to_datetime(ltp.index) - pd.to_datetime(ltp.index).normalize()


    meteo['Dia'] = pd.to_datetime(meteo.index).date
    meteo['Delta'] = pd.to_datetime(meteo.index) - pd.to_datetime(meteo.index).normalize()

    # ltpPdia = ltpP.loc[meteoP['R_Neta_Avg']>0]

    ltp=ltp.set_index(['Dia','Delta']).unstack(0)
    meteo=meteo.set_index(['Dia','Delta']).unstack(0).stack(0)
    valdatapd=valdatapd.unstack()

    #common_col = ltp.columns.intersection(valdatapd.index)
    #ltp=ltp[common_col]
    y=valdatapd#[common_col]

    meteoPext=pd.DataFrame(columns=ltp.columns)
    for col in meteoPext:
        meteoPext[col]=meteo[col[1]]
    x=meteoPext.unstack(0)
    x.loc['LTP']=ltp.unstack(0)
    print(x)
    x=x.stack(2)
    return (x, y)


#KRIGGING

class KriggingClassifier:
    '''Krigging classifier object. The matrix Xtrain contains the training database, with a row for each feature and a column for each sample. The vector ytrain contains the classes of each sample.'''
    # Constructor
    def __init__(self, Xtrain, alpha, ytrain):
        self.Xtrain = Xtrain
        self.alpha = alpha
        if ytrain is not None:
            self.ytrain = ytrain
        else:
            self.ytrain = np.zeros(Xtrain.shape[0])
        self.num_classes = np.unique(ytrain).shape[0]
        # get the indices of the classes in the training set
        self.indices = [np.where(self.ytrain == i)[0] for i in range(len(np.unique(self.ytrain)))]
        
        self.update_matrices(Xtrain, alpha)

    def update_matrices(self, Xtrain, alpha):
        '''Updates the matrices P, q, G, h and A for the training data Xtrain and the parameter alpha.'''
        # Get P matrix as a square matrix of size 2*N+1 with a diagonal matrix of size N and value 2 in the upper left corner
        self.P = np.zeros([2*Xtrain.shape[1], 2*Xtrain.shape[1]])
        self.P[:Xtrain.shape[1], :Xtrain.shape[1]] = np.eye(Xtrain.shape[1])*2
        # Matrix P as a sparse matrix
        self.P = sp.csc_matrix(self.P)

        # Get q vector of size 2*N+1 with alpha values in the last N elements
        self.q = np.zeros([2*Xtrain.shape[1]])
        self.q[Xtrain.shape[1]:2*Xtrain.shape[1]] = alpha

        # Get G matrix of size 2*N+1 x 2*N with four identity matrices of size N and a column of zeros. All identity matrices have negative sign except the upper right corner
        self.G = np.zeros([2*Xtrain.shape[1], 2*Xtrain.shape[1]])
        self.G[:Xtrain.shape[1], :Xtrain.shape[1]] = np.eye(Xtrain.shape[1])
        self.G[Xtrain.shape[1]:2*Xtrain.shape[1], Xtrain.shape[1]:2*Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[Xtrain.shape[1]:2*Xtrain.shape[1], :Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        self.G[:Xtrain.shape[1], Xtrain.shape[1]:2*Xtrain.shape[1]] = -np.eye(Xtrain.shape[1])
        # G as a sparse matrix
        self.G = sp.csc_matrix(self.G)

        # Get h vector of size 2*N with zero values
        self.h = np.zeros([2*Xtrain.shape[1]])

        # Get A matrix of size M+1 x 2*N with the training data matrix and a row of zeros and a 1 in the last column
        self.A = np.zeros([Xtrain.shape[0]+1, 2*Xtrain.shape[1]])
        self.A[:Xtrain.shape[0], :Xtrain.shape[1]] = Xtrain
        self.A[Xtrain.shape[0], :Xtrain.shape[1]] = 1
        # A as a sparse matrix
        self.A = sp.csc_matrix(self.A)

    def update_ytrain(self, ytrain):
        '''Updates the vector ytrain of the training data.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]
        self.indices = [np.where(self.ytrain == i)[0] for i in range(len(np.unique(self.ytrain)))]

        

    def apply(self,x):
        '''Applies the classifier to a feature vector x. Returns the value of the objective function and the vector of lambdas.'''
        # Create an extended feature vector with a 1
        b = np.hstack([x, 1])

        # Get T that minimizes the QP function using OSQP
        T = qp.solve_qp(self.P, self.q.T, self.G, self.h, self.A, b, solver='osqp')

        P= self.P.toarray()

        # check if T is None
        if T is None:
            # set the value of the objective function to infinity
            f = np.inf
            # set the vector of lambdas of the training data to None
            lambda_i = None
        else:
            # get the value of the objective function
            f = 0.5*np.dot(T.T, np.dot(P, T)) + np.dot(self.q, T)

            # Get the vector of lambdas of the training data
            lambda_i = T[:self.Xtrain.shape[1]]
            t_i = T[self.Xtrain.shape[1]:]
        return (f, lambda_i)
    
    def lambda_classifier(self, x):
        '''Apply the classifier to a feature vector x. Returns the predicted class based on the lambda vector. If no class can be predicted, returns a class greater than the number of training classes.'''
        # Apply the classifier to x
        y_pred_lambda = self.apply(x)[1]
        # if lambda is None:
        if y_pred_lambda is None:
            # assign a class greater than the number of classes
            y_pred_lambda = self.num_classes
        else:
            # split the lambda values by classes
            y_pred_lambda = [y_pred_lambda[self.indices[j]] for j in range(len(np.unique(self.ytrain)))]
            # get the sum of the lambda values by classes
            y_pred_lambda = [np.sum(y_pred_lambda[j]) for j in range(len(np.unique(self.ytrain)))]
            # select the class with the greatest lambda value
            y_pred_lambda = np.argmax(y_pred_lambda)
        return y_pred_lambda

    def minTraining(self):
        '''Calculates the minimum number of training samples that can be used to train the classifier. Testing showed it is not useful and will be deprecated.'''
        # get the number of features from the training data
        n = self.Xtrain.shape[1]

        # get the number of classes from the training data
        nclases = np.unique(self.ytrain).shape[0]

        # For each class, calculate the number of training samples
        ntrain = np.zeros(nclases)
        for i in range(nclases):
            ntrain[i] = np.where(self.ytrain == i)[0].shape[0]
        
        # get the number of training samples
        m = self.Xtrain.shape[0]


        print("minTraining is not useful and will be deprecated. Use of this function is not necessary as the classifier should work by itself.")

class KriggingFunctionClassifier:
    '''Krigging classifier object based on the value of the objective function. Krigging classifier object. The matrix Xtrain contains the training database, with a row for each feature and a column for each sample. The vector ytrain contains the classes of each sample.'''
    
    def __init__(self, Xtrain, alpha, ytrain=None):
        '''Class constructor. Receives the training data and the value of alpha.'''
        # Store the training data
        self.Xtrain = Xtrain
        if ytrain is None:
            self.ytrain = np.zeros(Xtrain.shape[0])
        # Store alpha value
        self.alpha = alpha
        # Update ytrain vector
        self.update_ytrain(ytrain)

    def update_ytrain(self, ytrain):
        '''Updates ytrain vector with the training data classes.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]

        self.kriggings = []
        # create a krigging classifier for each class
        for i in range(self.num_classes):
            # get the training data for class i
            Xtrain_i = self.Xtrain[:,np.where(self.ytrain == i)[0]]
            ytrain_i = self.ytrain[np.where(self.ytrain == i)[0]]
            # create a krigging classifier in the list of classifiers
            self.kriggings.append(KriggingClassifier(Xtrain_i, self.alpha, ytrain_i))
        
    def fun_classifier(self,x):
        '''Applies the classifier to a feature vector x. Returns the predicted class based on the value of the objective function.'''
        # apply the classifier to x
        y_pred_fun = [self.kriggings[i].apply(x)[0] for i in range(self.num_classes)]
        # select the class with the lowest value of the objective function
        y_pred_fun = np.argmin(y_pred_fun)
        # if y_pred_fun is infinite, assign a class greater than the number of classes
        if y_pred_fun == np.inf:
            y_pred_fun = self.num_classes
        return y_pred_fun
    
class KriggingQDA:
    '''Emulates QDA using Kriging for the Mahalanobis distance calculation. It should be equivalent to regular QDA when the value of alpha is 0.'''

    def __init__(self, Xtrain, alpha, ytrain=None):
        # Store the training data
        self.Xtrain = Xtrain
        if ytrain is None:
            self.ytrain = np.zeros(Xtrain.shape[0])
        # Store alpha value
        self.alpha = alpha
        # Update ytrain vector
        self.update_ytrain(ytrain)
    
    def update_ytrain(self, ytrain):
        '''Updates ytrain vector with the training data classes.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]
        
        self.kriggings = []
        # create a krigging classifier for each class
        for i in range(self.num_classes):
            # get the training data for class i
            Xtrain_i = self.Xtrain[:,np.where(self.ytrain == i)[0]]
            ytrain_i = self.ytrain[np.where(self.ytrain == i)[0]]
            # create a krigging classifier in the list of classifiers
            self.kriggings.append(KriggingClassifier(Xtrain_i, self.alpha, ytrain_i))
        
        self.N=self.Xtrain.shape[1]
        #number of samples in each class
        self.Nk=[self.Xtrain[:,np.where(self.ytrain == i)[0]].shape[1] for i in range(self.num_classes)]
        self.CovMatrices=[]
        self.CovMatDet=[]
        self.PriorProb=[]
        for i in range(self.num_classes):
            self.CovMatrices.append(np.cov(self.Xtrain[:,np.where(self.ytrain == i)[0]]))
            self.CovMatDet.append(np.linalg.det(self.CovMatrices[i]))
            self.PriorProb.append(self.Xtrain[:,np.where(self.ytrain == i)[0]].shape[1]/self.Xtrain.shape[1])


        self.AvgX=[]
        for i in range(self.num_classes):
            self.AvgX.append(np.mean(self.Xtrain[:,np.where(self.ytrain == i)[0]],axis=1))

        
    def qda_classifier(self,x):
        '''Applies the classifier to a feature vector x. Returns the predicted class based on the value of the objective function.'''
        # apply the classifier to x
        y_pred_fun = [self.kriggings[i].apply(x)[0] for i in range(self.num_classes)]

        #print([np.divide(-np.dot(self.Nk[i]/2,y_pred_fun[i]),np.dot(np.dot((x-self.AvgX[i]).T,np.linalg.inv(self.CovMatrices[i])),(x-self.AvgX[i]))) for i in range(self.num_classes)])
        
        # adjust the value of the objective function so that it is equivalent to QDA
        y_pred_fun_qda=[-np.dot(self.Nk[i]/2,y_pred_fun[i])+np.log(self.PriorProb[i])-1/2*np.log(self.CovMatDet[i]) for i in range(self.num_classes)]
        # select the class with the highest value of the objective function
        y_pred_fun_qda = np.argmax(y_pred_fun_qda)
        return y_pred_fun_qda
    
    def qda_classifier_prob(self,x):
        '''Applies the classifier to a feature vector x. Returns the probability of each class.'''
        # apply the classifier to x
        y_pred_fun = [self.kriggings[i].apply(x)[0] for i in range(self.num_classes)]
        # adjust the value of the objective function so that it is equivalent to QDA
        y_pred_fun_qda=[-self.N/2*y_pred_fun[i]+np.log(self.PriorProb[i])-1/2*np.log(self.CovMatDet[i]) for i in range(self.num_classes)]
        # calculate the probability of each class
        P_class=[np.exp(y_pred_fun_qda[i])/np.sum(np.exp(y_pred_fun_qda)) for i in range(self.num_classes)]
        return P_class        
    
class qdaClassifier:
    '''Implements basic QDA'''

    def __init__(self, Xtrain, ytrain=None):
        # Store the training data
        self.Xtrain = Xtrain
        if ytrain is None:
            self.ytrain = np.zeros(Xtrain.shape[0])
        # Update ytrain vector
        self.update_ytrain(ytrain)

    def update_ytrain(self, ytrain):
        '''Updates ytrain vector with the training data classes.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]
        
        self.N=self.Xtrain.shape[1]
        self.CovMatrices=[]
        self.CovMatDet=[]
        self.PriorProb=[]
        self.AvgX=[]
        for i in range(self.num_classes):
            self.CovMatrices.append(np.cov(self.Xtrain[:,np.where(self.ytrain == i)[0]]))
            self.CovMatDet.append(np.linalg.det(self.CovMatrices[i]))
            self.PriorProb.append(self.Xtrain[:,np.where(self.ytrain == i)[0]].shape[1]/self.Xtrain.shape[1])
            self.AvgX.append(np.mean(self.Xtrain[:,np.where(self.ytrain == i)[0]],axis=1))

    def qda_classifier(self,x):
        '''Applies the classifier to a feature vector x. Returns the predicted class based on the value of the objective function.'''
        mahalanobis=[]
        for i in range(self.num_classes):
            mahalanobis.append(np.dot(np.dot((x-self.AvgX[i]).T,np.linalg.inv(self.CovMatrices[i])),(x-self.AvgX[i])))
        # calculate the proportional probability of each class
        y_pred_fun_qda=[-1/2*mahalanobis[i]+np.log(self.PriorProb[i])-1/2*np.log(self.CovMatDet[i]) for i in range(self.num_classes)]
        # select the class with the highest value of the objective function
        y_pred_fun_qda = np.argmax(y_pred_fun_qda)
        return y_pred_fun_qda
    
class KrigBayesian:
    '''Our proposed Bayesian-based Kriging classifier. It is equivalent to QDA when using default values.'''
    def __init__(self, Xtrain,krig_lambda=0, alphak=None, Fk=None, ytrain=None):
        # Store the training data
        self.Xtrain = Xtrain
        if ytrain is None:
            self.ytrain = np.zeros(Xtrain.shape[0])
        # Store alpha value
        self.alphak = alphak
        # Store F value
        self.Fk = Fk
        # Store lambda value
        self.krig_lambda = krig_lambda
        # Update ytrain vector
        self.update_ytrain(ytrain)

    def update_ytrain(self, ytrain,Fk=None):
        '''Updates ytrain vector with the training data classes.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]
        
        self.kriggings = []
        # create a krigging classifier for each class
        for i in range(self.num_classes):
            # get the training data for class i
            Xtrain_i = self.Xtrain[:,np.where(self.ytrain == i)[0]]
            ytrain_i = self.ytrain[np.where(self.ytrain == i)[0]]
            # create a krigging classifier in the list of classifiers
            self.kriggings.append(KriggingClassifier(Xtrain_i, self.krig_lambda, ytrain_i))
        
        self.N=self.Xtrain.shape[1]
        #number of samples in each class
        self.Nk=[self.Xtrain[:,np.where(self.ytrain == i)[0]].shape[1] for i in range(self.num_classes)]
        self.CovMatrices=[]
        self.CovMatDet=[]
        self.PriorProb=[]
        for i in range(self.num_classes):
            self.CovMatrices.append(np.cov(self.Xtrain[:,np.where(self.ytrain == i)[0]]))
            self.CovMatDet.append(np.linalg.det(self.CovMatrices[i]))
            self.PriorProb.append(self.Xtrain[:,np.where(self.ytrain == i)[0]].shape[1]/self.Xtrain.shape[1])

        self.AvgX=[]
        for i in range(self.num_classes):
            self.AvgX.append(np.mean(self.Xtrain[:,np.where(self.ytrain == i)[0]],axis=1))
        
        if self.alphak is None:
            self.alphak=[self.Nk[i]/2 for i in range(self.num_classes)]

        if self.Fk is None:
            # Fk is calculated as 1/(2*pi^(N/2)*sqrt(CovMatDet[i]))*e^(1/2)
            self.Fk=[1/(2*np.pi**(self.N/2)*np.sqrt(self.CovMatDet[i]))*np.exp(1/2) for i in range(self.num_classes)]
        

    def class_prob(self,x):
        '''Applies the classifier to a feature vector x. Returns the probability of each class.'''
        # apply the classifier to x
        y_pred_fun = [self.kriggings[i].apply(x)[0] for i in range(self.num_classes)]

        #print([np.divide(-np.dot(self.Nk[i]/2,y_pred_fun[i]),np.dot(np.dot((x-self.AvgX[i]).T,np.linalg.inv(self.CovMatrices[i])),(x-self.AvgX[i]))) for i in range(self.num_classes)])
        
        # calculate P=Fk*exp(-alphak*y_pred_funk)
        Prob=[self.Fk[i]*np.exp(-self.alphak[i]*y_pred_fun[i]) for i in range(self.num_classes)]
        # calculate the probability of each class
        Prob=[Prob[i]/np.sum(Prob) for i in range(self.num_classes)]
        return Prob
        
    def classify(self,x):
        '''Applies the classifier to a feature vector x. Returns the predicted class based on the value of the objective function.'''
        # apply the classifier to x
        Prob=self.class_prob(x)
        # select the class with the highest value of the objective function
        y_pred = np.argmax(Prob)
        return y_pred
    
class KrigOpt:
    '''Our proposed Bayesian-based Kriging classifier with optimized F, naive approach.'''
    def __init__(self, Xtrain,krig_lambda=0, alphak=None, Fk=None, ytrain=None):
        # Store the training data
        self.Xtrain = Xtrain
        if ytrain is None:
            self.ytrain = np.zeros(Xtrain.shape[0])
        # Store alpha value
        self.alphak = alphak
        # Store F value
        self.Fk = Fk
        # Store lambda value
        self.krig_lambda = krig_lambda
        # Update ytrain vector
        self.update_ytrain(ytrain)

    def update_ytrain(self, ytrain,Fk=None):
        '''Updates ytrain vector with the training data classes.'''
        self.ytrain = ytrain
        self.num_classes = np.unique(ytrain).shape[0]
        
        self.kriggings = []
        # create a krigging classifier for each class
        for i in range(self.num_classes):
            # get the training data for class i
            Xtrain_i = self.Xtrain[:,np.where(self.ytrain == i)[0]]
            ytrain_i = self.ytrain[np.where(self.ytrain == i)[0]]
            # create a krigging classifier in the list of classifiers
            self.kriggings.append(KriggingClassifier(Xtrain_i, self.krig_lambda, ytrain_i))
        
        self.N=self.Xtrain.shape[1]
        #number of samples in each class
        self.Nk=[self.Xtrain[:,np.where(self.ytrain == i)[0]].shape[1] for i in range(self.num_classes)]
        self.CovMatrices=[]
        self.CovMatDet=[]
        self.PriorProb=[]
        for i in range(self.num_classes):
            self.CovMatrices.append(np.cov(self.Xtrain[:,np.where(self.ytrain == i)[0]]))
            self.CovMatDet.append(np.linalg.det(self.CovMatrices[i]))
            self.PriorProb.append(self.Xtrain[:,np.where(self.ytrain == i)[0]].shape[1]/self.Xtrain.shape[1])

        self.AvgX=[]
        for i in range(self.num_classes):
            self.AvgX.append(np.mean(self.Xtrain[:,np.where(self.ytrain == i)[0]],axis=1))
        
        if self.alphak is None:
            self.alphak=[self.Nk[i]/2 for i in range(self.num_classes)]

        if self.Fk is None:
            # Fk is calculated initially as 1/(2*pi^(N/2)*sqrt(CovMatDet[i]))*e^(1/2)
            self.Fk=[1/(2*np.pi**(self.N/2)*np.sqrt(self.CovMatDet[i]))*np.exp(1/2) for i in range(self.num_classes)]

        Probs=[]
        # for every item in the training data
        for x in self.Xtrain.T:
            # calculate the probability of each class
            Prob=self.class_prob(x)
            # store the probabilities
            Probs.append(Prob)

        yProbs=[]
        # for every item in the training data classes
        for y in self.ytrain:
            # create an array with length equal to the number of classes and a 1 in the position of the class
            yProb=np.zeros(self.num_classes)
            yProb[y]=1
            # store the array
            yProbs.append(yProb)
        Fkfactor=[1 for i in range(self.num_classes)]
        #optimize Fkfactor
        res=minimize(self.correctError,Fkfactor,args=(Probs,yProbs),method='Nelder-Mead')

        #print the optimized factor
        print(res.x)

        # update Fk with the optimized value
        self.Fk=[self.Fk[i]*res.x[i] for i in range(self.num_classes)]



    def correctError(self,Fkfactor, Probs, yProbs):

        # for every item in Probs, multiply it by Fkfactor
        for i in range(len(Probs)):
            Probs[i]=[Probs[i][j]*Fkfactor[j] for j in range(self.num_classes)]
        
            # for every item in Probs, normalize it
            Probs[i]=[Probs[i][j]/np.sum(Probs[i]) for j in range(self.num_classes)]
        
        # calculate the error as the sum of the squared differences between the probabilities and the classes
        error=np.sum(np.square(np.subtract(Probs,yProbs)))
        return error

    def class_prob(self,x):
        '''Applies the classifier to a feature vector x. Returns the probability of each class.'''
        # apply the classifier to x
        y_pred_fun = [self.kriggings[i].apply(x)[0] for i in range(self.num_classes)]

        #print([np.divide(-np.dot(self.Nk[i]/2,y_pred_fun[i]),np.dot(np.dot((x-self.AvgX[i]).T,np.linalg.inv(self.CovMatrices[i])),(x-self.AvgX[i]))) for i in range(self.num_classes)])
        
        # calculate P=Fk*exp(-alphak*y_pred_funk)
        Prob=[self.Fk[i]*np.exp(-self.alphak[i]*y_pred_fun[i]) for i in range(self.num_classes)]
        # calculate the probability of each class
        Prob=[Prob[i]/np.sum(Prob) for i in range(self.num_classes)]
        return Prob
        
    def classify(self,x):
        '''Applies the classifier to a feature vector x. Returns the predicted class based on the value of the objective function.'''
        # apply the classifier to x
        Prob=self.class_prob(x)
        # select the class with the highest value of the objective function
        y_pred = np.argmax(Prob)
        return y_pred
    
class DisFunClass:
    '''Our proposed dissimilarity function classifier with optimized F and c.'''
    def __init__(self, Xtrain, ytrain, Xcal=None, ycal=None, gam=0, ck=None, Fk=None, ClassProb=None):
        """
        Initialize the IsadoraLib class.

        Parameters:
        - Xtrain: numpy array, training data.
        - ytrain: numpy array, training labels.
        - gam: float, gamma value.
        - ck: None or float, ck value.
        - Fk: None or float, Fk value.
        - ClassProb: None or numpy array, class probabilities.

        Returns:
        None
        """
        # Store the training data
        self.Xtrain = Xtrain
        if ytrain is None:
            self.ytrain = np.zeros(Xtrain.shape[0])
        else:
            self.ytrain = ytrain

        # Store calibrating data
        if Xcal is None:
            self.Xcal = Xtrain
        else:
            self.Xcal = Xcal
        if ycal is None:
            self.ycal = ytrain
        else:
            self.ycal = ycal

        # Store ck value
        self.ck = ck
        # Store Fk value
        self.Fk = Fk
        # Store gamma value
        self.gam = gam

        # Calculate the class probabilities if not provided
        if ClassProb is None:
            self.ClassProb=self.calculateClassProb()
        else:
            self.ClassProb=ClassProb
        
        # For each class, get the difference between the log of its class probability and the log of each other class probability
        self.prk=[[np.log(self.ClassProb[i])-np.log(self.ClassProb[j]) for j in range(np.unique(ytrain).shape[0])] for i in range(np.unique(ytrain).shape[0])]
        
        # Separate the training data by classes
        self.Dk=[Xtrain[:,np.where(ytrain == i)[0]] for i in range(np.unique(ytrain).shape[0])]

        # Separate the calibrating data by classes
        self.Dkcal=[self.Xcal[:,np.where(self.ycal == i)[0]] for i in range(np.unique(self.ycal).shape[0])]

        if (self.ck is None or self.Fk is None):
            self.calibrateCF()
    
    def calculateClassProb(self):
        '''Calculates the class probabilities from the training data.'''
        # get the number of classes
        nclases = np.unique(self.ytrain).shape[0]
        # calculate the class probabilities
        ClassProb=[np.sum(self.ytrain == i)/self.ytrain.shape[0] for i in range(nclases)]
        return ClassProb
    
    def getJ(self,Dk,x):
        '''Updates the matrices P, q, G, h and A for the training data Dk and the parameter gamma.'''
        # Get P matrix as a square matrix of size 2*N with a diagonal matrix of size N and value 2 in the upper left corner
        P = np.zeros([2*Dk.shape[1], 2*Dk.shape[1]])
        P[:Dk.shape[1], :Dk.shape[1]] = np.eye(Dk.shape[1])*2
        # Matrix P as a sparse matrix
        P = sp.csc_matrix(P)

        # Get q vector of size 2*N+1 with gamma values in the last N elements
        q = np.zeros([2*Dk.shape[1]])
        q[Dk.shape[1]:2*Dk.shape[1]] = self.gam

        # Get G matrix of size 2*N x 2*N with four identity matrices of size N. All identity matrices have negative sign except the upper left corner
        G = np.zeros([2*Dk.shape[1], 2*Dk.shape[1]])
        G[:Dk.shape[1], :Dk.shape[1]] = np.eye(Dk.shape[1])
        G[Dk.shape[1]:2*Dk.shape[1], Dk.shape[1]:2*Dk.shape[1]] = -np.eye(Dk.shape[1])
        G[Dk.shape[1]:2*Dk.shape[1], :Dk.shape[1]] = -np.eye(Dk.shape[1])
        G[:Dk.shape[1], Dk.shape[1]:2*Dk.shape[1]] = -np.eye(Dk.shape[1])
        # G as a sparse matrix
        G = sp.csc_matrix(G)

        # Get h vector of size 2*N with zero values
        h = np.zeros([2*Dk.shape[1]])

        # Get A matrix of size M+1 x 2*N with the training data matrix and a row of zeros and a 1 in the last column
        A = np.zeros([Dk.shape[0]+1, 2*Dk.shape[1]])
        A[:Dk.shape[0], :Dk.shape[1]] = Dk
        A[Dk.shape[0], :Dk.shape[1]] = 1
        # A as a sparse matrix
        A = sp.csc_matrix(A)
        
        # Create an extended feature vector with a 1
        b = np.hstack([x, 1])

        # Get T that minimizes the QP function using OSQP
        T = qp.solve_qp(P, q.T, G, h, A, b, solver='osqp')

        #check if T is None
        if T is None:
            # set the value of the objective function to infinity
            jx = np.inf
        else:
            # calculate the value of the objective function
            jx = 0.5*np.dot(T, np.dot(P.toarray(), T.T)) + np.dot(q.T, T)
        return jx
    
    def calibrateCF(self):
        '''Calibrates the values of c and F.'''
        # For each class, calculate the value of the objective function for each calibration sample over the training set; samples are stored as the columns of the matrix
        Jkx=[[self.getJ(self.Dk[i],self.Xcal[:,j]) for j in range(self.Xcal.shape[1])] for i in range(len(self.Dk))]

        # Build a vector c containing as many ones as samples in the calibrating set followed by 2*K zeros, where K is the number of classes
        c=np.zeros(self.Xcal.shape[1]+2*np.unique(self.ycal).shape[0])
        c[:self.Xcal.shape[1]]=1

        # Build a list of bounds containing as many pairs of bounds (0,None) as samples in total for e followed by K pairs of bounds (Nk/2,None) for c, followed by K pairs (None,None) for F, where K is the number of classes
        bounds=[(0,None) for i in range(self.Xcal.shape[1])]+[(self.Dkcal[i].shape[1]/2,None) for i in range(len(self.Dkcal))]+[(None,None) for i in range(len(self.Dkcal))]

        # Build a sparse matrix A of size N*K x (N+2k), where N is the number of samples and k is the number of classes
        A=np.zeros([self.Xcal.shape[1]*np.unique(self.ycal).shape[0],self.Xcal.shape[1]+2*np.unique(self.ycal).shape[0]])
        A = sp.csc_matrix(A)

        # Build a vector b of size N*K
        b=np.zeros(self.Xcal.shape[1]*np.unique(self.ycal).shape[0])
        # Get the number of samples of each class in the calibrating set
        Nk=[len(self.Dkcal[i][0]) for i in range(len(self.Dkcal))]

        # # get the max value of Jkx ignoring the infinite values
        # maxJkx=np.max([Jkx[i][j] for i in range(len(self.Dk)) for j in range(self.Xcal.shape[1]) if Jkx[i][j]!=np.inf])
        # # replace the infinite values with ten times the max value
        # Jkx=[[maxJkx*10 if Jkx[i][j]==np.inf else Jkx[i][j] for j in range(self.Xcal.shape[1])] for i in range(len(self.Dk))]

        # Put a row counter to 0
        row=0
        if self.ck is None:
            # Iterate over the classes
            for k in range(np.unique(self.ycal).shape[0]):
                # Iterate over the samples
                for i in range(self.Dkcal[k].shape[1]):
                    # Iterate over the classes
                    for r in range(np.unique(self.ycal).shape[0]):
                        # If k!=r
                        if k!=r:
                            # Set the values for this row. First value is -1 in the position corresponding to e_{x_{k,i}}. For that, get the sum of the number of samples of the previous classes and add the number of the sample
                            A[row,int(i+np.sum(Nk[:k]))]=-1
                            # Second value is Jkx[k][i] in the position corresponding to c_{k}, which is k positions after the last e position
                            A[row,int(self.Xcal.shape[1]+k)]=Jkx[k][i]
                            # Third value is -Jkx[r][i] in the position corresponding to c_{r}, which is r positions after the last e position
                            A[row,int(self.Xcal.shape[1]+r)]=-Jkx[r][i]
                            # Fourth value is -1 in the position corresponding to f_{gamma, c_k}, which is k positions after the last c position
                            A[row,int(self.Xcal.shape[1]+np.unique(self.ycal).shape[0]+k)]=-1
                            # Last value is 1 in the position corresponding to f_{gamma, c_r}, which is r positions after the last c position
                            A[row,int(self.Xcal.shape[1]+np.unique(self.ycal).shape[0]+r)]=1

                        # Set the value for this row in b
                        b[row]=-self.prk[r][k]+1
                        
                        # Increase the row counter
                        row+=1
        else:
            # Iterate over the classes
            for k in range(np.unique(self.ycal).shape[0]):
                # Iterate over the samples
                for i in range(self.Dkcal[k].shape[1]):
                    # Iterate over the classes
                    for r in range(np.unique(self.ycal).shape[0]):
                        # If k!=r
                        if k!=r:
                            # Set the values for this row. First value is -1 in the position corresponding to e_{x_{k,i}}. For that, get the sum of the number of samples of the previous classes and add the number of the sample
                            A[row,int(i+np.sum(Nk[:k]))]=-1
                            # # Second value is Jkx[k][i] in the position corresponding to c_{k}, which is k positions after the last e position
                            # A[row,int(self.Xcal.shape[1]+k)]=Jkx[k][i]
                            # # Third value is -Jkx[r][i] in the position corresponding to c_{r}, which is r positions after the last e position
                            # A[row,int(self.Xcal.shape[1]+r)]=-Jkx[r][i]
                            # Fourth value is -1 in the position corresponding to f_{gamma, c_k}, which is k positions after the last c position
                            A[row,int(self.Xcal.shape[1]+np.unique(self.ycal).shape[0]+k)]=-1
                            # Last value is 1 in the position corresponding to f_{gamma, c_r}, which is r positions after the last c position
                            A[row,int(self.Xcal.shape[1]+np.unique(self.ycal).shape[0]+r)]=1

                        # Set the value for this row in b
                        b[row]=-self.prk[r][k]-Jkx[k][i]*self.ck[k]+Jkx[r][i]*self.ck[r]
                        
                        # Increase the row counter
                        row+=1
        # Remove from A and b the rows that are all zeros
        b=b[~np.all(A.toarray()==0,axis=1)]
        A=A[~np.all(A.toarray()==0,axis=1)]

        # Solve the optimization problem
        res=linprog(c, A_ub=A.toarray(), b_ub=b, bounds=bounds, method='highs')

        # Get the optimized values for c, starting from the position of the last e and ending in the position of the last c
        if self.ck is None:
            self.ck=res.x[self.Xcal.shape[1]:self.Xcal.shape[1]+np.unique(self.ycal).shape[0]]

        # Get the optimized values for F, starting from the position of the last c
        if self.Fk is None:
            logFk=res.x[self.Xcal.shape[1]+np.unique(self.ycal).shape[0]:]

            # Calculate the values of F
            self.Fk=[np.exp(logFk[i]) for i in range(np.unique(self.ycal).shape[0])]

    def classifyProbs(self, x):
        '''Applies the classifier to a feature vector x. Returns the probability of each class.
        
        Args:
            x (list): The feature vector to classify.
            
        Returns:
            list: The probability of each class.
        '''
        # Calculate the value of the objective function for each class
        jx = [self.getJ(self.Dk[i], x) for i in range(len(self.Dk))]
        # Calculate the probability of each class
        Prob = [self.Fk[i] * np.exp(-self.ck[i] * jx[i]) * self.ClassProb[i] for i in range(len(self.Dk))]
        # Normalize the probabilities
        if np.sum(Prob) == 0:
            Probn = [1/len(self.Dk) for i in range(len(self.Dk))]
        else:
            Probn = [Prob[i] / np.sum(Prob) for i in range(len(self.Dk))]
        return Probn
    
    def classify(self, x):
        '''Applies the classifier to a feature vector x. Returns the predicted class based on the value of the objective function.

        Parameters:
        x (numpy.ndarray): The feature vector to classify.

        Returns:
        int: The predicted class based on the value of the objective function.
        '''
        # Apply the classifier to x
        Prob = self.classifyProbs(x)
        # Select the class with the highest value of the objective function
        y_pred = np.argmax(Prob)
        return y_pred