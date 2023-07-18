import sys
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import time
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
import isadoralib as isl
import time as tm
import seaborn as sns
import scipy.optimize as opt
import multiprocessing as mp
import tqdm
import qpsolvers as qp

#KRIGGING

# Crea una función para realizar el bucle en multiprocesamiento
def Krigging(inputs):
    (x_i, X_train, y_train, alph, X_train_pinv, classes, num_classes) = inputs

    # Crea una matriz P de tamaño (2*n_train, 2*n_train) definida por una matriz identidad de tamaño (n_train, n_train) en la esquina superior izquierda y 0 en el resto de posiciones
    P = np.zeros([2*X_train.shape[0], 2*X_train.shape[0]])
    P[:X_train.shape[0], :X_train.shape[0]] = np.eye(X_train.shape[0])

    # Crea una matriz q de tamaño (2*n_train, 1) definida por un vector de 0s de tamaño (n_train, 1) en la parte superior y un vector de 1s de tamaño (n_train, 1) en la parte inferior y lo multiplica por alph
    q = np.zeros([2*X_train.shape[0], 1])
    q[X_train.shape[0]:] = alph

    # Crea una matriz G de tamaño (2*n_train,2*n_train) definida por cuatro matrices de identidad de tamaño (n_train, n_train), negativas excepto en la esquina superior derecha
    G = np.zeros([2*X_train.shape[0], 2*X_train.shape[0]])
    G[:X_train.shape[0], :X_train.shape[0]] = -np.eye(X_train.shape[0])
    G[X_train.shape[0]:, X_train.shape[0]:] = -np.eye(X_train.shape[0])
    G[:X_train.shape[0], X_train.shape[0]:] = np.eye(X_train.shape[0])
    G[X_train.shape[0]:, :X_train.shape[0]] = -np.eye(X_train.shape[0])

    # Crea una matriz h de tamaño (2*n_train, 1) definida por un vector de 0s
    h = np.zeros([2*X_train.shape[0], 1])

    # Crea una matriz A que contenga los vectores de características de los datos de entrenamiento traspuestas en su esquina superior izquierda, una fila de 1s en la parte inferior izquierda, y una matriz de 0s en la parte derecha con tantas filas como características +1 y tantas columnas como datos de entrenamiento
    A = np.zeros([X_train.shape[1]+1, 2*X_train.shape[0]])
    A[:X_train.shape[1], :X_train.shape[0]] = X_train.T
    A[X_train.shape[1], X_train.shape[0]:] = 1

    # Crea un vector de características ampliado con un 1
    b = np.hstack([x_i, 1])


    #calcula el T que minimiza la función Qp usando OSQP
    T = qp.solve_qp(P, q, G, h, A, b, solver='osqp')

    # Obtiene el vector de lambdas de los datos de entrenamiento
    lambda_i = T[:X_train.shape[0]]

    scores_test = np.zeros([num_classes])
    # Para cada clase
    for j in range(num_classes):
        # Obtiene los índices correspondientes a la clase j en el conjunto de entrenamiento
        idx = np.where(y_train==classes[j])

        # Obtiene el score de la clase j para el dato de prueba sumando los lambdas de los datos de la clase j
        scores_test[j] = np.sum(lambda_i[idx])


    # devuelve la clase con mayor score para el dato de prueba
    return classes[np.argmax(scores_test[:])]

# clase main
if __name__ == "__main__":

    sns.set(rc={'figure.figsize':(11.7,8.27)})

    year_train="2014"
    year_datas=["2015","2016","2019"]
    sufix="rht"

    threads=12

    # maxltpitems=120
    # maxmeteoitems=120
    # minltpitems=3
    # minmeteoitems=3
    # stepltp=39
    # stepmeteo=39

    # lptlist=[*range(minltpitems,maxltpitems,stepltp)]
    # meteolist=[*range(minmeteoitems,maxmeteoitems,stepmeteo)]
    #Haz con LTP de [10:3:50] por ejemplo con PCA de [10:3:50] y meteo [0:1:9] Son 1000 datos más
    ltpitems=80
    meteoitems=4
    comp=13 #LDA 

    alph=1

    n_dias_print=5
    #matplotlib.use("Agg")



            
    res=pd.DataFrame()
    times=pd.DataFrame()
    #pd.set_option('display.max_rows', None)

    for year_data in year_datas:
        timepreprostart=tm.time()
        print(year_data)
        saveFolder="ignore/figures/PCALDAMETEOresults/"+year_train+"-"+year_data+"-"+sufix+"/"
        # Carga de datos de entrenamiento
        tdvT,ltpT,meteoT,trdatapd=isl.cargaDatos(year_train,sufix)

        # Carga de datos de predicción
        tdvP,ltpP,meteoP,valdatapd=isl.cargaDatos(year_data,sufix)

        # guarda la información raw para plots
        ltpPlot = ltpP.copy()
        meteoPlot = meteoP.copy()

        # añade meteo a ltpT
        ltpT=ltpT.join(meteoT)
        # elimina los valores NaN de ltp
        ltpT = ltpT.dropna(axis=1,how='all')

        # añade meteo a ltpP
        ltpP=ltpP.join(meteoP)

        # elimina los valores NaN de ltp
        ltpP = ltpP.dropna(axis=1,how='all')
        # rellena los valores NaN de ltp con el valor anterior
        ltpP = ltpP.fillna(method='ffill')
        ltpT = ltpT.fillna(method='ffill')
        # rellena los valores NaN de ltp con el valor siguiente
        ltpP = ltpP.fillna(method='bfill')
        ltpT = ltpT.fillna(method='bfill')

        # aplica un filtro de media móvil a ltp
        ltpP = ltpP.rolling(window=240,center=True).mean()
        ltpT = ltpT.rolling(window=240,center=True).mean()

        # calcula el valor medio de ltp para cada dia
        ltp_medioP = ltpP.groupby(ltpP.index.date).mean()
        ltp_medioT = ltpT.groupby(ltpT.index.date).mean()

        # calcula el valor de la desviación estándar de ltp para cada dia
        ltp_stdP = ltpP.groupby(ltpP.index.date).std()
        ltp_stdT = ltpT.groupby(ltpT.index.date).std()

        # cambia el índice a datetime
        ltp_medioP.index = pd.to_datetime(ltp_medioP.index)
        ltp_medioT.index = pd.to_datetime(ltp_medioT.index)
        ltp_stdP.index = pd.to_datetime(ltp_stdP.index)
        ltp_stdT.index = pd.to_datetime(ltp_stdT.index)

        # remuestrea ltp_medio y ltp_std a minutal
        ltp_medioP = ltp_medioP.resample('T').pad()
        ltp_medioT = ltp_medioT.resample('T').pad()
        ltp_stdP = ltp_stdP.resample('T').pad()
        ltp_stdT = ltp_stdT.resample('T').pad()

        # normaliza ltp para cada dia

        ltpP = (ltpP - ltp_medioP) / ltp_stdP
        ltpT = (ltpT - ltp_medioT) / ltp_stdT

        # obtiene todos los cambios de signo de R_Neta_Avg en el dataframe meteo
        signosP = np.sign(meteoP.loc[:,meteoP.columns.str.startswith('R_Neta_Avg')]).diff()
        signosT = np.sign(meteoT.loc[:,meteoT.columns.str.startswith('R_Neta_Avg')]).diff()
        # obtiene los cambios de signo de positivo a negativo
        signos_pnP = signosP<0
        signos_pnT = signosT<0
        # elimina los valores falsos (que no sean cambios de signo)
        signos_pnP = signos_pnP.replace(False,np.nan).dropna()
        signos_pnT = signos_pnT.replace(False,np.nan).dropna()
        # obtiene los cambios de signo de negativo a positivo
        signos_npP = signosP>0
        signos_npT = signosT>0
        # elimina los valores falsos (que no sean cambios de signo)
        signos_npP = signos_npP.replace(False,np.nan).dropna()
        signos_npT = signos_npT.replace(False,np.nan).dropna()

        # duplica el índice de signos np como una columna más en signos_np
        signos_npP['Hora'] = signos_npP.index
        signos_npT['Hora'] = signos_npT.index
        # recorta signos np al primer valor de cada día
        signos_npP = signos_npP.resample('D').first()
        signos_npT = signos_npT.resample('D').first()

        #elimina los dias en los que no haya cambio de signo
        signos_npP=signos_npP.dropna()
        signos_npT=signos_npT.dropna()

        # duplica el índice de signos pn como una columna más en signos_pn
        signos_pnP['Hora'] = signos_pnP.index
        signos_pnT['Hora'] = signos_pnT.index
        # recorta signos pn al último valor de cada día
        signos_pnP = signos_pnP.resample('D').last()
        signos_pnT = signos_pnT.resample('D').last()

        #elimina los días en los que no haya cambio de signo
        signos_pnP = signos_pnP.dropna()
        signos_pnT = signos_pnT.dropna()

        # recoge los valores del índice de ltp donde la hora es 00:00
        ltp_00P = ltpP.index.time == time.min
        ltp_00T = ltpT.index.time == time.min
        # recoge los valores del índice de ltp donde la hora es la mayor de cada día
        ltp_23P = ltpP.index.time == time(23,59)
        ltp_23T = ltpT.index.time == time(23,59)

        # crea una columna en ltp que vale 0 a las 00:00
        ltpP.loc[ltp_00P,'Hora_norm'] = 0
        ltpT.loc[ltp_00T,'Hora_norm'] = 0
        # iguala Hora_norm a 6 en los índices de signos np
        ltpP.loc[signos_npP['Hora'],'Hora_norm'] = 6
        ltpT.loc[signos_npT['Hora'],'Hora_norm'] = 6
        # iguala Hora_norm a 18 en los índices de signos pn
        ltpP.loc[signos_pnP['Hora'],'Hora_norm'] = 18
        ltpT.loc[signos_pnT['Hora'],'Hora_norm'] = 18
        # iguala Hora_norm a 24 en el último valor de cada día
        ltpP.loc[ltp_23P,'Hora_norm'] = 24
        ltpT.loc[ltp_23T,'Hora_norm'] = 24
        # iguala el valor en la última fila de Hora_norm a 24
        ltpP.loc[ltpP.index[-1],'Hora_norm'] = 24
        ltpT.loc[ltpT.index[-1],'Hora_norm'] = 24
        # interpola Hora_norm en ltp
        ltpP.loc[:,'Hora_norm'] = ltpP.loc[:,'Hora_norm'].interpolate()
        ltpT.loc[:,'Hora_norm'] = ltpT.loc[:,'Hora_norm'].interpolate()

        # almacena los valores antes de recortar
        ltpPBase=ltpP

        # recorta ltp a los tramos de 6 a 18 de hora_norm
        ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
        ltpT = ltpT.loc[ltpT['Hora_norm']>=6,:]
        ltpP = ltpP.loc[ltpP['Hora_norm']<=18,:]
        ltpT = ltpT.loc[ltpT['Hora_norm']<=18,:]


        # añade la hora normalizada al índice de ltp
        ltpP.index = [ltpP.index.strftime('%Y-%m-%d'),ltpP['Hora_norm']]
        ltpT.index = [ltpT.index.strftime('%Y-%m-%d'),ltpT['Hora_norm']]

        #crea el índice de ltpPBase
        ltpPBase['Hora_norm']=ltpPBase['Hora_norm'].apply(pd.to_timedelta,unit='H')
        ltpPBase['dia_norm'] = ltpPBase.index.strftime('%Y-%m-%d')
        ltpPBase.index = [ltpPBase['dia_norm'].apply(pd.to_datetime,format='%Y-%m-%d'),ltpPBase['Hora_norm']]
        ltpPBase=ltpPBase.drop('Hora_norm',axis=1)
        ltpPBase=ltpPBase.drop('dia_norm',axis=1)
        ltpPBase=ltpPBase.unstack(level=0)

        valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')
        trdatapd.index = trdatapd.index.strftime('%Y-%m-%d')

        #obtiene el índice interseccion de valdatapd y el primer nivel del índice de ltp
        ltpPdates = ltpP.index.get_level_values(0)
        ltpTdates = ltpT.index.get_level_values(0)

        valdatapd_ltp = valdatapd.index.intersection(ltpPdates)
        trdatapd_ltp = trdatapd.index.intersection(ltpTdates)

        # vuelve a separar los valores de meteo de ltp
        meteoP_norm=ltpP.drop(ltpP.columns[ltpP.columns.str.startswith('LTP')], axis=1)
        meteoT_norm=ltpT.drop(ltpT.columns[ltpT.columns.str.startswith('LTP')], axis=1)

        # elimina los valores de ltp que no estén en valdatapd
        ltpv = ltpP.loc[valdatapd_ltp,valdatapd.columns]
        # elimina los valores de ltp que no estén en trdatapd
        ltpt = ltpT.loc[trdatapd_ltp,trdatapd.columns]

        # unstackea meteoP_norm y meteoT_norm
        meteoP_norm = meteoP_norm.unstack(level=0)
        meteoT_norm = meteoT_norm.unstack(level=0)

        # unstackea ltpv
        ltpv = ltpv.unstack(level=0)
        # unstackea ltpt
        ltpt = ltpt.unstack(level=0)

        # crea un índice para ajustar frecuencias
        ltpv_index_float=pd.Int64Index(np.floor(ltpv.index*1000000000))
        ltpt_index_float=pd.Int64Index(np.floor(ltpt.index*1000000000))
        meteoP_index_float=pd.Int64Index(np.floor(meteoP_norm.index*1000000000))
        meteoT_index_float=pd.Int64Index(np.floor(meteoT_norm.index*1000000000))

        ltpv.index = pd.to_datetime(ltpv_index_float)
        ltpt.index = pd.to_datetime(ltpt_index_float)
        meteoP_norm.index = pd.to_datetime(meteoP_index_float)
        meteoT_norm.index = pd.to_datetime(meteoT_index_float)

        ltpv_orig=ltpv.copy()
        ltpt_orig=ltpt.copy()
        meteoP_norm_orig=meteoP_norm.copy()
        meteoT_norm_orig=meteoT_norm.copy()

        timepreproend=tm.time()
        preprotime=timepreproend-timepreprostart
        print('ltpitems:'+str(ltpitems))
        print('meteoitems:'+str(meteoitems))
        timeloopstart=tm.time()
        fltp=12/ltpitems
        if meteoitems>0:
            fmeteo=12/meteoitems
        else:
            fmeteo=0
        # convierte el indice a datetime para ajustar frecuencias
        ltpv=ltpv_orig.resample(str(int(fltp*1000))+'L').mean()
        ltpt=ltpt_orig.resample(str(int(fltp*1000))+'L').mean()
        if meteoitems>0:
            meteoP_norm=meteoP_norm_orig.resample(str(int(fmeteo*1000))+'L').mean()
            meteoT_norm=meteoT_norm_orig.resample(str(int(fmeteo*1000))+'L').mean()

        # conserva los valores de 1970-01-01 00:00:06.000 a 1970-01-01 00:00:17.900
        ltpv = ltpv.loc[ltpv.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
        ltpt = ltpt.loc[ltpt.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
        ltpv = ltpv.loc[ltpv.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
        ltpt = ltpt.loc[ltpt.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]

        if meteoitems>0:
            meteoP_norm = meteoP_norm.loc[meteoP_norm.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
            meteoP_norm = meteoP_norm.loc[meteoP_norm.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
            meteoT_norm = meteoT_norm.loc[meteoT_norm.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
            meteoT_norm = meteoT_norm.loc[meteoT_norm.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]


        # Crea una serie para restaurar el índice
        norm_index=pd.Series(np.arange(6,18,fltp))
        #recorta norm_index para que coincida con el tamano de ltpt si se ha producido un desajuste al calcular el dataframe
        norm_index=norm_index.loc[norm_index.index<len(ltpt)]
        # Ajusta el índice de ltpv a la serie
        ltpv.index=norm_index
        # Ajusta el índice de ltpt a la serie
        ltpt.index=norm_index

        if meteoitems>0:
            # Crea una serie para restaurar el índice
            norm_index=pd.Series(np.arange(6,18,fmeteo))
            #recorta norm_index para que coincida con el tamano de meteoP_norm si se ha producido un desajuste al calcular el dataframe
            norm_index=norm_index.loc[norm_index.index<len(meteoT_norm)]
            # Ajusta el índice de meteoP_norm a la serie
            meteoP_norm.index=norm_index
            # Ajusta el índice de meteoT_norm a la serie
            meteoT_norm.index=norm_index

            # dropea la columna Hora_norm de meteo
            meteoP_norm = meteoP_norm.drop('Hora_norm',axis=1)
            meteoT_norm = meteoT_norm.drop('Hora_norm',axis=1)

            # stackea meteoP_norm y meteoT_norm
            meteoP_norm = meteoP_norm.stack(level=0)
            meteoT_norm = meteoT_norm.stack(level=0)

            #intercambia los niveles del índice de meteo
            meteoP_norm.index = meteoP_norm.index.swaplevel(0,1)
            meteoT_norm.index = meteoT_norm.index.swaplevel(0,1)

            meteoP_norm=meteoP_norm.dropna(axis=1,how='all')
            meteoT_norm=meteoT_norm.dropna(axis=1,how='all')

            #combina los dos índices de meteo
            meteoP_norm.index = meteoP_norm.index.map('{0[1]}/{0[0]}'.format)
            meteoT_norm.index = meteoT_norm.index.map('{0[1]}/{0[0]}'.format)

            #elimina los indices no comunes de meteo
            meteoP_norm = meteoP_norm.loc[meteoP_norm.index.isin(meteoT_norm.index)]
            meteoT_norm = meteoT_norm.loc[meteoT_norm.index.isin(meteoP_norm.index)]
        else:
            meteoP_norm = pd.DataFrame()
            meteoT_norm = pd.DataFrame()

        #crea un array de numpy en blanco
        array_ltpv=np.empty((len(ltpv)+len(meteoP_norm),0))
        array_ltpt=np.empty((len(ltpt)+len(meteoT_norm),0))

        #por cada elemento en el primer índice de columnas de ltp
        for i in ltpv.columns.levels[0]:
            ltpv_col=ltpv.loc[:,i]
            if meteoitems>0:
                # elimina los valores de meteo que no estén en ltp_col
                meteo_ltp = ltpv_col.columns.intersection(meteoP_norm.columns)
                meteoP_col = meteoP_norm.loc[:,meteo_ltp]

                # combina los valores de ltpv con los de meteo
                merge_ltp_meteo = pd.merge(ltpv.loc[:,i],meteoP_col,how='outer')
            else:
                merge_ltp_meteo = ltpv.loc[:,i]
            # añade la unión al array de numpy
            array_ltpv=np.append(array_ltpv,merge_ltp_meteo.values,axis=1)

        #por cada elemento en el primer índice de columnas de ltp
        for i in ltpt.columns.levels[0]:
            ltpt_col=ltpt.loc[:,i]
            if meteoitems>0:
                # elimina los valores de meteo que no estén en ltp_col
                meteo_ltp = ltpt_col.columns.intersection(meteoT_norm.columns)
                meteoT_col = meteoT_norm.loc[:,meteo_ltp]

                # combina los valores de ltpv con los de meteo
                merge_ltp_meteo = pd.merge(ltpt.loc[:,i],meteoT_col,how='outer')
            else:
                merge_ltp_meteo = ltpt.loc[:,i]
            # añade la unión al array de numpy
            array_ltpt=np.append(array_ltpt,merge_ltp_meteo.values,axis=1)
        # print("ltpt")
        # print(ltpt)
        # print("meteo_ltp")
        # print(meteo_ltp)
        # print("meteoT_norm")
        # print(meteoT_norm)
        # print("ltpt_col")
        # print(ltpt_col)

        # crea los valores X e y para el modelo
        Xtr=array_ltpt.transpose()
        Ytr=trdatapd.unstack().values
        Xv=array_ltpv.transpose()
        Yv=valdatapd.unstack().values

        # Resta 1 a las clases
        Ytr=Ytr-1
        Yv=Yv-1

        #print(np.shape(Xtr))
        #print(np.shape(Xv))

        # elimina los valores NaN de Xtr y Xv
        XtrBase = np.nan_to_num(Xtr)
        XvBase = np.nan_to_num(Xv)
        timeloopend=tm.time()
        looptime=timeloopend-timeloopstart
        timeclasstart=tm.time()

        #aplica PCA
        pca = skdecomp.PCA(n_components=comp)
        pca.fit(XtrBase)
        Xtr = pca.transform(XtrBase)
        Xv = pca.transform(XvBase)

        # # crea el modelo
        # clf = sklda.LinearDiscriminantAnalysis(solver='svd')
        # # entrena el modelo
        # clf.fit(Xtr,Ytr)
        # # predice los valores de Yv
        # Ypred=clf.predict(Xv)

        

        # Obtiene las clases únicas
        classes = np.unique(Ytr)
        # Obtiene el número de clases
        num_classes = classes.shape[0]

        # Crea un array de scores vacío
        scores_test = np.zeros([num_classes, Xv.shape[0]])
        # crea una matriz de numpy de datos ampliada con una columna de unos y la transpone
        X_train_ext = np.hstack([Xtr, np.ones([Xtr.shape[0], 1])])
        # calcula la pseudo-inversa
        X_train_pinv = np.linalg.pinv(X_train_ext)

        print(str(0)+"/"+str(Xv.shape[0]))
        # # obtiene el tiempo de inicio del bucle
        # timeclasstart=tm.time()

        print(Xtr.shape)
        
        # Crea una lista de inputs para los procesos
        inputs = [(Xv[i], Xtr, Ytr, alph, X_train_pinv, classes, num_classes) for i in range(Xv.shape[0])]

        # Crea una pool de procesos
        pool = mp.Pool(processes=threads)

        print('Esperando a que terminen los procesos...')
        # Lanza los procesos
        #r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))
        #y_test_pred = pool.map(Krigging,inputs)
        y_test_pred = list(tqdm.tqdm(pool.imap(Krigging, inputs), total=Xtr.shape[0]))
        # convierte la lista en un array de numpy
        y_test_pred = np.array(y_test_pred)


        # Cierra la pool
        pool.close()
        # Para cada dato de prueba
        # for i in range(Xv.shape[0]):# X_train, X_test, y_train, alph, i, X_train_pinv, classes, num_classes, scores_test
        #     # Extrae el vector de características del dato de prueba i
        #     x_i = Xv[i]

        #     # Crea un vector de características ampliado con un 1
        #     x_i_ext = np.hstack([Xv[i], 1])
        #     # Multiplica el vector de características ampliado por la pseudo-inversa para obtener el vector lambda de partida
        #     lambda_i = np.dot(x_i_ext,X_train_pinv)

        #     #calcula el vector lambda que minimiza la función fun sujeto a las constraints const
        #     lambda_i = opt.minimize(fun, lambda_i, args=(alph), constraints={'fun': const, 'type': 'eq', 'args': (Xtr, x_i)}, tol=1e-6).x

        #     # Para cada clase
        #     for j in range(num_classes):
        #         # Obtiene los índices correspondientes a la clase j en el conjunto de entrenamiento
        #         idx = np.where(Ytr==classes[j])

        #         # Obtiene el score de la clase j para el dato de prueba i sumando los lambdas de los datos de la clase j
        #         scores_test[j, i] = np.sum(lambda_i[idx])

        #     # guarda la clase con mayor score para el dato de prueba i en el array de predicciones
        #     y_test_pred[i] = np.argmax(scores_test[:, i])
            # print(str(i+1)+"/"+str(Xv.shape[0]))
            # # obtiene el tiempo de finalización de la iteración
            # timeclasend=tm.time()
            # # calcula el tiempo hasta el momento
            # timeclas=timeclasend-timeclasstart
            # # calcula el tiempo restante
            # timeremaining=timeclas/(i+1)*(Xv.shape[0]-(i+1))
            # # imprime el tiempo restante en formato hh:mm:ss
            # print("Tiempo restante: "+str(int(timeremaining/3600))+":"+str(int((timeremaining%3600)/60))+":"+str(int(timeremaining%60)))

        # calcula la precisión del modelo sobre el conjunto de prueba
        acc = np.sum(y_test_pred==Yv)/Yv.shape[0]
        # calcula la precisión del modelo sobre el conjunto de prueba por clases
        acc_class = np.zeros(num_classes)
        for i in range(num_classes):
            idx = np.where(Yv==classes[i])
            acc_class[i] = np.sum(y_test_pred[idx]==Yv[idx])/Yv[idx].shape[0]
        # calcula la precisión balanceada del modelo sobre el conjunto de prueba
        acc_bal = np.mean(acc_class)
        # calcula la matriz de confusión sobre el conjunto de prueba
        conf = np.zeros([num_classes, num_classes])
        for i in range(Yv.shape[0]):
            conf[int(Yv[i]), int(y_test_pred[i])] += 1

        # calcula la matriz de confusión normalizada sobre el conjunto de prueba
        conf_n = conf/np.sum(conf, axis=1)[:, np.newaxis]

        # # plotea Yv y Ypred
        # fig, ax = plt.subplots()
        # plt.plot(Ypred, color="#C22F00", marker='+')
        # plt.plot(Yv, color="#4E94EC", marker='x')
        # plt.legend(["Automatic classification","Manual classification"])
        # #plt.grid()
        # plt.xlabel('Sample number')
        # plt.ylabel('Hydric stress level')
        # #fig.savefig('ignore/resultadosPCALDA/'+year_data+'.png')
        # plt.show()

        # calcula la matriz de confusion

        # confusion_matrix = skmetrics.confusion_matrix(Yv, Ypred)
        # print(confusion_matrix)
        
        print('accuracy: ',acc)
        print('accuracy per class: ',acc_class)
        print('balanced accuracy: ',acc_bal)
        print('confusion matrix: ')
        print(conf)
        print('confusion matrix normalized: ')
        print(conf_n)

        
    #     res=res.append({'ltp samples':ltpitems,'meteo samples':meteoitems,'year data':year_data,'components from PCA':comp,'fraction from total PCA':np.around(comp/XtrBase.shape[1],1),'accuracy':accuracy},ignore_index=True)
    #     # res=res.append({'ltp samples':ltpitems,'meteo samples':meteoitems,'year train':year_train,'year data':year_data,'components from PCA':comp,'accuracy':accuracy,'confusion matrix':np.around(bcm,2)},ignore_index=True)
    #     # res=res.append({'ltp':ltpitems,'meteo':meteoitems,'year train':year_train,'year data':year_data,'comp':comp,'acc':accuracy,'cmatrix':"\\begin{tabular}{ ccc }"+(" \\\\\n".join([" & ".join(map(str,line)) for line in bcm]))+"\\end{tabular}"},ignore_index=True)
        
    #     timeclasend=tm.time()
    #     clastime=timeclasend-timeclasstart
    #     print('Porcentaje de acierto: '+str(accuracy*100)+'%')
    #     print('frac PCA: '+str(np.around(comp/XtrBase.shape[1],1)))

    #     times=times.append({'ltp samples':ltpitems,'meteo samples':meteoitems,'year data':year_data,'components from PCA':comp,'fraction from total PCA':np.around(comp/XtrBase.shape[1],1),'preprocess':preprotime,'process':looptime,'classifier':clastime,'total':preprotime+looptime+clastime},ignore_index=True)
    # res.set_index(['year data','ltp samples','meteo samples','components from PCA','fraction from total PCA'], inplace=True) 
    # res=res.unstack(level=0)

    # times.set_index(['year data','ltp samples','meteo samples','components from PCA','fraction from total PCA'], inplace=True) 
    # times=times.unstack(level=0)

    # mean=res.mean(numeric_only=True, axis=1)
    # var=res.var(numeric_only=True, axis=1)
    # res['total accuracy']=mean
    # res['accuracy variation']=var
    # #res=res.sort_values(['total accuracy'], ascending=False)
    # print(res)
    # #res.to_csv('ignore/analisisPCALDA/resultadosPCALDAMeteo.csv')
    # #times.to_csv('ignore/analisisPCALDA/tiemposPCALDAMeteo.csv')
