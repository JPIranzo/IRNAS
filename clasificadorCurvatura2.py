import sys
from time import time
from matplotlib.markers import MarkerStyle
import matplotlib
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics

year_train="2014"
year_data="2019"

matplotlib.use("Agg")

#pd.set_option('display.max_rows', None)

# Carga de datos de entrenamiento
dfT = pd.read_csv("rawMinutales"+year_train+".csv",na_values='.')
dfT.loc[:,"Fecha"]=pd.to_datetime(dfT.loc[:,"Fecha"])# Fecha como datetime
dfT=dfT.drop_duplicates(subset="Fecha")
dfT.dropna(subset = ["Fecha"], inplace=True)
dfT=dfT.set_index("Fecha")
dfT=dfT.apply(pd.to_numeric, errors='coerce')

# separa dfT en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteoT
tdvT = dfT.loc[:,dfT.columns.str.startswith('TDV')]
ltpT = dfT.loc[:,dfT.columns.str.startswith('LTP')]
meteoT = dfT.drop(dfT.columns[dfT.columns.str.startswith('TDV')], axis=1)
meteoT = meteoT.drop(meteoT.columns[meteoT.columns.str.startswith('LTP')], axis=1)

# Carga de datos de predicción
dfP = pd.read_csv("rawMinutales"+year_data+".csv",na_values='.')
dfP.loc[:,"Fecha"]=pd.to_datetime(dfP.loc[:,"Fecha"])# Fecha como datetime
dfP=dfP.drop_duplicates(subset="Fecha")
dfP.dropna(subset = ["Fecha"], inplace=True)
dfP=dfP.set_index("Fecha")
dfP=dfP.apply(pd.to_numeric, errors='coerce')

# separa dfP en tdv y ltp en función del principio del nombre de cada columna y guarda el resto en meteoP
tdvP = dfP.loc[:,dfP.columns.str.startswith('TDV')]
ltpP = dfP.loc[:,dfP.columns.str.startswith('LTP')]
meteoP = dfP.drop(dfP.columns[dfP.columns.str.startswith('TDV')], axis=1)
meteoP = meteoP.drop(meteoP.columns[meteoP.columns.str.startswith('LTP')], axis=1)

# Carga datos de validacion
valdatapd=pd.read_csv("validacion"+year_data+".csv")
valdatapd.dropna(inplace=True)
valdatapd['Fecha'] = pd.to_datetime(valdatapd['Fecha'])
valdatapd.set_index('Fecha',inplace=True)

# Carga datos de resultados de entrenamiento
trdatapd=pd.read_csv("validacion"+year_train+".csv")
trdatapd.dropna(inplace=True)
trdatapd['Fecha'] = pd.to_datetime(trdatapd['Fecha'])
trdatapd.set_index('Fecha',inplace=True)

# elimina los valores NaN de ltp
ltpP = ltpP.dropna(axis=1,how='all')
ltpT = ltpT.dropna(axis=1,how='all')
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

# duplica el índice de signos pn como una columna más en signos_pn
signos_pnP['Hora'] = signos_pnP.index
signos_pnT['Hora'] = signos_pnT.index
# recorta signos pn al último valor de cada día
signos_pnP = signos_pnP.resample('D').last()
signos_pnT = signos_pnT.resample('D').last()

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
# recorta ltp a los tramos de 6 a 18 de hora_norm
ltpP = ltpP.loc[ltpP['Hora_norm']>=6,:]
ltpT = ltpT.loc[ltpT['Hora_norm']>=6,:]
ltpP = ltpP.loc[ltpP['Hora_norm']<=18,:]
ltpT = ltpT.loc[ltpT['Hora_norm']<=18,:]

# añade la hora normalizada al índice de ltp
ltpP.index = [ltpP.index.strftime('%Y-%m-%d'),ltpP['Hora_norm']]
ltpT.index = [ltpT.index.strftime('%Y-%m-%d'),ltpT['Hora_norm']]

valdatapd.index = valdatapd.index.strftime('%Y-%m-%d')
trdatapd.index = trdatapd.index.strftime('%Y-%m-%d')

#obtiene el índice interseccion de valdatapd y el primer nivel del índice de ltp
ltpPdates = ltpP.index.get_level_values(0)
ltpTdates = ltpT.index.get_level_values(0)

valdatapd_ltp = valdatapd.index.intersection(ltpPdates)
trdatapd_ltp = trdatapd.index.intersection(ltpTdates)

# elimina los valores de ltp que no estén en valdatapd
ltpv = ltpP.loc[valdatapd_ltp,valdatapd.columns]
# elimina los valores de ltp que no estén en trdatapd
ltpt = ltpT.loc[trdatapd_ltp,trdatapd.columns]

# unstackea ltpv
ltpv = ltpv.unstack(level=0)
# unstackea ltpt
ltpt = ltpt.unstack(level=0)

# sustituye el indice de ltpv por el valor de la columna Hora de ltpv_index
ltpv_index_float=pd.Int64Index(np.floor(ltpv.index*1000000000))

# sustituye el indice de ltpt por el valor de la columna Hora de ltpt_index
ltpt_index_float=pd.Int64Index(np.floor(ltpt.index*1000000000))

# convierte el indice a datetime para ajustar frecuencias
ltpv.index = pd.to_datetime(ltpv_index_float)
ltpv=ltpv.resample('0.1S').mean()
ltpt.index = pd.to_datetime(ltpt_index_float)
ltpt=ltpt.resample('0.1S').mean()

# conserva los valores de 1970-01-01 00:00:06.000 a 1970-01-01 00:00:17.900
ltpv = ltpv.loc[ltpv.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
ltpt = ltpt.loc[ltpt.index>=pd.to_datetime('1970-01-01 00:00:06.000'),:]
ltpv = ltpv.loc[ltpv.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]
ltpt = ltpt.loc[ltpt.index<=pd.to_datetime('1970-01-01 00:00:17.900'),:]

# Crea una serie de 0.01 a 24 para restaurar el índice
norm_index=pd.Series(np.arange(6,18,0.1))
# Ajusta el índice de ltpv a la serie de 0.01 a 24
ltpv.index=norm_index
# Ajusta el índice de ltpt a la serie de 0.01 a 24
ltpt.index=norm_index

# crea los valores de datos e y para el modelo
Dtr=ltpt.values
Ytr=trdatapd.unstack().values
Dv=ltpv.values
Yv=valdatapd.unstack().values

print('Dtr:',Dtr.shape)
print('Ytr:',Ytr.shape)
print('Dv:',Dv.shape)
print('Yv:',Yv.shape)

# crea un vector de tiempos de longitud igual a las filas de Dtr
t=np.arange(0,Dtr.shape[0])

print('t:',t.shape)
# crea los valores X calculando la parábola de mejor ajuste de los valores de Dtr y Dv por columnas
Xtr = np.transpose(np.polyfit(t,Dtr,2))
Xv = np.transpose(np.polyfit(t,Dv,2))

# crea el modelo
clf = sklda.LinearDiscriminantAnalysis(solver='svd')#,shrinkage='auto')
# entrena el modelo
clf.fit(Xtr,Ytr)
# predice los valores de Yv
Ypred=clf.predict(Xv)

# calcula la matriz de confusion
confusion_matrix = skmetrics.confusion_matrix(Yv, Ypred)

print(confusion_matrix)

res=pd.DataFrame()
res['valdata']=valdatapd.unstack()
res['estado']=pd.DataFrame(Ypred,index=res.index)
res = res.transpose()

#si no existen, crea las carpetas 1_1 a 3_3 en figurasClasificadorCurvLDA
if not os.path.exists('figurasClasificadorCurvLDA'+year_train+year_data):
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data)
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/1_1')
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/1_2')
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/1_3')
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/2_1')
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/2_2')
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/2_3')
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/3_1')
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/3_2')
    os.makedirs('figurasClasificadorCurvLDA'+year_train+year_data+'/3_3')

# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==1:
            cols.append('red')
        elif l==2:
            cols.append('blue')
        else:
            cols.append('green')
    return cols
# Create the colors list using the function above
cols=pltcolor(Ytr)
# abre una figura f con dos subplots
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# plotea en el subplot 1 los valores de la primera y segunda componente de Xtr coloreados por cols
ax1.scatter(Xtr[:,0],Xtr[:,1],color=cols,s=1)
#añade nombres a los ejes
ax1.set_xlabel('Término independiente')
ax1.set_ylabel('Término lineal')
# plotea en el subplot 2 los valores de la primera y tercera componente de Xtr coloreados por cols
ax2.scatter(Xtr[:,0],Xtr[:,2],color=cols,s=1)
#añade nombres a los ejes
ax2.set_xlabel('Término independiente')
ax2.set_ylabel('Término cuadrático')
# guarda la figura f en la carpeta figurasClasificadorConvolucion con el nombre de la columna de res_estado, indice de res_estado y valor de res_estado y valdatapd
f.savefig('figurasClasificadorCurvLDA'+year_train+year_data+'/'+year_train+'blobs.png')
# cierra la figura f
plt.close(f)

cols=pltcolor(Yv)
# abre una figura f con dos subplots
f, (ax3, ax4) = plt.subplots(2, 1, sharex=True)
# plotea en el subplot 1 los valores de la primera y segunda componente de Xv coloreados por cols
ax3.scatter(Xv[:,0],Xv[:,1],color=cols,s=1)
#añade nombres a los ejes
ax3.set_xlabel('Término independiente')
ax3.set_ylabel('Término lineal')
# plotea en el subplot 2 los valores de la primera y tercera componente de Xv coloreados por cols
ax4.scatter(Xv[:,0],Xv[:,2],color=cols,s=1)
#añade nombres a los ejes
ax4.set_xlabel('Término independiente')
ax4.set_ylabel('Término cuadrático')
# guarda la figura f en la carpeta figurasClasificadorConvolucion con el nombre de la columna de res_estado, indice de res_estado y valor de res_estado y valdatapd
f.savefig('figurasClasificadorCurvLDA'+year_train+year_data+'/'+year_data+'blobs.png')
# cierra la figura f
plt.close(f)

# guarda res en un archivo csv
res.to_csv('resClasCurvLDA'+year_train+year_data+'.csv')
# crea un dataframe con la fila estado de res
res_estado = res.loc['estado',:]
res_valdata = res.loc['valdata',:]

# unstackea res_estado
res_estado = res_estado.unstack(level=0)
# unstackea res_valdata
res_valdata = res_valdata.unstack(level=0)

res_estado=res_estado.astype(np.float64)
res_valdata=res_valdata.astype(np.float64)

#renombra el indice de res_estado a Fecha
res_estado.index = res_estado.index.rename('Fecha')
#renombra el indice de res_valdata a Fecha
res_valdata.index = res_valdata.index.rename('Fecha')

# convierte el indice de res_estado a datetime
res_estado.index = pd.to_datetime(res_estado.index)
# convierte el indice de res_valdata a datetime
res_valdata.index = pd.to_datetime(res_valdata.index)

# crea un dataframe con la matriz de confusión
conf_matrix=pd.DataFrame(index=['est LTP_1','est LTP_2','est LTP_3'],columns=['LTP_1','LTP_2','LTP_3'])
# inicializa la matriz de confusión con 0
conf_matrix.iloc[:,:]=0

# carga de secuencias tipo
sec=pd.read_csv("ltp_media"+year_train+".csv",index_col=0)
# asegura que el tipo de dato sea float
sec=sec.astype(float)
# recorta entre el índice de valor 6 y el de valor 18
sec=sec.loc[6:18]

# convierte las columnas LTP_1, LTP_2 y LTP_3 de sec a vectores de numpy
ltp_1 = sec['LTP_1'].values
ltp_2 = sec['LTP_2'].values
ltp_3 = sec['LTP_3'].values

# recorre las filas de res_estado
for i in range(len(res_estado)):
    # recorre las columnas de res_estado
    for j in range(len(res_estado.columns)):
        conf_matrix.iloc[int(res_estado.iloc[i,j]-1),int(res_valdata.iloc[i,j]-1)] += 1
        
        # abre una figura f
        f = plt.figure()
        # plotea en f el vector de referencia
        plt.plot(ltp_1,'r--')
        plt.plot(ltp_2,'g--')
        plt.plot(ltp_3,'b--')
        # plotea en f el vector de estimación
        plt.plot(ltpv.loc[:,res_estado.columns[j]].loc[:,res_estado.index[i].strftime('%Y-%m-%d')].values)
        # añade una leyenda a f
        plt.legend(['LTP_1','LTP_2','LTP_3','valores medidos'])
        # guarda la figura f en la carpeta figurasClasificadorConvolucion con el nombre de la columna de res_estado, indice de res_estado y valor de res_estado y valdatapd
        f.savefig('figurasClasificadorCurvLDA'+year_train+year_data+'/'+str(int(res_estado.iloc[i,j]))+'_'+str(int(res_valdata.iloc[i,j]))+'/'+res_estado.columns[j]+'_'+res_estado.index[i].strftime('%Y-%m-%d')+'.png')
        # cierra la figura f
        plt.close(f)

print(conf_matrix)