# -*- coding: utf-8 -*-
"""
Practica 4 - PCA y Analogia

Cristina Vílchez y Elena Perez

Referencias:
    
    Fuente primaria del reanálisis
    https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1498
    
    Temperatura en niveles de presión:
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1497

"""
import os
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA

#workpath = "C:\Users\usu312\Documents\GC"
os.getcwd()
#os.chdir(workpath)
files = os.listdir(".")

#################################################
#################################################
#################################################

#  Lectura de temperaturas 2019 - air

#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = nc.netcdf_file("./air.2019.nc", 'r')

print(f.history)
print(f.dimensions)
print(f.variables)
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
air = f.variables['air'][:].copy()
air_units = f.variables['air'].units
print(air.shape)

f.close()


#################################################
#################################################
#################################################


#time_idx = 237  # some random day in 2012
# Python and the renalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) #- offset\
           for t in time]
np.min(dt_time)
np.max(dt_time)



#############################################
#############################################
#############################################

#EJERCICIO 1
#Lectura de altura geopotencial Z de 2019 - hgt

f = nc.netcdf_file("./hgt.2019.nc", 'r')


time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
hgt = f.variables['hgt'][:].copy()
hgt_units = f.variables['hgt'].units


f.close()


#Buscamos la posicion de los 500hPa en el vector de presiones
p500 = np.where(level == 500)[0][0];

#hgt500 coniene todos los datos con presion fija 500hPa
hgt500 = hgt[:,p500,:,:].reshape(len(time),len(lats)*len(lons))

n_components=4

X = hgt500
Y = hgt500.transpose()
pca = PCA(n_components=n_components)

# Componentes e la base que maximiza la varianza explicada -> modelizar comportamietnos


#Hallamos la varianza explicada
pca.fit(X)
print(pca.explained_variance_ratio_)
pca.components_.shape

pca.fit(Y)
print(pca.explained_variance_ratio_)
pca.components_.shape


#Representamos las cuatro componentes principales 
State_pca = pca.fit_transform(X)

Element_pca = pca.fit_transform(Y)
Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca[i-1,:,:])
plt.show()


#############################################
#############################################
#############################################

#EJERCICIO 2


#Lectura de altura geopotencial de 2020 - hgt20

files = os.listdir(".")

#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = nc.netcdf_file("./hgt.2020.nc", 'r')

time20 = f.variables['time'][:].copy()
time_bnds20 = f.variables['time_bnds'][:].copy()
time_units20 = f.variables['time'].units
level20 = f.variables['level'][:].copy()
lats20 = f.variables['lat'][:].copy()
lons20 = f.variables['lon'][:].copy()
hgt20 = f.variables['hgt'][:].copy()
hgt_units20 = f.variables['hgt'].units


f.close()


#time_idx = 237  # some random day in 2012
# Python and the renalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time20 = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) #- offset\
           for t in time20]
np.min(dt_time)
np.max(dt_time)


#en20 es el indice del 20 de enero en el vector de dias
en20 = dt_time20.index(dt.date(2020, 1, 20));


# i es longitud, j es latitud, k es presion
#longitud en (-20,20)
i_ini = np.where(lons20 == 20.)[0][0] + 1 #(0, ini)
i_fin = np.where(lons20 == 340.)[0][0]  # (fin, 360)

#Latitud en (30,50)
j_fin = np.where(lats20 == 30.)[0][0]
j_ini = np.where(lats20 == 50.)[0][0]+1

#presion en 1000 y 500
k1 = np.where(level20 == 500)[0][0]
k2 = np.where(level20 == 1000)[0][0]


#Vector dist: vector de distancias al día 20 de enero
#Contiene pares dia 2019 - distancia del día al 20 de enero de 2020
dist = []
for d in range(365):
    di = 0
    for k in range(17):
        if k == k1 or k == k2:
            for j in range(j_ini,j_fin,1):
                for i in range (0,i_ini, 1):
                    di += 0.5*(hgt20[en20,k,j,i]-hgt[d,k,j,i])**2
                for i in range (i_fin+1, len(lons20), 1):
                    di += 0.5*(hgt20[en20,k,j,i]-hgt[d,k,j,i])**2
    dist.append((np.sqrt(di),d))


#Encontrar los 4 días más cercanos
#Ordenamos el vector por la componente distancia y cogemos los 4 primeros - analogos
dist.sort(key=lambda tup:tup[0])
analogos = []
for i in range(4):
    analogos.append(dist[i][1])

print(analogos)



# Lectura de temperaturas de 2020 - air20
f = nc.netcdf_file("./air.2020.nc", 'r')

air20 = f.variables['air'][:].copy()
air_units20 = f.variables['air'].units
factor = f.variables['air'].scale_factor.copy()
add = f.variables['air'].add_offset.copy()


f.close()

# Calculamos el error absoluto medio: 
# Para cada coordenada, calculamos la media de las temperatuas de los cuatro días análogos
# y su distancia a la temperatura del 20 de enero de 2020 en esa coordenada.
# Después hacemos la media de todas las distancias 
media = 0
for i in range(144):
    for j in range(73):
        suma = 0
        for d in analogos:
            suma += air[d,0,j,i]
        suma = suma/4
        
        resta = np.abs(factor*suma - factor*air20[en20,0,j,i])
        media += resta

eam = media/(144*73)
        
print("El error absoluto medio de la temperatura prevista para el 20 de enero de 2020 es ", eam)


