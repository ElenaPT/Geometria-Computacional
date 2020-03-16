# -*- coding: utf-8 -*-
"""
Practica 1 - Atractor logistico

Elena Pirez y Cristina Vilchez
"""

import matplotlib.pyplot as plt
import numpy as np

'''
Funcion logística
'''
def f(x):
    return(r*x*(1-x))
    


#EJERCICIO 1

    
#Para encontrar conjuntos atractores, vamos a probar distintos valores de x0 y r    
EPS = 0.00001 #precision para el calculo de la orbita
N=500 #numero de elementos del sistema dinamico que calculamos inicialmente 
      #para observar su evolucion en la grafica
INF = 10 #numero de veces que vamos a iterar el conjunto atractor VN, haciendo
         #'como si' la orbita tendiera a infinito

#######################
## VALORES INICIALES - TEST 1.1
r = 3.01
x0 = .1

  
# Calculamos f_n para hacernos una idea de la gráfica y elegir una M adecuada
values = []
values.append(x0)
for i in range(1,N):
    values.append(f(values[i-1]))
    
plt.plot(values)
plt.show()


#Observando la grafica, elegimos M = 400, pues parece que para ese valor se ha
#estabilizado
M = 400

'''
Funcion que calcula el n-esimo elemento del sistema dinamico x_n+1=f(x_n)
'''
def fN(x,n):
    p = x
    for i in range(n):
        p = f(p)
    return(p)

'''
Funcion que calcula el numero de elementos del conjunto atractor de un sistema.
Devuelve -1 si no encuentra tal conjunto atractor
input:
    x0: valor inicial
    M: numero de iteraciones iniciales a partir del cual buscamos la orbita
'''
def orbita(x0,M):
    fM = fN(x0,M)
    fK = fM
    
    i=1
    while i < 200:
        fK = f(fK)
        if abs(fM-fK) < EPS:
            break
        i+=1
    
    if i >= 200:
        return (-1)
    else:
        k = i
        return k



'''
Funcion que calcula el conjunto atractor de k elementos.
Coge los k últimos valores calculados del sistema, y después itera este conjunto
hasta el infinito.
También calcula el error de precision
input:
    k: numero de elementos del conjunto
    values: vector con los valores del sistema (cogeremos los k ultimos)
'''
def conjuntoAtractor(k, values):
    VN = []
    for i in range(k):
        VN.append(values[N-1-i])
    VN.sort()
    errors = []
    for j in range(INF):
        VNnew = []
        error = []
        for i in range(k):
            a = f(VN[i])
            VNnew.append(a)
        VNnew.sort()
        for i in range(k):
            error.append(VNnew[i]-VN[i])
        errors.append(error)
        VN = VNnew
    return VN, errors[2] 

# k = 2, luego tenemos dos valores a los que tiene la funcion.
k = orbita(x0,M)
if k == -1:
    print("No se ha encontrado k tal que |f_m - f_m+k| < EPS")
else:
    print("Para r = ",r,", x0 = ",x0,", hemos obtenido una k = ",k)
    VN, E = conjuntoAtractor(k, values)
    print("El conjunto de puntos atractores es ", VN, ", con un error de ",E)
    

#######################
#######################
    
    
## VALORES INICIALES - TEST 1.2

#Variamos x0 para estudiar la estabilidad del conjunto atractor

r = 3.01
x0 = .7



# Calculamos f_n para hacernos una idea de la grafica y elegir una M adecuada
values = []
values.append(x0)
for i in range(1,N):
    values.append(f(values[i-1]))
    
plt.plot(values)
plt.show()

k = orbita(x0,M)
if k == -1:
    print("No se ha encontrado k tal que |f_m - f_m+k| < EPS")
else:
    print("Para r = ",r,", x0 = ",x0,", hemos obtenido una k = ",k)
    VN, E = conjuntoAtractor(k, values)
    print("El conjunto de puntos atractores es ", VN, ", con un error de ",E)
    

########################
########################
    
## VALORES INICIALES - TEST 1.3

#Volvemos a variar x0
r = 3.01
x0 = .3



# Calculamos f_n para hacernos una idea de la grafica y elegir una M adecuada
values = []
values.append(x0)
for i in range(1,N):
    values.append(f(values[i-1]))
    
plt.plot(values)
plt.show()

k = orbita(x0,M)
if k == -1:
    print("No se ha encontrado k tal que |f_m - f_m+k| < EPS")
else:
    print("Para r = ",r,", x0 = ",x0,", hemos obtenido una k = ",k)
    VN, E = conjuntoAtractor(k, values)
    print("El conjunto de puntos atractores es ", VN, ", con un error de ",E)
    


########################
########################
########################
########################

## VALORES INICIALES - TEST 2.1

#Ahora variamos la r para buscar otro conjunto atractor
    
r = 3.47
x0 = .1


# Calculamos f_n para hacernos una idea de la grafica y elegir una M adecuada
values = []
values.append(x0)
for i in range(1,N):
    values.append(f(values[i-1]))
    
plt.plot(values)
plt.show()

k = orbita(x0,M)
if k == -1:
    print("No se ha encontrado k tal que |f_m - f_m+k| < EPS")
else:
    print("Para r = ",r,", x0 = ",x0,", hemos obtenido una k = ",k)
    VN, E = conjuntoAtractor(k, values)
    print("El conjunto de puntos atractores es ", VN, ", con un error de ",E)
    

## VALORES INICIALES - TEST 2.2

#Volvemos a variar x0
r = 3.47
x0 = .3



# Calculamos f_n para hacernos una idea de la grafica y elegir una M adecuada
values = []
values.append(x0)
for i in range(1,N):
    values.append(f(values[i-1]))
    
plt.plot(values)
plt.show()

k = orbita(x0,M)
if k == -1:
    print("No se ha encontrado k tal que |f_m - f_m+k| < EPS")
else:
    print("Para r = ",r,", x0 = ",x0,", hemos obtenido una k = ",k)
    VN, E = conjuntoAtractor(k, values)
    print("El conjunto de puntos atractores es ", VN, ", con un error de ",E)
    

########################
########################
########################
########################
    
#EJERCICIO 2 - Encontrar un conjunto atractor con 8 elementos
    
x0 = 0.1
EPS = 0.000001
N=800
M=700

#Variamos el valor de r entre (3,4), hasta que encontremos un punto tal que
#el numero de elementos del conjunto atractor k sea 8
#Almacenamos los puntos con esta propiedad en r_values
r_values = []
values = []
for r in np.arange(3.001,3.999,0.001):
    values = []
    values.append(x0)
    for i in range(1,N):
        values.append(f(values[i-1]))
        
    k = orbita(x0,M)
    if (k == 8):
        r=round(r,3)
        print("El valor de r para el que el conjunto atractor tiene 8 elementos es"
              ,r)
        r_values.append(r)
 
k = 8 
r = 3.961
values = []
values.append(x0)
for i in range(1,N):
    values.append(f(values[i-1]))
plt.plot(values)
plt.show()
VN, E = conjuntoAtractor(k, values)
print("El conjunto de puntos atractores es ", VN, ", con un error de ",E)
    