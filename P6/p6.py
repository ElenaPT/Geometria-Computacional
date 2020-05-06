# -*- coding: utf-8 -*-
"""
Practica 6 - Espacio Fasico

Elena Perez y Cristina Vilchez
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps
from numpy import trapz



'''
Funcion que calcula la derivada como limite como cociente de diferencias

input:
        q: vector de posiciones
        dq0: valor inicial de la derivada
        d: granularidad del parametro temporal

output:
        dq: vector de derivadas
'''
def deriv(q,dq0,d):
   #dq = np.empty([len(q)])
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq,0,dq0) #dq = np.concatenate(([dq0],dq))
   return dq


'''
Funcion F - Ecuaciones de Hamilton-Jacobi para oscilador no lineal

input:
        q: vector de posiciones

output:
        ddq: vector de derivadas segundas
'''
def F(q):
    ddq = -2*q*(q**2-1)
    return ddq


'''
Funcion que resuelve la ecuación dinámica ddq = F(q), obteniendo la orbita q(t)

input:
        n: numero de puntos de la orbita
        q0: posicion inicial
        dq0: valor inicial de la derivada
        F: funcion del sistema
        d: granularidad del parametro temporal

output:
        q: vector de posiciones calculado
'''
def orb(n,q0,dq0,F, args=None, d=0.001):
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q


'''
Funcion que calcula el periodo de un vector de datos

input:
        q: vector de datos
        d: granularidad
        max: si se quiere calcular ondas por picos o valles

output:
        pers: distancia entre picos/valles
        waves: indices de los picos/valles
'''
def periodos(q,d,max=True):
    #Si max = True, tomamos las ondas a partir de los máximos/picos
    #Si max == False, tomamos los las ondas a partir de los mínimos/valles
    epsilon = 5*d
    dq = deriv(q,dq0=None,d=d) #La primera derivada es irrelevante
    if max == True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q >0))
    if max != True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q <0))
    diff_waves = np.diff(waves)
    waves = waves[0][1:][diff_waves[0]>1]
    pers = diff_waves[diff_waves>1]*d
    return pers, waves


#################################################################    
#  EJERCICIO 1
#  CALCULO DE ÓRBITAS
################################################################# 

#Vamos a ver qué delta en el intervalo [10**-4,10**-3] da mejores resultados 
#(una granularidad mayor para los puntos calculados de la orbita)
q0 = 0.
dq0 = 1.
fig, ax = plt.subplots(figsize=(12,5))
plt.ylim(-1.5, 1.5)  
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("q(t)", fontsize=12)
iseq = np.array([3,3.1,3.5,3.8,4])
for i in iseq:
    d = 10**(-i)
    n = int(32/d)
    t = np.arange(n+1)*d
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    plt.plot(t, q, 'ro', markersize=0.5/i,label='$\delta$ ='+str(np.around(d,4)),c=plt.get_cmap("winter")(i/np.max(iseq)))
    ax.legend(loc=3, frameon=False, fontsize=12)
#plt.savefig('Time_granularity.png', dpi=250)



#Nos quedamos con d = 10**-4. Calculamos q(t) y p(t)
#
q0 = 0.
dq0 = 1.
d = 10**(-4)
n = int(32/d)
t = np.arange(n+1)*d
q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
dq = deriv(q,dq0=dq0,d=d)
p = dq/2


#################################################################    
#  ESPACIO FÁSICO
################################################################# 

'''
Funcion que dibuja una orbita del espacio fasico

input:
        n: numero de puntos de la orbita
        q0: posicion inicial
        dq0: valor inicial de la derivada
        F: funcion del sistema
        d: granularidad del parametro temporal

output:
        q: vector de posiciones calculado
'''
def simplectica(q0,dq0,F,col=0,d = 10**(-4),n = int(16/d),marker='-'): 
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    plt.plot(q, p, marker,c=plt.get_cmap("winter")(col), linewidth=0.5)


fig = plt.figure(figsize=(8,5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

#Dibujamos el espacio fasico para un total de 12*12 puntos iniciales
seq_q0 = np.linspace(0.,1.,num=10)
seq_dq0 = np.linspace(0.,2,num=10)
for i in range(len(seq_q0)):
    for j in range(len(seq_dq0)):
        q0 = seq_q0[i]
        dq0 = seq_dq0[j]
        ax = fig.add_subplot(1,1, 1)
        col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
        #ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
        simplectica(q0=q0,dq0=dq0,F=F,col=col,marker='ro',d= 10**(-4),n = int(16/d))
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
#fig.savefig('Simplectic.png', dpi=250)
plt.show()


#################################################################   
#  EJERCICIO 2 


#  CÁLCULO DEL ÁREA DEL ESPACIO FÁSICO
#################################################################  


'''
Funcion que calcula el area contenida en una orbita
input:
        q0, dq0: valores iniciales
        d: granularidad temporal
        F: funcion del sistema
        n: numero de puntos
'''
def area(q0,dq0,d,n, F):
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    
    fig, ax = plt.subplots(figsize=(5,5)) 
    plt.rcParams["legend.markerscale"] = 6
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    plt.plot(q, p, '-')
    plt.show()
    
    #Tomaremos los periodos de la órbita, que definen las ondas
    T, W = periodos(q,d,max=False)
    #Nos quedamos con el primer trozo
    #Tomamos la mitad de la "curva cerrada" para integrar más fácilmente
    mitad = np.arange(W[0],W[0]+np.int((W[1]-W[0])/2),1)
    
    # Regla del trapezoide
    areaT = 2*trapz(p[mitad],q[mitad])
    
    # Regla de Simpson
    areaS = 2*simps(p[mitad],q[mitad])
    return areaT, areaS



#Tomamos un par (q(0), p(0)) y nos quedamos sólo en un trozo/onda de la órbita, sin repeticiones
#Para eso, tomaremos los periodos de la órbita, que definen las ondas
    
#Paso1: Buscamos las condiciones iniciales que minimizan en área. 
#Como el (0,0) es punto inestable, cogemos un punto cercano
q0 = 0.
dq0 = 10**(-10)
d = 10**(-4)
n = int(60/d)
t = np.arange(n+1)*d

areaMinT, areaMinS = area(q0,dq0,d,n,F)


print("Area con regla de Simpson =", areaMinS)
    
print("Area con regla del trapezoide =", areaMinT)


#Paso2: Buscamos las condiciones iniciales que maximizan en área
#Vamos a coger un punto de la frontera, (0,1)
q0 = 0.
dq0 = 2.
n = int(32/d)
t = np.arange(n+1)*d


areaMaxT, areaMaxS = area(q0,dq0,d,n,F)


print("Area con regla de Simpson =", areaMaxS)
    
print("Area con regla del trapezoide =", areaMaxT)


# El area total es el area maxima menos el area minima

areaTotalT = areaMaxT-areaMinT/2
print("Area total Trapezoide =", areaTotalT)
areaTotalS = areaMaxS-areaMinS/2
print("Area total Simpson =", areaTotalS)



####################################

#   CALCULO DEL ERROR

####################################


#Paso1: Buscamos las condiciones iniciales que minimizan en área. 
#Como el (0,0) es punto inestable, cogemos un punto cercano
q0 = 0.
dq0 = 10**(-10)
d = 10**(-5)
n = int(100/d)
t = np.arange(n+1)*d

areaMinT, areaMinS = area(q0,dq0,d,n,F)


print("Area con regla de Simpson =", areaMinS)
    
print("Area con regla del trapezoide =", areaMinT)


#Paso2: Buscamos las condiciones iniciales que maximizan en área
#Vamos a coger un punto de la frontera, (0,1)
q0 = 0.
dq0 = 2.
n = int(32/d)
t = np.arange(n+1)*d


areaMaxT, areaMaxS = area(q0,dq0,d,n,F)


print("Area con regla de Simpson =", areaMaxS)
    
print("Area con regla del trapezoide =", areaMaxT)


# El area total es el area maxima menos el area minima

areaTotalT_5 = areaMaxT-areaMinT/2
print("Area total Trapezoide =", areaTotalT_5)
areaTotalS_5 = areaMaxS-areaMinS/2
print("Area total Simpson =", areaTotalS_5)


#Para calcular el error, restamos las dos areas calculadas
error = max(abs(areaTotalT-areaTotalT_5),abs(areaTotalS-areaTotalS_5))

print("El error del area es ", error)




#####################
# Teorema de Liouville
#####################


'''
Funcion que calcula el area encerrada por un conjunto de puntos

input:
    x : lista de primera coordenada de los puntos
    y: lista de segunda coordenada

output:
    valor del área
'''
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


#Generamos los puntos del borde de D0 ([0,1]x[0,1])
x_d0 = []
y_d0 = []
for i in np.arange(0,1,0.1):
    x_d0.append(i)
    y_d0.append(0)
for i in np.arange(0,1,0.1):
    x_d0.append(1)
    y_d0.append(i)    
for i in np.arange(1,0,-0.1):
    x_d0.append(i)
    y_d0.append(1)
for i in np.arange(1,-0.1,-0.1):
    x_d0.append(0)
    y_d0.append(i)
    
print(PolyArea(x_d0,y_d0))



#Evolucionamos cada uno de los puntos a lo largo del tiempo.
#Guardamos en (qt,pt) el valor del punto inicial en varios tiempos (0,1000,5000,9000)
qt = []
pt = []
for i in range (0,len(x_d0)):   
   q =  orb(10000,x_d0[i],2*y_d0[i],F=F, d=10**-4)
   dq = deriv(q,y_d0[i]*2,10**-4)
   p = dq/2
   qt.append([q[0],q[1000],q[5000],q[9000]])
   pt.append([p[0],p[1000],p[5000],p[9000]])



fig, ax = plt.subplots(figsize=(5,5)) 
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
plt.plot(qt, pt, '-')
plt.show()   
   

#Mostramos que las áreas son (casi) constantes
for i in range(0,len(qt[0])):
    print(PolyArea(np.array(qt)[:,i],np.array(pt)[:,i]))



