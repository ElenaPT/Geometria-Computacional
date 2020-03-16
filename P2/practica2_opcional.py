"""
Practica 2 - Codigo Huffman (Ejercicio Opcional)

Cristina Vilchez y Elena Perez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


error_comp = 0.000001

#### Carpeta donde se encuentran los archivos ####
#ubica = "/"

#### Vamos al directorio de trabajo####
#os.getcwd()
#os.chdir(ubica)
#files = os.listdir(ubica)

with open('auxiliar_en_pract2.txt', 'r', encoding='cp1252') as file:
      en = file.read()

#### Pasamos todas las letras a minusculas
en = en.lower()

#### Contamos cuantas letras hay en cada texto
from collections import Counter
tab_en = Counter(en)

##### Transformamos en formato array de los caracteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))


###########################################
###########################################
###########################################



#EJERCICIO 4

'''
Funcion que dibuja una grafica con la curva de Lorenz del sistema
frente a la recta y=x. Devuelve los valores normalizados de los ejes
xs (caracteres) e ys (probabilidad acumulada)
input: 
    prob: vector de probabilidades
'''
def curvaLorenz(prob):
    N = len(prob)
    xs = []
    ys = []
    acumulado = 0
    for i in range(N):
        xs.append(i/N)
        acumulado += prob[i]
        ys.append(acumulado)
    plt.plot(xs, ys)
    plt.plot(xs,xs,'r')
    plt.show()
    return xs,ys
 
'''
Funcion que calcula el indice de Gini
input:
    x,y: vectores con los valores de la grafica de Lorenz
'''
def indiceGini(x,y):
    sum = 0
    for i in range(1,len(x)):
        sum += (y[i]+y[i-1])*(x[i]-x[i-1])
    return (1-sum)
    
#Calculamos el indice de Gini para S_en
x,y = curvaLorenz(distr_en['probab'])
gini = indiceGini(x,y)
print("El indice de Gini de la variable S_en es",gini, " +/- ", error_comp)


'''
Funcion que calcula la diversidad-q de Hill de un sistema
input:
    q: parametro q
    prob: vector de probabilidades
'''
def diversidadHill(prob, q):
    if q == 1:
        sum = 0
        for i in prob:
            sum += i*np.log(i)
        return (np.e**-sum)
        
    sum = 0
    for i in prob:
        sum+=i**q
    return (sum**(1/(1-q)))

#Calculamos la diversidad-2 de Hill para S_en
D2_en = diversidadHill(distr_en['probab'],2)
print("La diversidad 2D de Hill es ",D2_en, " +/- ", error_comp)      
        
        
        
        
        