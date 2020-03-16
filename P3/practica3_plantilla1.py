# -*- coding: utf-8 -*-
"""
Practica 3 - Clustering (DBSCAN)

Cristina Vilchez y Elena Perez
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# #############################################################################
# Aqui tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)
#Si quisieramos estandarizar los valores del sistema, hariamos:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()

# #############################################################################
# Los clasificamos mediante el algoritmo KMeans

#Usamos la inicialización aleatoria "random_state=0"
''''
Funcion que ejecuta el algoritmo de k_means sobre un sistema
input:
        n: num clusters
        X: dataset a evaluar
output:
		silhouette: coeficiente de silhouette
		kmeans: objeto kmeans
'''

def f_kmeans(n_clusters,X):
    if (n_clusters == 1):
        return -1,None
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    silhouette = metrics.silhouette_score(X, labels)
    return silhouette, kmeans

'''
Funcion que encuentra el numero de clusters con el que se obtiene el mejor
valor del coeficiente de Silhouette
input:
    X: dataset a evaluar
output:
    best_k: numero de clusters optimo
    max_s: maximo valor del coeficiente de Silhouette
    s_values: vector con todos los valores de Silhouette obtenidos para todos los epsilon
'''
def findBestK(X):
    s_values = []
    max_s = -1
    best_k = 0 
    
    for k in range(1,15):
    	s,_ = f_kmeans(k,X)
    	if (s>max_s):
    		max_s=s
    		best_k = k
    	s_values.append(s)
        
    return best_k, max_s, s_values

best_k, max_s, s_values = findBestK(X)

plt.plot(list(range(1,15)),s_values)
plt.show()


# El valor de k que maximiza el coeficiente silhouette es 3. 
# Vamos a  mostrar las etiquetas, centros y el valor de silhouette para esta k
silhouette, kmeans = f_kmeans(best_k,X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print("Silhouette Coefficient: %0.3f" % silhouette)
print("Number of clusters: ", best_k)



#Prediccion de elementos para pertenecer a una clase:
problem = np.array([[0, 0], [1, -1]])
clases_pred = kmeans.predict(problem)
print(clases_pred)

# Representamos el resultado con un plot
labels = kmeans.labels_
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

plt.plot(problem[:,0],problem[:,1],'o', markersize=12, markerfacecolor="red")

plt.title('Fixed number of KMeans clusters: %d' % best_k)
plt.show()

# #############################################################################





