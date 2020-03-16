# -*- coding: utf-8 -*-
"""
Practica 3 - Clustering (DBSCAN)

Cristina Vilchez y Elena Perez
"""

import numpy as np

from sklearn.cluster import DBSCAN
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
# Los clasificamos mediante el algoritmo DBSCAN
epsilon = 0.3

'''
Funcion que ejecuta el algoritmo DBSCAN sobre un sistema
input:
        eps: umbral de distancia
        X: dataset a evaluar
		metric: metrica para el algoritmo DBSCAN
output:
		silhouette: coeficiente de silhouette
		labels: vector de etiquetas de los puntos del sistema
'''
def f_Dbscan(eps,X,metric):
	db = DBSCAN(eps=eps, min_samples=10, metric=metric).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	if(len(set(labels))==1):
		return -1, labels
	silhouette = metrics.silhouette_score(X, labels)
	return silhouette, labels

#Creamos el vector de epsilon en el intervalo (0.1, 0.99) con paso 0.001
eps_values = np.arange(0.1,0.99,0.001)

'''
Funcion que encuentra el epsilon con el que se obtiene el mejor valor del
coeficiente de Silhouette
input:
    metric: metrica para el algoritmo DBSCAN
    X: dataset a evaluar
    eps_values: valores de epsilon que se prueban
output:
    best_eps: epsilon optimo
    max_s: maximo valor del coeficiente de Silhouette
    s_values: vector con todos los valores de Silhouette obtenidos para todos los epsilon
'''
def findBestEps(metric, X, eps_values):
	s_values = []
	max_s = -1
	best_eps = 0 

	# Considerando la distancia euclidea, buscamos el epsilon optimo
	for eps in eps_values:
		s,_ = f_Dbscan(eps,X, metric)
		if (s>max_s):
			max_s=s
			best_eps = eps
		s_values.append(s)
	
	return best_eps, max_s, s_values
	
#Encontramos el mejor epsilon y coeficiente de Silhouette para cada metrica y
# representamos la grafica de los valores del coeficiente de Silhouette frente a epsilon
best_euclidean, s_euclidean, euclidean_values = findBestEps('euclidean', X, eps_values)
best_manhattan, s_manhattan, manhattan_values = findBestEps('manhattan', X, eps_values)

print("Para la distancia euclidea, el mejor epsilon es ", best_euclidean, " con un coeficiente de Silhouette de ", s_euclidean)
plt.plot(eps_values, euclidean_values)
plt.show()

print("Para la distancia manhattan, el mejor epsilon es ", best_manhattan, " con un coeficiente de Silhouette de ", s_manhattan)
plt.plot(eps_values, manhattan_values)
plt.show()

#Guardamos el vector de etiquetas para el epsilon optimo de cada metrica
_, labels_euclidean = f_Dbscan(best_euclidean, X, 'euclidean')
_, labels_manhattan = f_Dbscan(best_manhattan, X, 'manhattan')

'''
Funcion que muestra la informacion y la representacion grafica de los clusters
encontrados para el epsilon optimo
input:
    labels: vector con las etiquetas asociadas a cada uno de los puntos
    X: conjunto de todos los puntos
'''
def infoMetric(labels, X):
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)


	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)
	print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
	print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


	# #############################################################################
	# Representamos el resultado con un plot

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

	plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
	plt.show()
    
#Pintamos los resultados para las dos metricas estudiadas
print("Metric: euclidean")
infoMetric(labels_euclidean, X)

print("Metric: Manhattan")
infoMetric(labels_manhattan, X)