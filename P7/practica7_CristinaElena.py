# -*- coding: utf-8 -*-
"""

Practica 7 - Transformacion Isometrica Afin

Cristina Vilchez y Elena Perez
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

vuestra_ruta = "/Users/CristinaVM/Desktop/Uni/GeoComp"

os.getcwd()
os.chdir(vuestra_ruta)


#####################

# EJERCICIO 1

#####################


#Nuestra figura a representar es un toro

n=120
theta = np.linspace(0, 2.*np.pi, n)
phi = np.linspace(0, 2.*np.pi, n)
theta, phi = np.meshgrid(theta, phi)
c, a = 2, 1
X = a * np.sin(theta)
Y = (c + a*np.cos(theta)) * np.sin(phi)
Z = (c + a*np.cos(theta)) * np.cos(phi)

fig = plt.figure()
ax = plt.axes(xlim=(-5,5), ylim=(-5,5),projection='3d')
ax.set_zlim(-3,3)
cset = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap = plt.cm.get_cmap('viridis'), edgecolors='w')
ax.clabel(cset, fontsize=9, inline=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


"""
Funcion que aplica una transformacion afin a un sistema
input:
    x,y,z= coordenadas de los puntos del sistema - en vector unidimensional
    M,v = parametros de la transformacion x' = M(x-c) + v
    c = centroide del sistema
"""
def transf1D(x,y,z,M, v=np.array([0,0,0]), c=np.array([0,0,0])):
    xt = np.empty(len(x))
    yt = np.empty(len(x))
    zt = np.empty(len(x))
    for i in range(len(x)):
        q = np.array([x[i]-c[0],y[i]-c[1],z[i]-c[2]])
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v + c
    return xt, yt, zt


'''
Funcion que calcula el centroide de un sistema 3D
input:
    x,y,z= coordenadas de los puntos del sistema - en vector unidimensional
output
    coordenadas (x,y,z) del centroide
'''
def centroide(x,y,z):
    xc = 0
    yc = 0
    zc = 0
    for i in range(len(x)):
        xc += x[i]
        yc += y[i]
        zc += z[i]
    return xc/len(x),yc/len(x),zc/len(x)


'''
Funcion que genera un fotograma para la animacion de una transformacion afin
input:
    t: tiempo del fotograma
'''
def animate(t):
    M = np.array([[np.cos(t*3*np.pi),-np.sin(t*3*np.pi),0],[np.sin(t*3*np.pi),np.cos(t*3*np.pi),0],[0,0,1]])
    v=np.array([d,d,0])*t
    
    ax = plt.axes(xlim=(0,10), ylim=(0,10), projection='3d')
    ax.set_zlim(-3,3)
    #ax.view_init(60, 30)

    XYZ = transf1D(x0, y0, z0, M=M, v=v, c=cent)
    col = plt.get_cmap("viridis")(np.array(0.1+XYZ[2]))
    X = XYZ[0].reshape(120,120)
    Y = XYZ[1].reshape(120,120)
    Z = XYZ[2].reshape(120,120)
    cset = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap = plt.cm.get_cmap('viridis'))
    return ax,

def init():
    return animate(0),

'''
Funcion que calcula el diametro de una figura en 3D
input:
    x0,y0,z0= coordenadas de los puntos del sistema - en vector unidimensional
'''
def diam(x0,y0,z0):

    #Calculamos los puntos de la envolvente convexa O(N log N)
    points = np.array([x0,y0,z0]).transpose()
    hull = ConvexHull(points)
    
    # Extract the points forming the hull
    hullpoints = points[hull.vertices,:]
    
    fig = plt.figure()
    
    ax = plt.axes(xlim=(-5,5), ylim=(-5,5), projection='3d')
    ax.set_zlim(-3,3)
    cset = ax.scatter3D(hullpoints[:,0], hullpoints[:,1], hullpoints[:,2], color='r',edgecolors='w',zorder=3)
    ax.clabel(cset, fontsize=9, inline=1)

    
    X = x0.reshape(120,120)
    Y = y0.reshape(120,120)
    Z = z0.reshape(120,120)
   #cset = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap = plt.cm.get_cmap('viridis'), edgecolors='w')
    ax.clabel(cset, fontsize=9, inline=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    
    # El diametro lo calculamos como la mayor distancia entre los puntos de la envolvente
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    
    p1 = hullpoints[bestpair[0]]
    p2 = hullpoints[bestpair[1]]
    
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)
    


#Convertimos nuestros vectores de coordenadas en vectores unidimensionales
#para poder utilizar las funciones anteriores
x0 = X.flatten()
y0 = Y.flatten()
z0 = Z.flatten()

#Calculamos centroide
cent = centroide(x0,y0,z0)


#Calculamos diametro mayor de nuestra figura. En este caso, sabemos que es 6
d = 6

animate(np.arange(0.1, 1,0.1)[5])
plt.show()

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.05), init_func=init,
                              interval=20)
ani.save("ejercicio1.gif", fps = 5) 




#########################################

# EJERCICIO 2

#########################################



'''
Funcion que calcula el diametro de una figura en 2D
input:
    x0,y0,z0= coordenadas de los puntos del sistema - en vector unidimensional
'''
def diam2D(x0,y0):

    #Calculamos los puntos de la envolvente convexa O(N log N)
    points = np.array([x0,y0]).transpose()
    hull = ConvexHull(points)
    
    # Extract the points forming the hull
    hullpoints = points[hull.vertices,:]
    
    fig = plt.figure()
    
    ax = plt.axes(xlim=(0,400), ylim=(0,400), projection='3d')
    ax.set_zlim(-3,3)
    cset = ax.scatter(hullpoints[:,0], hullpoints[:,1], color='r')
    ax.clabel(cset, fontsize=9, inline=1)
    
    cset = ax.scatter(x0,y0,c=col,s=0.1,animated=True)
    ax.clabel(cset, fontsize=9, inline=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    
    # El diametro lo calculamos como la mayor distancia entre los puntos de la envolvente
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    
    p1 = hullpoints[bestpair[0]]
    p2 = hullpoints[bestpair[1]]
    
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    
'''
Funcion que calcula el centroide de un sistema 3D
input:
    x,y,z= coordenadas de los puntos del sistema - en vector unidimensional
output
    coordenadas (x,y,z) del centroide
'''
def centroide2D(x,y):
    xc = 0
    yc = 0
    for i in range(len(x)):
        xc += x[i]
        yc += y[i]
    return xc/len(x),yc/len(x)


#Importamos la imagen

img = io.imread('arbol.png')
dimensions = color.guess_spatial_dimensions(img)
print(dimensions)
io.show()
#io.imsave('arbol2.png',img)

#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
fig = plt.figure(figsize=(5,5))
p = plt.contourf(img[:,:,0],cmap = plt.cm.get_cmap('viridis'), levels=np.arange(0,240,2))
plt.axis('off')
#fig.colorbar(p)


#xyz = (350,350,4)  pixel x pixel x colores
xyz = img.shape



#x: pixeles del eje x
#y: pixeles del eje y
#xx,yy: indexando en estos dos vectores, recorres todos los pixeles de la imagen
#zz: color rojo de cada pixel
x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,0]
zz = np.asarray(z).reshape(-1)



#Consideraremos sólo los elementos con color rojo  < 240 
#Variables de estado coordenadas
x0 = xx[zz<240]
y0 = yy[zz<240]
z0 = zz[zz<240]/256.
#Variable de estado: color
col = plt.get_cmap("viridis")(np.array(0.1+z0))

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1, 2, 1)
plt.contourf(x,y,z,cmap = plt.cm.get_cmap('viridis'), levels=np.arange(0,240,2))
ax = fig.add_subplot(1, 2, 2)
plt.scatter(x0,y0,c=col,s=0.1)
plt.show()



#Calculamos el centroide de la figura, teniendo en cuenta la componente de color
centr3D = centroide(x0,y0,z0)
centr2D = centroide2D(x0,y0)


'''
Funcion que genera un fotograma para la animacion de una transformacion afin
dentro de un plano (XY)
input:
    t: tiempo del fotograma
'''
def animate(t):
    M = np.array([[np.cos(t*3*np.pi),-np.sin(t*3*np.pi),0],[np.sin(t*3*np.pi),np.cos(t*3*np.pi),0],[0,0,1]])
   
    v=np.array([d2D,d2D,0])*t
    
    ax = plt.axes(xlim=(0,400), ylim=(0,400), projection='3d')
    #ax.view_init(60, 30)

    XYZ = transf1D(x0, y0, z0, M=M, v=v, c = [centr2D[0],centr2D[1],0])
    col = plt.get_cmap("viridis")(np.array(0.1+XYZ[2]))
    ax.scatter(XYZ[0],XYZ[1],c=col,s=0.1,animated=True)
    return ax,

def init():
    return animate(0),

#Calculamos el diametro sin tener en cuenta la coordenada de color (vector constante)
d2D = diam2D(x0,y0)


animate(np.arange(0.1, 1,0.1)[5])
plt.show()

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.05), init_func=init,
                              interval=20)
ani.save("ejercicio2.gif", fps = 5) 









