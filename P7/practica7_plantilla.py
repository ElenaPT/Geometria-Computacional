# -*- coding: utf-8 -*-
"""

Practica 7 - 

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


"""
Ejemplo para el apartado 1.

Modifica la figura 3D y/o cambia el color
https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
"""


fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, 20, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
ax.clabel(cset, fontsize=9, inline=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


"""
Transformación para el segundo apartado

NOTA: Para el primer aparado es necesario adaptar la función o crear otra similar
pero teniendo en cuenta más dimensiones
"""


def transf1D(x,y,z,M, v=np.array([0,0,0]), c=np.array([0,0,0])):
    xt = np.empty(len(x))
    yt = np.empty(len(x))
    zt = np.empty(len(x))
    for i in range(len(x)):
        q = np.array([x[i]-c[0],y[i]-c[1],z[i]-c[2]])
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v
    return xt, yt, zt


def centroide(x,y,z):
    xc = 0
    yc = 0
    zc = 0
    for i in range(len(x)):
        xc += x[i]
        yc += y[i]
        zc += z[i]
    return xc/len(x),yc/len(x),zc/len(x)

def animate(t):
    M = np.array([[np.cos(t*3*np.pi),-np.sin(t*3*np.pi),0],[np.sin(t*3*np.pi),np.cos(t*3*np.pi),0],[0,0,1]])
    v=np.array([d,d,0])*t
    
    ax = plt.axes(xlim=(0,40), ylim=(0,40), projection='3d')
    #ax.view_init(60, 30)

    XYZ = transf1D(x0, y0, z0, M=M, v=v, c=cent)
    col = plt.get_cmap("viridis")(np.array(0.1+XYZ[2]))
    X = XYZ[0].reshape(120,120)
    Y = XYZ[1].reshape(120,120)
    Z = XYZ[2].reshape(120,120)
    cset = ax.contour(X, Y, Z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
    return ax,

def init():
    return animate(0),



x0 = X.flatten()
y0 = Y.flatten()
z0 = Z.flatten()

cent = centroide(x0,y0,z0)

######
#Calculamos diametro mayor de nuestra figura
######

def diam(x0,y0,z0):

    # Find a convex hull in O(N log N)
    points = np.array([x0,y0,z0]).transpose()
    
    hull = ConvexHull(points)
    
    # Extract the points forming the hull
    hullpoints = points[hull.vertices,:]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    cset = ax.scatter3D(hullpoints[:,0], hullpoints[:,1], hullpoints[:,2], cmap = plt.cm.get_cmap('viridis'))
    ax.clabel(cset, fontsize=9, inline=1)
    
    cset = ax.contour(x0, y0, z0, 120, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
    ax.clabel(cset, fontsize=9, inline=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    
    # Naive way of finding the best pair in O(H^2) time if H is number of points on
    # hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    
    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    
    #Print them
    p1 = hullpoints[bestpair[0]]
    p2 = hullpoints[bestpair[1]]
    
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)
    

d = diam(x0,y0,z0)

animate(np.arange(0.1, 1,0.1)[5])
plt.show()

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.05), init_func=init,
                              interval=20)
ani.save("ejercicio3.gif", fps = 5) 




#########################################

# SEGUNDO APARTADO

#########################################


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

xyz = img.shape

x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,0]
zz = np.asarray(z).reshape(-1)


"""
Consideraremos sólo los elementos con zz < 240 

Por curiosidad, comparamos el resultado con contourf y scatter!
"""
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


centr = centroide(x0,y0,z0)


def animate(t):
    M = np.array([[np.cos(t*3*np.pi),-np.sin(t*3*np.pi),0],[np.sin(t*3*np.pi),np.cos(t*3*np.pi),0],[0,0,1]])
   
    v=np.array([d,d,0])*t
    
    ax = plt.axes(xlim=(0,400), ylim=(0,400), projection='3d')
    #ax.view_init(60, 30)

    XYZ = transf1D(x0, y0, z0, M=M, v=v, c = centr)
    col = plt.get_cmap("viridis")(np.array(0.1+XYZ[2]))
    ax.scatter(XYZ[0],XYZ[1],c=col,s=0.1,animated=True)
    return ax,

def init():
    return animate(0),


d = diam(x0,y0,z0)

z_aux = np.zeros(len(x0))

animate(np.arange(0.1, 1,0.1)[5])
plt.show()

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.05), init_func=init,
                              interval=20)
ani.save("ejercicio3.gif", fps = 5) 









