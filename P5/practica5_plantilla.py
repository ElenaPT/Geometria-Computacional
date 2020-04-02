
"""
Practica 5 - Deformación de variedades diferenciales

Cristina Vilchez y Elena Perez
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


############################################
#EJERCICIO 1
############################################


#Definimos nuestra curva: curva de Lissajous

fig = plt.figure()
ax = plt.axes(projection='3d')


#p y q son parámetros modificables
#m=0 contenida en una esfera; m=1 contenida en un cilindro
#r radio de la esfera
p = 7
q = 4
m = 0
r = 1
t2 = np.linspace(0, 2*np.pi, 360)
div = np.sqrt(1+(1-m)**2 * np.sin(p*t2)**2)
x2 = r*np.cos(q*t2)/div
y2 = r*np.sin(q*t2)/div
z2 = r*np.sin(p*t2)/div

c2 = x2 + y2

ax.scatter(x2, y2, z2, c=c2)
ax.plot(x2, y2, z2, '-b')


#Dibujamos la curva sobre la esfera de radio 1


u = np.linspace(0, np.pi, 25)#75
v = np.linspace(0, 2 * np.pi, 50)#50

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot(x2, y2, z2, '-b', c="gray", zorder=3)
ax.set_title('surface');


#Proyectamos la esfera sobre el plano z=1

"""
Proyección estereográfica 
input:
    - x: coordenada x
    - z: coordenada z
    - z0: punto que se envia al infinito
    - alpha: tasa de deformacion
output:
    - x_trans: x proyectada
"""

def proj(x,z,z0=-1,alpha=1):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = x/(abs(z0-z)**alpha+eps)
    return(x_trans)
    #Nótese que añadimos un épsilon para evitar dividi entre 0!!

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot(x2, y2, z2, '-b',c="gray", zorder=3)
ax.set_title('2-sphere');
#ax.text(0.5, 90, 'PCA-'+str(i), fontsize=18, ha='center')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_xlim3d(-8,8)
ax.set_ylim3d(-8,8)
#ax.set_zlim3d(0,1000)
ax.plot_surface(proj(x,z), proj(y,z), z*0-1, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot(proj(x2,z2), proj(y2,z2), -1, '-b',c="gray", zorder=3)
ax.set_title('Stereographic projection');

plt.show()
#fig.savefig('C:/Users/Robert/Dropbox/Importantes_PisoCurro/Universitat/Profesor Asociado/GCOM/LaTeX/stereo2.png')   # save the figure to file
plt.close(fig) 





############################################
#EJERCICIO 2
############################################


"""
Proyección estereográfica con la familia paramétrica del enunciado

input:
    - x: coordenada x
    - z: coordenada z
    - t: valor de t, parámetro de la familia paramétrica
output:
    - x_trans: x proyectada
"""

def proj2(x,z,t):
    eps = 1e-16
    x_trans = x/(1-t+t*abs(-1-z)+eps)
    return(x_trans)



"""
HACEMOS LA ANIMACIÓN
"""

from matplotlib import animation
#from mpl_toolkits.mplot3d.axes3d import Axes3D


def animate(t):
    xt = proj2(x,z,t)
    yt = proj2(y,z,t)
    zt=-t+z*(1-t)
    x2t = proj2(x2,z2,t)
    y2t = proj2(y2,z2,t)
    z2t=-t+z2*(1-t)
    
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b',c="gray", zorder=3)
    return ax,

def init():
    return animate(0),

animate(np.arange(0, 1,0.1)[1])
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.05), init_func=init,
                              interval=20)
ani.save("ejercicio2.gif", fps = 5) 





############################################
#EJERCICIO OPCIONAL
############################################

"""
Proyección estereográfica con familia paramétrica nueva
Para t = 0 -> identidad
Para t = 1 -> z=-1 se lleva al infinito

input:
    - x: coordenada x
    - z: coordenada z
    - t: valor de t, parámetro de la familia paramétrica
output:
    - x_trans: x proyectada
"""

def proj3(x,z,t):
    eps = 1e-16
    #x_trans = x/(abs(np.tan(1-t+t*abs(-1-z))*(np.tan(1+t/2)**(-1)))+eps)
    x_trans = x/(t*np.tan(1-t+t*abs(-1-z))+(1-t)*np.arctan(1))
    return(x_trans)


def animate(t):
    xt = proj3(x,z,t)
    yt = proj3(y,z,t)
    #zt = (z*0+z0)*(1-t) + z*t
    zt=-t+z*(1-t)
    x2t = proj3(x2,z2,t)
    y2t = proj3(y2,z2,t)
    #z2t = (z0)*(1-t)+z2*t
    z2t=-t+z2*(1-t)
    
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b',c="gray", zorder=3)
    return ax,


def init():
    return animate(0),

animate(np.arange(0, 1,0.1)[1])
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.05), init_func=init,
                              interval=20)
ani.save("ejercicio3.gif", fps = 5) 


