"""
Practica 2 - Codigo Huffman

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
     
with open('auxiliar_es_pract2.txt', 'r',encoding='cp1252') as file:
      es = file.read()

#### Pasamos todas las letras a minúsculas
en = en.lower()
es = es.lower()

#### Contamos cuantas letras hay en cada texto
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)

##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))



## Ahora definimos una función que haga exáctamente lo mismo
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), 
                             axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), 
                             axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab, })
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)
 
    

#tree es el arbol para el texto en ingles
#arbol es el arbol para el texto en espanol
tree = huffman_tree(distr_en)
arbol = huffman_tree(distr_es)



###########################################
###########################################
###########################################


# EJERCICIO 1

## Hallar código Huffman de los textos (almacenados en 'en' y 'es')
'''
Funcion que genera  un diccionario caracter-codigo
a partir de un arbol de Huffman.
input:
    tree: arbol de Huffman
'''
def huffman_dict(tree):
        #Para rama del arbol, recorremos todos los caracteres clasificados en 
        #esa rama y añadimos 0 o 1 a su codigo en el diccionario segun
        #corresponda
        dictionary = {}
        for j in list(tree[tree.size-1].items())[0][0]:
            dictionary[j] = "0"
        for k in list(tree[tree.size-1].items())[1][0]:
            dictionary[k] = "1"
        for i in range(tree.size-2, 0,-1):
            for j in list(tree[i].items())[0][0]:
                dictionary[j] += "0"
            for k in list(tree[i].items())[1][0]:
                dictionary[k] += "1"
        return(dictionary)

#dictionary es el diccionario para el texto en ingles
#diccionario, en espanol
dictionary = huffman_dict(tree)
diccionario = huffman_dict(arbol)


#Diccionarios inversos: asocian a cada codigo (unico por ser una codificacion
#binaria prefijo) el caracter que codifica.
inv_dict = {v:k for k, v in dictionary.items()}
inv_dicc = {v:k for k, v in diccionario.items()}


#Traducimos los textos buscando cada caracter en el diccionario
translation=""
for c in en:
    translation += dictionary[c]
traduccion=""
for c in es:
    traduccion += diccionario[c]
    
    
'''
Funcion que calcula la longitud media de un sistema.
input: 
    dist: tabla de caracteres y su probabilidad
    dict: diccionario para esos caracteres en el idioma deseado
'''
def longitud_media(distr, dict):
    resultado = 0
    for i in range(len(distr)):
        prob = distr['probab'][i]
        long = len(dict[distr['states'][i]])
        resultado += prob*long
    return(resultado)

#Calculamos la longitud media para los dos idiomas       
L_English = longitud_media(distr_en,dictionary)
L_Espanol = longitud_media(distr_es,diccionario)
print("La longitud media de S_en es",L_English, " +/- ", error_comp)
print("La longitud media de S_es es",L_Espanol, " +/- ", error_comp)


#Teorema de Shannon: H(C) <= L(C) < H(C) + 1
'''
Funcion que calcula la entropia de un sistema 
input:
    prob: vector de probabilidades de cada caracter
'''
def entropiaSistema(prob):
    resultado = 0
    for i in prob:
        resultado += i*np.ceil(np.log2(i))
    return(-resultado)

#Calculamos la entropia de cad sistema
entropy = entropiaSistema(distr_en['probab'])
entropia = entropiaSistema(distr_es['probab'])

#Verificamos que se cumple el teorama de Shannon. Si se cumple, no imprime nada.
assert entropy <= L_English and L_English < entropy + 1, \
    "No se cumple el Primer Teorema de Shannon :("
assert entropia <= L_Espanol and L_Espanol < entropia + 1 , \
    "No se cumple el Primer Teorema de Shannon :(" 
 

###########################################
###########################################
###########################################


#EJERCICIO 2

'''
Funciones para codificar o decodificar un texto.
input:
    text: texto a traducir
    dict: diccionario del idioma
'''
def codificar(text,dict):
    resultado = ""
    for c in text:
        resultado += dict[c]
    return(resultado)

def decodificar(text,dict):
    #dict es el diccionario inverso (a cada codigo, un caracter)
    resultado = ""
    part = ""
    for c in text:
         part += c
         if part in dict:
             resultado += dict[part]
             part = ""
    return(resultado)

#Codificamos la palabra fractal
word = "fractal"
fractal_es = codificar(word,diccionario)
fractal_en = codificar(word,dictionary)

#Calculamos cuantos bit necesitaria para codificarse en cod.binaria habitual
#y comprobamos que Huffman es mas eficiente
usual_binary = len(word)*np.ceil(np.log2(len(distr_en)))
binario_usual = len(word)*np.ceil(np.log2(len(distr_es)))
assert len(fractal_es) < binario_usual and len(fractal_en) < usual_binary


###########################################
###########################################
###########################################



#EJERCICIO 3

#Traducimos el codigo que nos dio el profesor 
code = '1010100001111011111100'
word = decodificar(code, inv_dict)
print(code + " se traduce como " + word)

#Ahora codificamos la palabra 'hello', y la decodificamos para
#comprobar que funciona 
word = 'hello'
code = codificar(word, dictionary)
print(word + " se traduce como " + code)
word = decodificar(code, inv_dict)
print(code + " se traduce como " + word)
        
        