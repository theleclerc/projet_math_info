# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:15:16 2019

@author: Théophile Le Clerc
"""

def find_seed(g, c=0, eps=2**(-26)):
    """
    recherche d'une solution de t de [0,1] tq g(t) = c,
    à condition que c soit dans [g(0),g(1)]
    """
    a, b = g(0),g(1)
    ta, tb = 0, 1
    if a > b:
        a, b = b, a
        ta, tb = tb, ta
    try:
        if c < a or c > b :
            raise ValueError
    except ValueError:
        return(None)
    #invariants: f(ta)=a<=c<=b=f(tb)
    t = (ta + tb)/2
    while abs(g(t)-c) > eps :
        if g(t) < c:
            a = g(t)
            ta = t
        else:
            b = g(t)
            tb = t
        t = (ta + tb)/2
    return t

import numpy as np

def simple_contour(f,c=0.0,delta = 0.01):
    """
    renvoie une partie de courbe de niveau
    """
    x = np.arange(0,1,delta)
    y = np.zeros(len(x),dtype = float)
    for i,xi in enumerate(x):
        yi = find_seed(lambda t : f(xi,t),c)
        if yi == None:
            y[i] = None
        else:
            y[i] = yi
    return(x,y)

def contour(f, c=0.0, xc=[0.0,1.0], yc=[0.0,1.0], delta=0.01):
    """
    renvoie des fragments pour chaque case du quadrillage (xc,yc)
    """
    nx = len(xc)
    ny = len(yc)
    xs = []
    ys = []
    for i in range(0,nx-1):
        xa,xb = xc[i],xc[i+1]
        for j in range(0,ny-1):
            ya,yb = yc[j],yc[j+1]
            def ftilde(tx,ty):
                """
                fonction de [0,1]X[0,1]
                """
                return(f(xa+(xb-xa)*tx,ya+(yb-ya)*ty))
            TX,TY = simple_contour(ftilde,c,delta)
            x,y = (xb-xa)*TX + xa, (yb-ya)*TY + ya
            xs.append(x)
            ys.append(y)
    return(xs,ys)

from math import *

def f(x,y):
    """
    fonction test
    """
    res = 2*(np.exp(-np.power(x,2)-np.power(y,2))-np.exp(-np.power(x-1,2)-np.power(y-1,2)))
    return(res)

def g(x,y):
    return(sqrt(x**2+y**2))

import matplotlib.pyplot as plt
import time

top = time.time()
for c in [-1.5,-1.,-0.5,0,0.5,1.,1.5]:
    xc, yc = np.linspace(-2.,3.,25), np.linspace(-1.,2.,25)
    xs, ys = contour(f,c,xc,yc)
    
    for x,y in zip(xs,ys):
        plt.plot(x,y,'-b', label ='c=5')
print(time.time()-top)
plt.show()
