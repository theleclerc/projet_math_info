# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:44:45 2019

@author: Théophile Le Clerc
"""

from math import *
import numpy as np
import autograd
from autograd import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt



def f(x,y):
    """
    fonction test
    """
    res = 2*(np.exp(-np.power(x,2)-np.power(y,2))-np.exp(-np.power(x-1,2)-np.power(y-1,2)))
    return(res)


def find_seed(g, c=0, eps=2**(-26)):
    """
    recherche d'une solution de t de [0,1] tq g(t) = c,
    à condition que c soit dans [g(0),g(1)]
    solution renvoyé sous la forme d'un float
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


def grad_f(f,x,y):
    """
    donne le gradient de la fonction à valeurs réelles au point de coordonnées (x,y)
    sous la forme d'un vecteur ligne
    """
    g = autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]


def simple_contour(f,c=0.0,delta = 0.01,):
    """
    renvoie une partie de courbe de niveau
    """
    maxiter = int(10/delta)
    x,y = [],[]
    exist_seed = False
    i = 0
    while not exist_seed:
        if i * delta > 1:
            return(np.array([],dtype = float),np.array([],dtype = float))
        yi = find_seed(lambda t: f(i*delta,t),c)
        if yi == None:
            i += 1
        else:
            x.append(i*delta)
            y.append(yi)
            exist_seed = True
    Xc, Yc = x[-1], y[-1]
    N = 0
    xc,yc = Xc, Yc
    while N < maxiter:
        grad = grad_f(f,xc,yc)
        tangent = np.array([-grad[1],grad[0]])/np.power(grad[0]**2 + grad[1]**2,1/2)*delta
        x0, y0 = xc + tangent[0], yc + tangent[1]
        def F(t):
            return (f(x0+t*grad[0],y0+t*grad[1])-c)
        tc = newton(F, 0)
        xc, yc = x0 + tc*grad[0], y0 + tc*grad[1]
        if xc<0 or xc>1 or yc<0 or yc >1:
            N = maxiter
        else:
            x.append(xc)
            y.append(yc)
            N += 1
    x.reverse()
    y.reverse()
    N = 0
    xc,yc = Xc, Yc
    while N <= maxiter:
        grad = grad_f(f,xc,yc)
        tangent = np.array([grad[1],-grad[0]])/np.power(grad[0]**2 + grad[1]**2,1/2)*delta
        x0, y0 = xc + tangent[0], yc + tangent[1]
        def F(t):
            return (f(x0+t*grad[0],y0+t*grad[1])-c)
        tc = newton(F, 0)
        xc, yc = x0 + tc*grad[0], y0 + tc*grad[1]
        if xc<0 or xc>1 or yc<0 or yc >1:
            return(np.array(x),np.array(y))
        x.append(xc)
        y.append(yc)
        N += 1
    return(np.array(x),np.array(y))


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

import time

top = time.time()
for i,c in enumerate([-1.5,-1.,-0.5,0,0.5,1.,1.5]):
    xc, yc = np.linspace(-2.,3.,20), np.linspace(-1.,2.,20)
    xs, ys = contour(f,c,xc,yc)
    for x,y in zip(xs,ys):
        plt.plot(x,y,'b')

print(time.time()-top)
plt.show()
