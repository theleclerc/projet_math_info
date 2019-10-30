# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:34:43 2019

@author: Théophile Le Clerc
"""


from math import *
import numpy as np
import autograd
from autograd import numpy as np
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

def J_f(f,x,y):
    """
    donne le jacobien de f à valeurs dans R² au point de coordonnées (x,y)
    sous la forme d'une matrice carrée
    """
    j = autograd.jacobian
    return np.c_[j(f,0)(x,y), j(f,1)(x,y)]

def newton_2D(f, x0, y0, fx = 0, fy = 0, eps = 2**(-26)):
    """
    méthode de newton pour f une fonction de R² dans R²
    renvoie un couple (x0,y0) qui représente les coordonnées d'une solution au 
    problème f(x,y) = (fx,fy)
    """
    def J_f(x,y):
        """
        donne le jacobien de f au point de coordonnées (x,y)
        sous la forme d'une matrice carrée
        """
        j = autograd.jacobian
        return np.c_[j(f,0)(x,y), j(f,1)(x,y)]
    
    imx, imy = f(x0,y0)[0], f(x0,y0)[1]
    c = np.array([[fx],
                  [fy]])
    pos = np.array([[x0],
                    [y0]])
    im = np.array([[imx],
                    [imy]])
    while max(abs(imx - fx), abs(imy - fy)) > eps:
        J = J_f(x0,y0)
        a,b,c,d = J[0,0],J[0,1],J[1,0],J[1,1]
        #test inversibilite de J
        if a*d-b*c == 0:
            raise ValueError("matrice non inversible")
        pos = pos + np.dot(np.linalg.inv(J),c-im)
        x0,y0 = pos[0][0],pos[1][0]
        imx,imy = f(x0,y0)[0], f(x0,y0)[1]
        pos = np.array([[x0],
                    [y0]])
        im = np.array([[imx],
                    [imy]])
    
    return(x0,y0)

def simple_contour(f,c=0.0,delta = 0.01):
    """
    renvoie une partie de courbe de niveau
    """
    x,y = [],[]
    exist_seed = False
    i = 0
    while not exist_seed:
        if i * delta > 1:
            return(np.array(x),np.array(y))
        yi = find_seed(lambda t: f(i*delta,t),c)
        if yi == None:
            i += 1
        else:
            x.append(i*delta)
            y.append(yi)
            exist_seed = True
    xc, yc = x[-1], y[-1]
    try:
#        N = 0
        while True:
            grad = grad_f(f,xc,yc)
            tangent = np.array([-grad[1],grad[0]])/np.power(grad[0]**2 + grad[1]**2,1/2)*delta
            x0, y0 = xc + tangent[0], yc + tangent[1]
            def F(x,y):
                return np.array([f(x,y),np.power(x-x0,2)+np.power(y-y0,2)])
            xc, yc = newton_2D(F, x0, y0, fx = c, fy = delta**2)
            if xc<0 or xc>1 or yc<0 or yc >1:
                return(np.array(x),np.array(y))
            x.append(xc)
            y.append(yc)
#            N += 1
#            if N > 1000:
#                return(np.array(x),np.array(y))
    except ValueError as e:
        print(e)
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

def g(x,y):
    """
    seconde fonction de test
    """
    return(np.power((np.power(x,2)+np.power(y,2)),1),np.power(x,3))


for c in [1]:
    xc, yc = np.linspace(-2.,3.,10), np.linspace(-1.,2.,10)
    xs, ys = contour(f,c,xc,yc,delta=0.001)
    
    for x,y in zip(xs,ys):
        plt.plot(x,y,'-ob', label ='c=5')
plt.show()