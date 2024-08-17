# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:41:02 2021

@author: CHENTIEJUN
"""

# asPLS

import scipy.sparse as sp
import numpy as np

def bgasPLS(data,lambd=1e+7,ratio=1e-5,nDiff=2): #lambd-> 1e+5 ~ 1e+9   ratio = 1e-5
    
    nData,nLen = data.shape
    removeData = np.zeros((nData,nLen))
    bgData = np.zeros((nData,nLen))
    D = np.diff(np.eye(nLen),nDiff)
    H = np.dot(D,D.T)
    yzWyz = np.zeros((nData,100))
    zDDz = np.zeros((nData,100))
    it = 100*np.ones((nData,1))
    
    if np.size(lambd) ==1:
        lambd = np.tile(lambd,[nData,1])

    for i in range(nData):
        iteration = 20
        w = np.ones((nLen,1))
        alpha = np.ones((nLen,1))
        y = data[i,:]
        y = y.reshape(512,1)
        
        for ir in range(iteration):
            W = sp.spdiags(w.T,0,nLen,nLen)
            W = W.toarray()
            
            z = np.linalg.solve((W+alpha*lambd[i]*H),(y*w))
            
            d = y-z
            dn = d[d<0]
            if dn.size == 0:
                it[i] = ir - 1
                break
            
            #m = np.mean(dn)
            s = np.std(dn)
            alpha[:] = np.abs(d)/np.max(np.abs(d))
            wt = 1 / (1+np.exp(2*(d-s)/s))
            yzWyz[i,ir] = np.sum((wt*d)**2)
          
            zDDz[i,ir] = np.sum((D*z)**2)
            if (np.sum(w)-np.sum(wt)) / np.sum(w) < ratio:
                it[i] = ir
                break
            
            w = wt
            zz = z

        bgData[i,:] = zz.T
        removeData[i,:] = data[i,:]-bgData[i,:]
        
    return removeData,bgData,yzWyz,zDDz,it