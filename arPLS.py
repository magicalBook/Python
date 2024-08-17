# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:43:07 2021

@author: CHENTIEJUN
"""

# arPLS
import scipy.sparse as sp
import numpy as np

def bgarPLS(data,lambd=1e+5,ratio=1e-5,nDiff=2):
    
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
        w = np.ones((nLen,1))
        y = data[i,:]
        y = y.reshape(512,1)
        
        for ir in range(20):
            W = sp.spdiags(w.T,0,nLen,nLen)
            W = W.toarray()
            
            z = np.linalg.solve((W+lambd[i]*H),(y*w))
            
            d = y-z
            dn = d[d<0]
            if dn.size == 0:
                it[i] = ir - 1
                break
            
            m = np.mean(dn)
            s = np.std(dn)
            wt = 1 / (1+np.exp(2*(d-(2*s-m))/s))
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