# -*- coding:utf-8 -*-
import numpy as np
np.set_printoptions(precision=32)

#M = np.random.randn(4,2)
#M = np.array([[1.0,2.0],[3.0,4.0],[5.0,6.0],[7.0,8.0]])
M = np.array([[1.5,2.5],[3.5,4.5],[5.5,6.5],[7.5,8.5]])

# nur positive werte wegen wurzel
for i in range(np.size(M,0)):
    for k in range(np.size(M,1)):
        if(M[i][k] < 0):
            M[i][k] *= -1

#quadratische matrix erstellen
height = np.size(M, 0) 
width = np.size(M, 1) 
topMatrix = np.zeros((height, height))
topMatrix = np.concatenate((topMatrix, M), axis=1)
bottomMatrix = np.zeros((width,width))
bottomMatrix = np.concatenate((np.transpose(M), bottomMatrix), axis=1)
A = np.concatenate((topMatrix, bottomMatrix), axis=0)
#DiagonalMatrix
DiagonalMatrix = np.zeros((np.size(A,0),np.size(A,1)))
for i,x in enumerate(A):
    rowSum = np.sum(x)
    if rowSum == 0:
        DiagonalMatrix[i][i] = 0     
    else:
        DiagonalMatrix[i][i] = np.sqrt(1.0 / rowSum)

#L errechnet mit numpys matrix multiplikation
L1 = DiagonalMatrix.dot(A).dot(DiagonalMatrix)

#Implementierung 2
L2 = A
dvec=np.sum(L2,axis=1)
dinv=[np.sqrt(1.0/el) for el in dvec]
for i in range(len(dinv)):
    L2[:,i]=L2[:,i]*dinv[i]
for i in range(len(dinv)):
    L2[i,:]= L2[i,:]*dinv[i]


print "L1 AND L2 EQUAL? " + str(np.array_equal(L1,L2))
print L1
print L2
