import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import getdata

def y2indicator(Y,K):
    N=len(Y)
    ind=np.zeros((N,K))
    for i in range(N):
        ind[i,Y[i]]=1
    return ind

X,Y=getdata() ##@ X is in shape of 500,4 and Y in shape of 500
X,Y=shuffle(X,Y)
Y=Y.astype(np.int32)
D=X.shape[1] ##@ get columns number of X matrix
K=len(set(Y)) ##@ number of diffrent value in Y is equall to number of class

print(Y.shape)
Xtrain=X[:-100]
Ytrain=Y[:-100]
Ytrain_ind=y2indicator(Ytrain,K)

Xtest=X[-100:]
Ytest=Y[-100:]
Ytest_ind=y2indicator(Ytest,K)

##@ DEFINE RANDOM WEIGHT
W=np.random.randn(D,K)
b=np.zeros(K)