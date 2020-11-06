import numpy as np
from process import getdata

X,Y=getdata()
M=5 ##@ number of neurons in hidden layer(D+1)
D=X.shape[1]
K=len(set(Y)) ##@ set remove repeated item. just return number of none repeated
W1=np.random.randn(D,M)
b1=np.zeros(M)
W2=np.random.randn(M,K)
b2=np.zeros(K)

def softmax(a):
    expA=np.exp(a)
    return expA/expA.sum(axis=1,keepdims=True)

def Forward(X,W1,b1,W2,b2):
    Z=np.tanh(X.dot(W1)+b1)
    return softmax(Z.dot(W2)+b2)

P_y_given_x=Forward(X,W1,b1,W2,b2)
prediction=np.argmax(P_y_given_x,axis=1)

def classification_rate(Y,P):
    return np.mean(Y==P)
print('Final Score', classification_rate(Y,prediction))