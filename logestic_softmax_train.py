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

def softmax(a):
    expA=np.exp(a)
    return expA/expA.sum(axis=1,keepdims=True)

def forward(X,W,b):
    return softmax(X.dot(W)+b)

def predic(p_y_given_x):
    return np.argmax(p_y_given_x)

def classification_rate(Y,P):
    return np.mean(Y==P)

def cross_entropy(T,predict_Y):
    return -np.mean(T*np.log(predict_Y))

train_cost=[]
test_cost=[]
learning_rate=0.0001
for i in range(100000):
    P_ytrain=forward(Xtrain,W,b)
    P_ytest=forward(Xtest,W,b)
    a=Ytrain_ind.shape
    n=P_ytrain.shape
    Ctrain=cross_entropy(Ytrain_ind,P_ytrain) ## find cost of train
    Ctest=cross_entropy(Ytest_ind,P_ytest)    ##find cost of test data

    train_cost.append(Ctrain)
    test_cost.append(Ctest)

    W=W-learning_rate*Xtrain.T.dot(P_ytrain-Ytrain_ind)
    b=b-learning_rate*(P_ytrain-Ytrain_ind).sum(axis=0)
    if i%1000==0:
        print(i,Ctrain,Ctest)
print('final training model classification rate is',classification_rate(Ytrain,P_ytrain))
print('final testing model classification rate is',classification_rate(Ytest,P_ytest))

legend1,=plt.plot(train_cost)
legend2,=plt.plot(test_cost)
plt.legend([legend1,legend2])
plt.show()