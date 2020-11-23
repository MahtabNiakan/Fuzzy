from __future__ import print_function,division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def forward(X,W,b,V,c):
    Z = X.dot(W) + b
    Z = Z * (Z > 0)  # relu
    # Z = np.tanh(Z)

    Yhat = Z.dot(V) + c
    return Z, Yhat



def derivative_W(X,Z,Y,Yhat,V):
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # relu
    return X.T.dot(dZ)


def derivative_b(Z,Y,Yhat,V):
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # this is for relu activation
    return dZ.sum(axis=0)


def derivative_V(Z,Y,Yhat):
    return (Y-Yhat).dot(Z)

def derivative_c(Y,Yhat):
    return (Y-Yhat).sum()

def get_cost(Y,Yhat):
    return ((Y-Yhat)**2).mean()

def update(X,W,b,Z,V,c,Yhat,Y,learning_rate=0.00001):
    gradian_v=derivative_V(Z,Y,Yhat)
    gradian_w=derivative_W(X,Z,Y,Yhat,V)
    gradian_b=derivative_b(Z,Y,Yhat,V)
    gradian_c=derivative_c(Y,Yhat)

    V+=learning_rate*gradian_v
    W+=learning_rate*gradian_w
    b+=learning_rate*gradian_b
    c+=learning_rate*gradian_c

    return W,b,V,c

##@ generate data
N=500
X=np.random.random((N,2))*4-2 ##@ X BETWEEN -2,2

##@ FUNCTION IS :Y=X1*X2
Y=X[:,0]*X[:,1]


D=2
M=100
W=np.random.randn(D,M)/np.sqrt(D)

b=np.zeros(M)

V=np.random.randn(M)/np.sqrt(M)

c=0

costs=[]
for i in range(200):
    Z,Yhat=forward(X,W,b,V,c)
    W,b,V,c=update(X,W,b,Z,V,c,Yhat,Y)
    cost=get_cost(Y,Yhat)
    costs.append(cost)
    if i%10==0:
        print('cost is:',cost)


    ########################################  Here is point of learning model ########################################
             # The cost function hs converged .so we have obtain the weights which result in the minimum error
             # the last weigts in all layer are the best weight . so we insert this weights as weights which
             #model has learnt. Now we can use these weights for linear or nonlinear output

    ##################################################################################################################
plt.plot(costs)
plt.show()

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
plt.show()

##@ surface plot
##@ Making x,y array as input for testing how model classified data with model which has learnt
line=np.linspace(-20,20,50) ##@ return 50 point between -20 and 20
XX,YY=np.meshgrid(line,line) ##@return grid that intersection points come from lines point
Xgrid=np.vstack((XX.flatten(),YY.flatten())).T ##@ flatten concatet all element of array into one dimention
_,Yhat=forward(Xgrid,W,b,V,c) ##@ this W,b,V,c are the last and optomized data
ax.plot_trisurf(Xgrid[:,0],Xgrid[:,1],Yhat,linewidth=0.2, antialiased=True)
plt.show()

##@ residual scatter plot
Ygrid=Xgrid[:,0]*Xgrid[:,1]
R=np.abs([Ygrid-Yhat])
plt.scatter(Xgrid[:,0],Xgrid[:,1],R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=0.2, antialiased=True)
ax.scatter(Xgrid[:,0], Xgrid[:,1], R)

plt.show()
