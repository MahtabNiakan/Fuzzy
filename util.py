import numpy as np
import pandas as pd

def init_weight_bios(M1,M2):
    W=np.random.randn(M1,M2)/np.sqrt(M1*M2) ##@ sqr(m1*m2) is starndard deviation
    b1=np.zeros(M2)
    return W.astype(np.float32),b1.astype(np.float32) ##@ convert to flat32 for using in tensor flew and theano

def init_filter(shape,poolsz):
    W=np.random.randn(*shape)
def relu(x)  :
    return x*(x>0)
def sigmoid(A):
    return 1/(1+np.exp(-A))
def softmax(A):
    return np.exp(A)/np.exp(A).sum(axis=1,keepdim=True)

def sigmoid_cost(T,Y): ##@ LOGESTIC regression coST function
    return -(T*np.log(Y)+(T-1)*np.log(1-Y)).sum()

def cost2(T,Y): ##@ USING INDACTOR FUNCTION AS TARGET
    N=len(T)
    return -np.log(Y[np.arrang(N),T]).sum()

def error_rate(target,prediction):
    return np.mean(target!=prediction)

def Y2indicator(Y):
    N=len(Y)
    K=len(set(Y))
    IND=np.zeros((N,K))
    for i in range(N):
        IND[i,Y[i]]=1
    return IND

def getdata(balance_ones=True):
    X=[]
    Y=[]
    First=True
    for line in open('fer2013.csv'):
        if  First:
            First=False
        else:
           row=line.split(',')
           Y.append(int(row[0]))
           X.append([int(p) for p in row[1].split()])
    X,Y=np.array(X)/255.0,np.array(Y) ##@ divided by 255 for normalizig data (now we have data between zero and one)
    ##@ class one is inbalance so repeat it 9 times
    if balance_ones:
       X0,Y0=X[Y!=1,:],Y[Y!=1]
       print('X0 SHAPE IS',X0.shape)
       print('Y0 SHAPE IS', Y0.shape)
       X1=X[Y==1,:]
       print('X1 SHAPE IS BEFORE REPEAT', X1.shape)
       X1=np.repeat(X1,9,axis=0)
       print('X1 SHAPE IS', X1.shape)
       X=np.vstack([X0,X1])
       print('total X SHAPE IS', X.shape)
       Y=np.concatenate((Y0,[1]*len(X1)))
       print('tOTAL Y SHAPE IS', Y.shape)
    return X,Y

def getImageData(): ##@ FOR CONVOLUTIONAL
    X,Y=getdata()
    N,D=X.shape
    d=int(np.sqrt(D))
    X=X.reshape(N,1,d,d)
    return X,Y
def getBinaryData():
    X = []
    Y = []
    First = True
    for line in open('fer2013.csv'):
        if First:
            First = False
        else:
            row = line.split(',')
            y=int(row[0])
            if y==0 or y==1:
               Y.append(y)
               X.append([int(p) for p in row[1].split()] )

    return np.array(X)/255.0,np.array(Y)

if __name__=='__main__':
    getdata()