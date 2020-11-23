import numpy as np
from sklearn.utils import shuffle
from util import sigmoid,sigmoid_cost,getBinaryData,error_rate,relu
import matplotlib.pyplot as plt


class ANN(object):
    def __init__(self,M):
       self.M=M


    def fit(self,X,Y,learningrate=5*10e-7,reg=1,epoch=1000,show_figure=False):
        X,Y=shuffle(X,Y)
        Xvalid,Yvalid=X[-1000:],Y[-1000:]
        X,Y=X[:1000],Y[:1000]

        N,D=X.shape
        self.W1=np.random.randn(D,self.M)/np.sqrt(D+self.M)
        self.b1=np.random.randn(self.M)
        self.W2=np.random.randn(self.M)/np.sqrt(self.M)
        self.b2=0
        costs=[]
        best_validation_error=1
        for i in range(epoch):
            pY,Z=self.forward(X)
            PY_y=pY-Y
            self.W2-=learningrate*(Z.T.dot( PY_y)+reg*self.W2)
            self.b2-=learningrate*((PY_y).sum()+reg*self.b2)

            #dz=np.outer(PY_y,self.W2)*(Z>0) ##@ RELUE AS ACTIVATION FUNCTION
            dz=np.outer(PY_y,self.W2)*(1-Z*Z) ##@ USING tanh instead of relue as activation function
            self.W1-=learningrate*((X.T.dot(dz))+reg*self.W1)
            self.b1-=learningrate*(np.sum(dz,axis=0)+reg*self.b1)
            if i%20==0 :
               # print('W1: ',self.W1[0],' AND W2 :',self.W2[0])
                pY_valid,_=self.forward(Xvalid)
                C=sigmoid_cost(Yvalid,pY_valid)
                costs.append(C)
                err=error_rate(Yvalid,np.round(pY_valid))
                print('i:',i,' cost:',C,' error:',err)
                if err<best_validation_error:
                   best_validation_error=err
        print('best validation error:',best_validation_error)
        if show_figure:
           plt.plot(costs)
           plt.show()
    def forward(self,X):
        #Z=relu(X.dot(self.W1)+self.b1) ##@ ACTIVATION FUNCTION IS RELUE
        Z=np.tanh(X.dot(self.W1)+self.b1) ##@ using tangh as activation function
        return sigmoid(Z.dot(self.W2)+self.b2),Z ##@ logestic so finalazed  by sigmoid
    def predict(self,X):
        Py,_=self.forward(X)
        return np.round(Py)
    def score(self,X,Y):
        prediction=self.predict(X)
        return 1-error_rate(Y,prediction)

def main():
    X,Y=getBinaryData()
    X0=X[Y==0,:] ##@ FIND ALL SAMPLE WHICH THEIR VALUE IS EQUAL TO 0
    X1=X[Y==1,:] ##@ BALANCE DATA WITH MORE Y==1
    X1=np.repeat(X1,9,axis=0)
    X=np.concatenate([X0,X1])
    Y=np.array([0]*len(X0)+[1]*len(X1))

    model=ANN(100)
    model.fit(X,Y,show_figure=True)

if __name__=='__main__':
    main()