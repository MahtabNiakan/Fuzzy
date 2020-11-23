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

M=5
D=X.shape[1]
K=len(set(Y))

Xtrain=X[:-100]
Ytrain=Y[:-100]
Ytrain_ind=y2indicator(Ytrain,K)

Xtest=X[-100:]
Ytest=Y[-100:]
Ytest_ind=y2indicator(Ytest,K)

W1=np.random.randn(D,M)
b1=np.zeros(M)
W2=np.random.randn(M,K)
b2=np.zeros(K)

def softmax(a):
    expA=np.exp(a)
    return expA/expA.sum(axis=1,keepdims=True)

def forward(X,W1,b1,W2,b2):
     Z=np.tanh(X.dot(W1)+b1) # Here activation function is tangant hyperbolic
     return softmax(Z.dot(W2)+b2),Z

def predict(p_y_given_x):
    return np.argmax(p_y_given_x)

def classification_rate(Y,P):
    return np.mean(Y==P)
def cross_entropy(T,predict_Y):
    return -np.mean(T*np.log(predict_Y))

train_costs=[]
test_costs=[]
learning_rate=0.001
for i in range(10000):
    p_ytrain,z_train=forward(Xtrain,W1,b1,W2,b2)
    p_ytest,z_test=forward(Xtest,W1,b1,W2,b2)

    Ctrain=cross_entropy(Ytrain_ind,p_ytrain)
    Ctest=cross_entropy(Ytest_ind,p_ytest)

    train_costs.append(Ctrain)
    test_costs.append(Ctest)

    W2=W2-learning_rate*(z_train.T.dot(p_ytrain-Ytrain_ind))
    b2=b2-learning_rate*(p_ytrain-Ytrain_ind).sum()
    dz=(p_ytrain-Ytrain_ind).dot(W2.T)*(1-z_train*z_train)

    W1=W1-learning_rate* Xtrain.T.dot(dz)
    b1=b1-learning_rate*(dz).sum(axis=0)

    if i%100==0:
        print(i,Ctrain,Ctest)
print('final training model classification rate is',classification_rate(Ytrain,predict(p_ytrain)))
print('final testing model classification rate is',classification_rate(Ytest,predict(p_ytest)))

legend1,=plt.plot(train_costs, label='train_c0st')
legend2,=plt.plot(test_costs, label='test_cost')
plt.legend()
plt.show()


