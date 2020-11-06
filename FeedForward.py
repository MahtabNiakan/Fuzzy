import numpy as np
import matplotlib.pyplot as plt
Nclass=500

##@ creat data in normal distribution with different <mu> and <standard deviation>=sigma=1
##@ if we wanted change sigama , it have to multipled with root of standard deviation
x1=np.random.randn(Nclass,2) +np.array([0,-2])
x2=np.random.randn(Nclass,2) +np.array([2,2])
x3=np.random.randn(Nclass,2) +np.array([-2,2])
##@vstack() function is used to stack the sequence of input arrays vertically to make a single array
X=np.vstack([x1,x2,x3]) ##@ number of data=Nclass *3 and number of features=2


y=np.array([0]*Nclass+[1]*Nclass+[2]*Nclass)
plt.scatter(X[:,0],X[:,1],c=y,s=30,alpha=0.5) ##@ c is class for color,classifiying t class of data using change color class
plt.show()

D=2 ##@ dimention of inputs
M=3 ##@ neurons numbers
K=3 ##@ number of classification

w1=np.random.randn(D,M)
b1=np.random.randn(M)
w2=np.random.randn(M,K)
b2=np.random.randn(K)

def forward(X,W1,b1,W2,b2):
    Z=1/1+np.exp(-X.dot(W1)-b1) ##@ sigmoid activation function (1/1+exp(-x))
    A=Z.dot(W2)+b2
    expA=np.exp(A)
    Y=expA/expA.sum(axis=1,keepdims=True)
    return Y

def classification_rate(P,Y):##@Compair two matrix of prediction and real classificatin(train)
    n_correct=0
    n_total=0
    for i in range(len(Y)):
        n_total=n_total+1
        if Y[i]==P[i]:
            n_correct=n_correct+1
    return float(n_correct/n_total)


p_y_given_x=forward(X,w1,b1,w2,b2)
predictin_result=np.argmax(p_y_given_x,axis=1)
print('result of prediction:',classification_rate(predictin_result,y))

assert (len(y)==len(predictin_result))
