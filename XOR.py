import numpy as np
import matplotlib.pyplot as plt

def forward(X,W1,b1,W2,b2):
    Z=1/(1+np.exp(-X.dot(W1)-b1)) ##@ sigmoid activation function (1/1+exp(-x))
    activation=Z.dot(W2)+b2
    Y=1/(1+np.exp(-activation))
    return Y,Z

def predict(X,W1,b1,W2,b2):
    Y,_=forward(X,W1,b1,W2,b2)
    return np.round(Y)

def derivative_w1(X, Z, T, Y, W2):
    #dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return X.T.dot(dZ)


def derivative_b1(Z, T, Y, W2):
    #dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return dZ.sum(axis=0)

def derivative_w2(Z, T, Y):##@ derivative of cost function with respect to w2
    return Z.T.dot(T-Y)

def derivative_b2(T,Y):##@ derivative cost function with respect to b2(bios of hidden layer)
    return (T-Y).sum(axis=0)

def cost(T,Y):
    total=0
    for n in range(len(T)):
        if T[n]==1:
            total+=np.log(Y[n])
        else:
            total+=np.log(1-Y[n])
    return total

def get_log_likelihood(T,Y):
    return np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))


def test_xor():
    X=np.array([[0,0],[0,1],[1,0],[1,1]])
    Y=np.array([0,1,1,0])
    learning_rate=0.0005
    last_err_rate=0.00005
    regulization=0.0

    W1=np.random.randn(2,4)
    b1=np.random.randn(4)
    W2=np.random.randn(4)
    b2=np.random.randn(1)

    LL=[] ##@KEEP TRACK OF LIKELIHOOD
    for i in range(30000):
        PY,Z=forward(X,W1,b1,W2,b2)
        ll= get_log_likelihood(Y,PY)
        prediction=predict(X,W1,b1,W2,b2)
        #err=np.abs(prediction-Y).mean()
        err = np.mean(prediction != Y)

        if err!=last_err_rate:
           last_err_rate=err
           print('error rate:',err)
           print('true',Y)
           print('pred',prediction)
        if LL and ll<LL[-1]:##@ if increased we have to go out
           print('early exit')
           break
        LL.append(ll)

        W2+=learning_rate*(derivative_w2(Z,Y,PY)-regulization*W2)
        b2+=learning_rate*(derivative_b2(Y,PY)-regulization*b2)
        W1+=learning_rate*(derivative_w1(X,Z,Y,PY,W2)-regulization*W1)
        b1 += learning_rate * (derivative_b1(Z, Y, PY, W2) - regulization * b1)

        if i%1000==0:
           print(ll)
    print('final classification rate',np.mean(prediction == Y))
    plt.plot(LL)
    plt.show()

def test_dounat():
    N=500
    R_inner=5
    R_outer=10

    R1=np.random.randn(N//2)+R_inner ##@ make 250 radius randomly around inner circle dounat
    Teta=2*np.pi*np.random.randn(N//2) ##@ make 250 random angle for each radius (R1)
    X_inner=np.concatenate([[R1*np.cos(Teta)],[R1*np.sin(Teta)]]).T

    R2=np.random.randn(N//2)+R_outer
    Teta=2*np.pi*np.random.randn(N//2)
    X_outer=np.concatenate([[R2*np.cos(Teta),R2*np.sin(Teta)]]).T

    X=np.concatenate([X_inner,X_outer])
    Y=np.array([0]*(N//2)+[1]*(N//2))

    n_hidden=8
    W1=np.random.randn(2,n_hidden)
    b1=np.random.randn(n_hidden)
    W2=np.random.randn(n_hidden)
    b2=np.random.randn(1)

    LL=[]
    learning_rate=0.00005
    regulization=0.2

    for i in range(3000):
        PY,Z=forward(X,W1,b1,W2,b2)
        ll=get_log_likelihood(Y,PY)
        LL.append(ll)
        prediction=predict(X,W1,b1,W2,b2)
        err=np.abs(prediction-Y).mean() ##@ FIND mean of all TARGET and Prediction WHICH ARE NOT EQUALL


        W2+=learning_rate*(derivative_w2(Z,Y,PY)-regulization*W2)
        b2+=learning_rate*(derivative_b2(Y,PY)-regulization*b2)
        W1+=learning_rate*(derivative_w1(X,Z,Y,PY,W2)-regulization*W1)
        b1+=learning_rate*(derivative_b1(Z,Y,PY,W2)-regulization*b1)

        if i%300==0:
            print("i:", i, "ll:", ll, "classification rate:", 1 - err)

    plt.plot(LL)
    plt.show()



if __name__==('__main__'):
   test_xor()
   #test_dounat()


