import numpy as np
import matplotlib.pyplot as plt

def forward(X,W1,b1,W2,b2):
    Z=1/(1+np.exp(-X.dot(W1)-b1)) ##@ sigmoid activation function (1/1+exp(-x))
    A=Z.dot(W2)+b2
    expA=np.exp(A)
    Y=expA/expA.sum(axis=1,keepdims=True)
    return Y,Z

##@Compair two matrix of prediction and real classificatin(train)
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

def cost(T,Y): ##@ it is cost function in presentation was called "J" OR "L"(likeihood for dcending and J for accending)
    TEST2=np.log(Y)
    tot = T * TEST2
    TEST=tot.sum()
    return tot.sum()

def derivative_w2(Z, T, Y):##@ derivative of cost function with respect to w2
    N,K=T.shape
    M=Z.shape[1]

    ##@ Find derivative without using vector product

    ##@ slow
    # ret1=np.zeros((M,K)) ##@ making empty matrix for insert derivative value
    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             ret1[m,k]=ret1[m,k]+(T[n,k]-Y[n,k])*Z[n,m]
    ### @######
    # ret2=np.zeros((M,K))
    # for n in range(N):
    #     for k in range(K):
    #         ret2[:, k] = ret2[:, k] + (T[n, k] - Y[n, k]) * Z[n, :]
    ### @######
    # ret3=np.zeros((M,K))
    # for n in range(N):
    #     ret3=ret3+np.outer(Z[n],T[n]-Y[n])

    ret4 = np.zeros((M, K))
    ret4=Z.T.dot(T-Y)
    return ret4

def derivative_b2(T,Y):##@ derivative cost function with respect to b2(bios of hidden layer)
    return (T-Y).sum(axis=0)


def derivative_w1(X,Z,T,Y,W2):
    N,D=X.shape
    M,K=W2.shape
    # ret1=np.zeros((D,M))
    ##@ slow
    # for n in range(N):
    #     for k in range(K):
    #         for m in range(M):
    #             for d in range(D):
    #                 ret1[d,m]=ret1[d,m]+(T[n,k]-Y[n,k])*W2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]

    ret2=np.zeros((D,M))
    dZ=((T-Y).dot(W2.T))*Z*(1-Z)
    ret2=X.T.dot(dZ)

    return ret2

def derivative_b1(T,Y,W2,Z):
    return ((T-Y).dot(W2.T)*Z*(1-Z)).sum(axis=0)


def main():
    #creat data
    Nclass = 1500
    D = 2  ##@ dimention of inputs
    M = 3  ##@ neurons numbers
    K = 3  ##@ number of classification

    ##@ creat data in normal distribution with different <mu> and <standard deviation>=sigma=1
    ##@ if we wanted change sigama , it have to multipled with root of standard deviation

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    ##@vstack() function is used to stack the sequence of input arrays vertically to make a single array
    X = np.vstack([X1, X2, X3])  ##@ number of data=Nclass *3 and number of features=2
    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

    N = len(Y)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    # plt.scatter(X[:, 0], X[:, 1], c=Y, s=30,alpha=0.5)  ##@ c is class for color,classifiying t class of data using change color class
    # plt.show()

    #randomly initiated weight and bios
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate=10e-7
    costs=[]
    for epoch in range(100000):
        output,hidden=forward(X,W1,b1,W2,b2)
        if epoch%100==0:
           c=cost(T,output)
           P=np.argmax(output,axis=1) ##@ find prediction
           r=classification_rate(Y, P)
           print("cost:",c," and classification rate is :",r)
           costs.append(c)
        ##@ IT IS TIME TO TUNE WEIGHTS AND BIOS
        W2=W2+learning_rate*derivative_w2(hidden,T,output)
        b2=b2+learning_rate*derivative_b2(T,output)
        W1=W1+learning_rate*derivative_w1(X,hidden,T,output,W2)
        b1=b1+learning_rate*derivative_b1(T,output,W2,hidden)
    plt.plot(costs)
    plt.show()




if __name__=='__main__':
    main()