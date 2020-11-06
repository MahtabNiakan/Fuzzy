import pandas as pd
import numpy as np

def getdata():
    ##@ Making new data with having classfied target in different column
    df=pd.read_csv('ecommerce_data.csv')
    data=df[['is_mobile','n_products_viewed','visit_duration','is_returning_visitor','time_of_day','user_action']].to_numpy()
    x=data[:,:-1]
    y=data[:,-1]
    N,D=x.shape

    ##@ lets normalize data. using mu and sigma in Normal distribution
    x[:,1]=(x[:,1]-x[:,1].mean())/x[:,1].std()
    x[:, 2] = (x[:, 2] - x[:, 2].mean()) / x[:, 2].std()
    x2=np.zeros((N,D+3)) ##@ create a zero matrix with normal dimentio pluse numer of catergory
    x2[:,0:D-1]=x[:,0:D-1]

    for n in range(N):
        t=int(x[n,D-1]) ##@ getting time from each row
        x2[n,t+D-1]=1 ##@ MAKING indicator matrix . related to each class ,that column would be 1

    z=np.zeros((N,4))
    z[np.arange(N),x[:,D-1].astype(np.int32)]=1

    assert (np.abs(x2[:,-4:]-z).sum()<10e-10)
    return x2,y

def get_binary_data(): ##@ FIND ALL ROWS THAT ARE CLASSES ARE LESS THAN TWO (BINARY)
    X,Y=getdata()
    X2=X[Y<=1]
    Y2=Y[Y<=1]

    return X2,Y2

A,B=get_binary_data()

