
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from process import getdata
import sys

X,Y=getdata()
X,Y=shuffle(X,Y)
Ntrain=int(0.7*len(X)) ##@ 70% as much as data we have

X_train,Y_train=X[:Ntrain],Y[:Ntrain]
X_test,Y_test=X[Ntrain:],Y[Ntrain:]

model=MLPClassifier(hidden_layer_sizes=(20,20),max_iter=2000)
model.fit(X_train,Y_train)
train_accu=model.score(X_train,Y_train)
test_accu=model.score(X_test,Y_test)
print('train accuracy:',train_accu)
print('test accuracy:',test_accu)