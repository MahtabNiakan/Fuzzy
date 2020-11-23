import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

Nclass=500
K=3
M=3
D=2

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=0.5)
plt.show()
N=len(Y)
T=np.zeros((N,K)) ##@ DEFINE INDICATOR MATRIX
for i in range(N):
    T[i,Y[i]]=1

def init_weight(shape):
    return tf.Variable(tf.random_normal_initializer(shape,stddev=0.01))

def forward(X,W1,b1,W2,b2):
    Z=tf.sigmoid(tf.matmul(X,W1)+b1)
    return tf.sigmoid(tf.matmul(Z,W2)+b2)

tfX=tf.placeholder(tf.float32,(None,D))
tfY=tf.placeholder(tf.float32,(None,K))

W1=init_weight([D,M])
b1=init_weight([M])
W2=init_weight([M,K])
b2=init_weight([K])

logit=forward(tfX,W1,b1,W2,b2)
##@ define cost and loss data in Tenserflow
##@ why? becuse the loss function we are going to call comoute loss function per sample

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY,logits=logit,))
##@ we do not do backprpagation ,Tensorflow does automaticly
train=tf.train.Gradiandecentoptimizer(0.05).minimize(cost)
predict=tf.argmax(logit,1)

sess=tf.Session()
init=tf.random_normal_initializer
sess.run(init)
