import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

inf ="/home/roberto/Desktop/work/src/classifiers/flv2.out"
### Input
filename = inf
### Process
data = np.genfromtxt(filename, delimiter=' ')

y = data[:, 0]
X = data[:, 1:]

kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    svc = svm.SVC(kernel=kernel).fit(X,y)
    plotSVC('kernel=' + str(kernel))

def plotSVC(title):
    #create a mesh to plot in`
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    h = (x_max / x_min)/100
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 1:], c=y, cmap=plt.cm.Paired)
    plt.xlabel('vector')
    plt.ylabel('label')
    plt.xlim(xx.min(), xx.max())
    plt.title(title)
    plt.show()
