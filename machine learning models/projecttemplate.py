# Load libraries
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
#data directory
inf ="/home/roberto/Desktop/work/src/classifiers/flv2.out"
### Input
filename = inf
### read data as np array
data = np.genfromtxt(filename, delimiter=' ')
#y is label column x is the variable columns
y = data[:, 0]
X = data[:, 1:]
#split the datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression(C= 0.010)))
models.append(('RFC', RandomForestClassifier(max_depth= 7,
                                             n_estimators= 500,
                                             min_samples_leaf= 0.2,
                                             min_samples_split= 0.58)))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3, p=3)))
models.append(('DTC', DecisionTreeClassifier(max_depth=5, min_samples_split=0.3, min_samples_leaf=0.3)))
models.append(('GBC', GradientBoostingClassifier(learning_rate= 0.1,
                                                    n_estimators=40,
                                                    max_depth=3,
                                                    min_samples_split=0.6,
                                                    min_samples_leaf=0.35)))
models.append(('SVM', SVC(C=10000, kernel='linear')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state= None)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle( 'Classifier Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results, showfliers=False)
ax.set_xticklabels(names)
pyplot.savefig("classifier_comp3.png")
pyplot.show()
