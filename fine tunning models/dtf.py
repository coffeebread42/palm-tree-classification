from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
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
#break data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
#Develop model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
#apply model
model.score(X_test, y_test)
#predict
y_hat = model.predict(X_test)

###########AUC_evaluation metric############
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC of default model is:")
print(roc_auc)
#####################################################
finding max_depth, find how deep the tree should go, usinf 1-32depth
give range for neighbors to iterate thru
max_depths = np.linspace(1, 32, 32, endpoint=True)
#make list to fill with data from loop
train_results = []
test_results = []
#loop to search right amount off neighbors
for max_depth in max_depths:
   model = DecisionTreeClassifier(max_depth=max_depth)
   model.fit(X_train, y_train)
   train_pred = model.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   #add auc score to previous train results
   train_results.append(roc_auc)
   y_hat = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   #add auc score to previous test results
   test_results.append(roc_auc)

line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.savefig("max_depth_optimal.png")
plt.show()
#########################################
#######finding min_samples_split########
min_samples_splits = np.linspace(0.1, 1.0, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   model = DecisionTreeClassifier(min_samples_split=min_samples_split)
   model.fit(X_train, y_train)

   train_pred = model.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   y_hat = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.savefig("min_samples_split.png")
plt.show()

##################################
min_samples_leaf
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []

for min_samples_leaf in min_samples_leafs:
   model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
   model.fit(X_train, y_train)

   train_pred = model.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   y_hat = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.savefig("min_samples_leaf.png")
plt.show()


#############################################
#max_features
max_features = list(range(1,data.shape[1]))
train_results = []
test_results = []

for max_feature in max_features:
   model = DecisionTreeClassifier(max_features=max_feature)
   model.fit(X_train, y_train)

   train_pred = model.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   y_hat = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)



line1, = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.savefig("max_features.png")
plt.show()
