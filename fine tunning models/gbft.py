from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

inf ="/home/roberto/Desktop/work/src/classifiers/flv2.out"
### Input
filename = inf
data = np.genfromtxt(filename, delimiter=' ')
y = data[:, 0]
X = data[:, 1:]
o_a=[]
s_s=[]
s_p=[]

#break data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
Develop model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
#apply model
model.score(X_test, y_test)
#predict
y_hat = model.predict(X_test)

######################AUC_evaluation metric
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC of default model is:")
print(roc_auc)
######################find Learning rate#############################
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
train_results = []
test_results = []
for eta in learning_rates:
    model = GradientBoostingClassifier(learning_rate=eta)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_hat = model.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(learning_rates, train_results, 'b', label="Train AUC")
line2, = plt.plot(learning_rates, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('learning rate')
plt.savefig("optimal_learning_rate_gb.png")
plt.show()
#####################################################
#N_estimators, finding number of trees in the forest, searching for sweet spot
give range for N_estimators
n_estimators = [1,2,4,8,16,32,64,100,200]
#make list to fill with data from loop
train_results = []
test_results = []
#loop to search right amount off neighbors
for estimator in n_estimators:
   model = GradientBoostingClassifier(n_estimators=estimator)
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

line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.savefig("optimal_n_estimators_gb.png")
plt.show()
#########################################
max_depth
max_depths = np.linspace(1,32,32, endpoint=True)
#make list to fill with data from loop
train_results = []
test_results = []
#loop to search right amount off neighbors
for max_depth in max_depths:
   model = GradientBoostingClassifier(max_depth=max_depth)
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
plt.savefig("optimal_tree_depth_gb.png")
plt.show()

#finding min_samples_split
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   model = GradientBoostingClassifier(min_samples_split=min_samples_split)
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
plt.savefig("min_samples_split_gb.png")
plt.show()

##################################
min_samples_leaf
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []

for min_samples_leaf in min_samples_leafs:
   model = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)
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
plt.savefig("min_samples_leaf_gb.png")
plt.show()


############################################
#max_features
max_features = list(range(1,data.shape[1]))
train_results = []
test_results = []

for max_feature in max_features:
   model = GradientBoostingClassifier(max_features=max_feature)
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
plt.savefig("max_features_rf.png")
plt.show()
