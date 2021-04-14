import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# kernel is 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train, y_train)
score_rbf = clf_rbf.score(X_test, y_test)
print("the rbf kernel's score on iris is %f" % score_rbf)

# kernel is 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train, y_train)
score_linear = clf_rbf.score(X_test, y_test)
print("the linear kernel's score on iris is %f" % score_linear)

# kernel is 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train, y_train)
score_poly = clf_poly.score(X_test, y_test)
print("the poly kernel's score on iris is %f" % score_poly)