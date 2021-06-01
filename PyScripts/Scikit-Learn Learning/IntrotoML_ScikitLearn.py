"""
================================
An introduction to machine
learning with scikit-learn
================================

This section of the tutorial introduces machine learning vocabulary and gives
a simple learning example. Some other examples that branch off of this guide
were split into their own scripts by me.
"""


# Loading an example dataset
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)
digits.target
# To access the original sample (different dimensions):
digits.images[0] # Image of shape (8,8)

# To load external datasets:
"""https://scikit-learn.org/stable/datasets/loading_other_datasets.html#external-datasets"""



# Learning and Predicting
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.) # support vector classification
# ^ CLF = classifier. Currently, it's just an instance of an estimator.
# This is a classification problem, by the way.
clf.fit(digits.data[:-1], digits.target[:-1])
# Training set is all images but the last one (save for predicting)
# Predicting = determining the image from the training set that best matches the test image
clf.predict(digits.data[-1:])



# Conventions; Refitting and Updating Parameters
# Hyper-parameters of an estimator can be updated after it has been
# constructed via the set_params() method. Calling fit() more than once
# will overwrite what was learned by any previous fit():
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)
clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
clf.predict(X[:5])
clf.set_params(kernel='rbf').fit(X, y)
clf.predict(X[:5])



# Multiclass vs.Multilabel fitting:
# When using multiclass classifiers, the task that is performed depends
# on the format of the target data
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X, y).predict(X)
# Here, the classifier is fit on a 1d array of multiclass labels, so the model
# provides multiclass predictions.

# Alternatively, if y is 2d:
y = LabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)
#there's one more example but idk what it means