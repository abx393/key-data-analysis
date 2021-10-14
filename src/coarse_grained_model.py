"""
Builds and trains ML model to classify mouse press, mouse scroll, and key press.
"""

import os
import pandas as pd
import numpy as np
import random

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

DIR_IN = "../features"
KEYBOARD_TYPE = "Dell"
DATA_FILE = "vggish_embeddings.csv"

model = "NN"

df = pd.read_csv(os.path.join(DIR_IN, KEYBOARD_TYPE, DATA_FILE))
print(df.head())
print(df["key"].value_counts())

# Every column except the 0th column is an input feature
x = np.array(df.iloc[:, 1:])
labels = np.array(df.iloc[:, 0])

# Downsample the "key" class
num_keys = 200
x_keys = []
keys = []
for i in range(len(labels)):
    if not labels[i].startswith("mouse"):
        x_keys.append(x[i])
        keys.append("key")

seed = 6
random.seed(seed)
keys_subset = random.choices(keys, k=num_keys)
random.seed(seed)
x_keys_subset = random.choices(x_keys, k=num_keys)

x_mouse = []
mouse = []
for i in range(len(labels)):
    if labels[i].startswith("mouse"):
        x_mouse.append(x[i])
        mouse.append(labels[i])

x_mouse = np.array(x_mouse)
x_keys_subset = np.array(x_keys_subset)
print(x_mouse)
print(x_keys_subset)

# Preprocessed data
x_pre = np.concatenate((x_mouse, x_keys_subset))
y_pre = np.concatenate((mouse, keys_subset))

print(pd.Series(y_pre).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x_pre, y_pre)

if model == "KNN":
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
elif model == "LR":
    solvers = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
    for solver in solvers:
        for fit_intercept in [True, False]:
            clf = LogisticRegression(max_iter=10000, solver=solver, fit_intercept=fit_intercept)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            print("Model: ", model)
            print("solver: ", solver)
            print("fit-intercept: ", fit_intercept)
            # print("k: ", k)
            print("Accuracy: ")
            print(accuracy_score(y_test, y_pred))
            print("Confusion matrix: ")
            print(confusion_matrix(y_test, y_pred))
elif model == "NN":
    lr = [10.0**pow for pow in np.arange(-4, -3, 0.1)]
    print("lr", lr)
    for lr_init in lr:
        clf = MLPClassifier(solver="adam", max_iter=10000, hidden_layer_sizes=(64), learning_rate_init=lr_init, early_stopping=False, verbose=False)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        print("Model: ", model)
        print("lr_init: ", lr_init)
        # print("k: ", k)
        print("Accuracy: ")
        print(accuracy_score(y_test, y_pred))
        print("Confusion matrix: ")
        print(confusion_matrix(y_test, y_pred))
        print()
elif model == "SVM":
    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

print("y_pred", y_pred)

print("Model: ", model)
# print("k: ", k)
print("Accuracy: ")
print(accuracy_score(y_test, y_pred))
print("Confusion matrix: ")
print(confusion_matrix(y_test, y_pred))
