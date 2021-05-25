"""
Builds and trains ML models on peak data
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

dir_in = "features"
keyboard_type = "mechanical"
model = "NN"

df = pd.read_csv(os.path.join(dir_in, keyboard_type, "touch_fft.csv"))
print(df.head())

# Every column except the 0th column is an input feature
x = np.array(df.iloc[:, 1:])
labels = np.array(df.iloc[:, 0])

# y = np.array([1 if label == "backspace" else 0 for label in labels])
# print(np.count_nonzero(y))
# print(np.shape(y))
lb = LabelBinarizer()
y = lb.fit_transform(labels)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

if model == "KNN":
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
elif model == "SVM":
    clf = SVC()
elif model == "LR":
    clf = LogisticRegression(max_iter=100)
elif model == "NN":
    clf = MLPClassifier(solver="sgd", alpha=1e-5, max_iter=10000)
elif model == "KMeans":
    clf = KMeans(n_clusters=3)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("y_pred ", y_pred)
print("y_test ", y_test)
acc = accuracy_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)

#labels = set(y)
print("Results: \n")
print("Accuracy: ", acc)
#print("Precision: ", precision)
#print("Recall: ", recall)
#print("F1 score: ", f1)
#print("Number of labels: ", len(labels))

df.set_index("key", inplace=True)

"""
for key in labels:
    print("Mean: key={}".format(key))
    print(df.loc[key, :].mean())
    print("Variance: key={}".format(key))
    print(df.loc[key, :].var())
    print()
"""
