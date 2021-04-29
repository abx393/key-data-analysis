"""
Builds and trains ML models on peak data
"""

import os
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

dir_in = "features"
model = "KNN"

df = pd.read_csv(os.path.join(dir_in, "peaks.csv"))

# Every column except the 0th column is an input feature
x = np.array(df.iloc[:, 1:])

# Key label
y = np.array(df["key"])

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

if model == "KNN":
    clf = KNeighborsClassifier(n_neighbors=9, weights='distance')
elif model == "SVM":
    clf = SVC()
elif model == "LR":
    clf = LogisticRegression(max_iter=100)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print("Results: \n")
print("Accuracy: ", acc)
print("Number of labels: ", len(set(y)))
