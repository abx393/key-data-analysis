"""
Builds and trains ML models on peak data
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

dir_in = "features"
keyboard_type = "mechanical"
model = "KMeans"

df = pd.read_csv(os.path.join(dir_in, keyboard_type, "vggish_embeddings.csv"))
print(df.head())

# Every column except the 0th column is an input feature
x = np.array(df.iloc[:, 1:])

# Key label
y = np.array(df["key"])

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
acc = accuracy_score(y_test, y_pred)

print("Results: \n")
print("Accuracy: ", acc)
print("Number of labels: ", len(set(y)))

df.set_index("key", inplace=True)

labels = ["a", "space", "backspace"]
for key in labels:
    print("Mean: key={}".format(key))
    print(df.loc[key, :].mean())
    print("Variance: key={}".format(key))
    print(df.loc[key, :].var())
    print()
