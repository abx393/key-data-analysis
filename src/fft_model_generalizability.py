"""
Builds and trains ML models on push peaks of one keyboard and tests on the other.
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

dir_in = "../features"
dir_out = "results"
train_keyboard_type = "Dell"
test_keyboard_type = "Dell"
model = "NN"

df = pd.read_csv(os.path.join(dir_in, train_keyboard_type, "touch_fft.csv"))
value_counts = df["key"].value_counts()
print(value_counts)

# Every column except the 0th column is an input feature
x_train = np.array(df.iloc[:, 1:])
labels = np.array(df.iloc[:, 0])

if model == "SVM":
    y_train = labels
else:
    lb = LabelBinarizer()
    y_train = lb.fit_transform(labels)
    classes = lb.classes_
    y_train = np.float64(y_train)
    print("x_train.shape ", x_train.shape)
    print("y_train.shape ", y_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
print(x_train.shape)
x_train = scaler.transform(x_train)

df = pd.read_csv(os.path.join(dir_in, test_keyboard_type, "touch_fft.csv"))
value_counts = df["key"].value_counts()
print(value_counts)

# Every column except the 0th column is an input feature
x_test = np.array(df.iloc[:, 1:])
labels = np.array(df.iloc[:, 0])

# y = np.array([1 if label == "backspace" else 0 for label in labels])
# print(np.count_nonzero(y))
# print(np.shape(y))
if model == "SVM":
    y_test = labels
else:
    lb = LabelBinarizer()
    y_test = lb.fit_transform(labels)
    classes = lb.classes_
    y_test = np.float64(y_test)
    print("x_test.shape ", x_test.shape)
    print("y_test.shape ", y_test.shape)

x_test = scaler.transform(x_test)

if model == "KNN":
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
elif model == "SVM":
    clf = SVC(probability=True)
elif model == "LR":
    clf = LogisticRegression(max_iter=100)
elif model == "NN":
    #clf = MLPClassifier(solver="sgd", alpha=1e-5, max_iter=100000, hidden_layer_sizes=(128))
    clf = MLPClassifier(solver="adam", max_iter=10000, hidden_layer_sizes=(128))
elif model == "KMeans":
    clf = KMeans(n_clusters=3)

clf.fit(x_train, y_train)

#print(classes[np.argmax(y_pred, axis=1)])
#print(lb.inverse_transform(y_pred))

if model == "SVM":
    y_pred = clf.predict(x_test)
    print("y_test", y_test)
    print("y_pred", y_pred)
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
else:
    y_pred = clf.predict_proba(x_test)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    conf_matrix = confusion_matrix(classes[np.argmax(y_test, axis=1)], classes[np.argmax(y_pred, axis=1)], labels=classes)

#precision = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)

#labels = set(y)
print("Results: \n")
print("Accuracy: ", acc)
print("Confusion Matrix: \n", conf_matrix)

# Output confusion matrix to CSV file
"""
f_conf = open(os.path.join(dir_out, keyboard_type, "confusion_matrix.csv"), "w")
f_conf.write("key")
for i in range(conf_matrix.shape[0]):
    f_conf.write("," + classes[i])
f_conf.write("\n")

for i in range(conf_matrix.shape[0]):
    f_conf.write(classes[i])
    for j in range(conf_matrix.shape[1]):
        f_conf.write("," + str(conf_matrix[i][j]))
    f_conf.write("\n")

f_conf.close()
"""

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