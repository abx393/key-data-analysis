import os
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DIR_IN = '../features/spectrogram'
DIR_OUT = '../results/'
KEYBOARD_TYPE = 'Dell'
model_type = 'NN'

x = []
y = []
classes = ['not space', 'space']

input_shape = (128, 7)
for f in os.listdir(os.path.join(DIR_IN, KEYBOARD_TYPE)):
    (basename, extension) = f.split('.')
    #print(basename)
    if basename.startswith('mouse_click') or basename.startswith('mouse_scroll'):
        continue

    (label, cnt) = basename.split('_')

    arr = np.load(os.path.join(DIR_IN, KEYBOARD_TYPE, f))
    #print('arr.shape', arr.shape)
    if input_shape != arr.shape:
        continue

    input_shape = arr.shape

    if label == 'space':
        x.append(arr.tolist())
        y.append(1)
    elif random.random() < 0.5:
        x.append(arr.tolist())
        y.append(0)

scaler = StandardScaler()

x = np.array(x)
y = np.array(y)
print('space count ', np.count_nonzero(y))
print('non-space count ', y.shape[0] - np.count_nonzero(y))

#x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
x = np.reshape(x, (x.shape[0], -1))
x = np.absolute(x)
print('x.shape ', x.shape)
print('y.shape ', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape)

max_x_train = np.amax(x_train)
x_train /= max_x_train
x_test /= max_x_train

#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

print('input_shape ', input_shape)

"""
model = Sequential()
model.add(Conv2D(input_shape=(input_shape[0], input_shape[1], 1), filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer=Adam(lr=0.0001), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test))
"""

if model_type == 'KNN':
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
elif model_type == 'SVM':
    clf = SVC(probability=True)
elif model_type == 'LR':
    clf = LogisticRegression(max_iter=100)
elif model_type == 'NN':
    #clf = MLPClassifier(solver="sgd", alpha=1e-5, max_iter=100000, hidden_layer_sizes=(128))
    clf = MLPClassifier(solver='adam', max_iter=10000, hidden_layer_sizes=(256))
elif model_type == 'KMeans':
    clf = KMeans(n_clusters=3)

clf.fit(x_train, y_train)
if model_type == 'SVM':
    y_pred = clf.predict(x_test)
    print('y_test', y_test)
    print('y_pred', y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
else:
    #y_pred = clf.predict_proba(x_test)
    #acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    #conf_matrix = confusion_matrix(classes[np.argmax(y_test, axis=1)], classes[np.argmax(y_pred, axis=1)], labels=classes)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

print('Results: \n')
print('Accuracy: ', acc)
print('F1 score', f1)
print('Confusion Matrix: \n', conf_matrix)

# Output confusion matrix to CSV file
f_conf = open(os.path.join(DIR_OUT, KEYBOARD_TYPE, 'confusion_matrix_space_detection.csv'), 'w')
f_conf.write('key')
for i in range(conf_matrix.shape[0]):
    f_conf.write(',' + classes[i])
f_conf.write('\n')

for i in range(conf_matrix.shape[0]):
    f_conf.write(classes[i])
    for j in range(conf_matrix.shape[1]):
        f_conf.write(',' + str(conf_matrix[i][j]))
    f_conf.write("\n")

f_conf.close()
