"""
Preprocess time series data for LSTM models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

import constants

model_type = "SVM"
granularity = "finger"

if granularity == "finger":
    PATH_IN = "../features/time_series/finger_pairs.csv"
elif granularity == "key":
    PATH_IN = "../features/time_series/key_pairs_cleaned.csv"

df = pd.read_csv(PATH_IN)

num_features = 2
look_back = 1

x = np.zeros((len(df) - look_back, look_back, num_features))
y = np.zeros((len(df) - look_back))

for i in range(len(df) - look_back):
    for j in range(look_back):
        if granularity == "finger":
            try:
                x[i][j][0] = constants.FINGERS.index(df.iloc[i + j, 0]) / len(constants.FINGERS)
            except ValueError:
                print(df.iloc[i + j, 0])
        elif granularity == "key":
            x[i][j][0] = constants.KEY_MAP.index(df.iloc[i + j, 0].lower()) / len(constants.KEY_MAP)

        x[i][j][1] = df.iloc[i + j, 2]
    if granularity == "finger":
        try:
            y[i] = constants.FINGERS.index(df.iloc[i + look_back, 0])
        except ValueError:
            print (df.iloc[i+look_back, 0])
    elif granularity == "key":
        y[i] = constants.KEY_MAP.index(df.iloc[i + look_back, 0].lower())

if model_type == "SVM":
    print(x.shape)
    x = x.reshape((len(df) - look_back, -1))
    print(x.shape)

train_test_cutoff = int(0.8 * len(x))
x_train = x[: train_test_cutoff]
y_train = y[: train_test_cutoff]
x_test = x[train_test_cutoff: ]
y_test = y[train_test_cutoff: ]

if model_type == "RNN":
    model = Sequential()
    model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    history = model.fit(x_train, y_train, epochs=10, batch_size=36, validation_data=(x_test, y_test), verbose=2,
                        shuffle=False)
elif model_type == "SVM":
    model = SVC(C=1.0, probability=True)
    model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

limit = 10

for i in range(limit):
    print("previous keys and time delays: ")
    if granularity == "finger":
        feat_map = constants.FINGER_MAP
    elif granularity == "key":
        feat_map = constants.KEY_MAP

    if model_type == "SVM":
        for j in range(0, len(x_test[i]), 2):
            prev_key = feat_map[int(round(x_test[i][j] * len(feat_map)))]
            time_delay = x_test[i][j+1]

            print("previous key ", prev_key)
            print("time delay ", time_delay)

        next_key_pred = feat_map[int(round(y_pred[i]))]
        print("next {} PREDICTION: ".format(granularity), next_key_pred)
        next_key_actual = feat_map[int(y_test[i])]
        print("next {} ACTUAL: ".format(granularity), next_key_actual)
        print("\n\n")

    elif model_type == "RNN":
        for j in range(len(x_test[i])):
            prev_key = feat_map[int(round(x_test[i][j][0] * len(feat_map)))]
            time_delay = x_test[i][j][1]

            print("previous key ", prev_key)
            print("time delay ", time_delay)

        next_key_pred = feat_map[int(round(y_pred[i][0]))]
        print("next {} PREDICTION: ".format(granularity), next_key_pred)
        next_key_actual = feat_map[int(y_test[i])]
        print("next {} ACTUAL: ".format(granularity), next_key_actual)
        print("\n\n")

if model == "RNN":
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
