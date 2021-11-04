"""
Preprocess time series data for LSTM model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

import constants

LOOK_BACK = 5
PATH_IN = "../features/time_series/key_pairs.csv"

df = pd.read_csv(PATH_IN)
print(df.head(30))

num_features = 2
look_back = 3

x = np.zeros((len(df) - look_back, look_back, num_features))
y = np.zeros((len(df) - look_back))

for i in range(len(df) - look_back):
    for j in range(look_back):
        x[i][j][0] = constants.KEY_MAP.index(df.iloc[i + j, 0].lower()) / len(constants.KEY_MAP)
        x[i][j][1] = df.iloc[i + j, 2]
    y[i] = constants.KEY_MAP.index(df.iloc[i + look_back, 0].lower())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(x_train, y_train, epochs=10, batch_size=72, validation_data=(x_test, y_test), verbose=2, shuffle=False)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

limit = 10

for i in range(limit):
    print("previous keys and time delays: ")
    for j in range(len(x_test[i])):
        prev_key = constants.KEY_MAP[int(round(x_test[i][j][0] * len(constants.KEY_MAP)))]
        time_delay = x_test[i][j][1]

        print("previous key ", prev_key)
        print("time delay ", time_delay)

    next_key_pred = constants.KEY_MAP[int(round(y_pred[i][0]))]
    print("next key PREDICTION: ", next_key_pred)
    next_key_actual = constants.KEY_MAP[int(y_test[i])]
    print("next key ACTUAL: ", next_key_actual)
    print("\n\n")

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
