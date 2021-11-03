"""
Preprocess time series data for LSTM model.
"""

import numpy as np
import pandas as pd
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
        x[i][j][0] = constants.KEY_MAP.index(df.iloc[i + j, 0].lower())
        x[i][j][1] = df.iloc[i + j, 2]
    y[i] = constants.KEY_MAP.index(df.iloc[i + look_back, 0].lower())
