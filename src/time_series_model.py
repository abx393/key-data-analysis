import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

PATH_IN = "../features/time_series/key_pairs.csv"
MIN_NUM_SAMPLES = 80

model = "NN"

df = pd.read_csv(PATH_IN)

# filter out training data for keys that we have insufficient data for
df['count'] = df.groupby(['prev_key', 'curr_key']).transform('count')
df_filtered = df[df['count'] >= MIN_NUM_SAMPLES].sort_values(by='count', ascending=False)

# print(df_filtered.to_string())
print("Number of classes: ", df_filtered['count'].nunique())

# print("test ", df[df.prev_key in value_counts])

# First and third columns (previous key and time delay) are input features
prev_keys = np.array(df_filtered.iloc[:,0])
prev_keys = np.reshape([ord(c) for c in prev_keys], (-1, 1)) # convert keys to their ASCII values
time_delays = np.reshape(np.array(df_filtered.iloc[:, 2]), (-1, 1))

x = np.concatenate((prev_keys, time_delays), axis=1)

# Second column (current key) is label
y = np.array(df_filtered.iloc[:, 1])

# One-hot encode labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Normalize input
scaler = StandardScaler()
scaler.fit(x_train)
print(x_train.shape)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

if model == "NN":
    clf = MLPClassifier(solver='adam', max_iter=5000, verbose=True)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

"""
y_pred_proba = clf.predict_proba(x_test)
print(y_pred_proba)

y_pred = np.argmax(y_pred_proba, axis=1)
"""

conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

print("Confusion matrix:")
print(conf_matrix)
