"""
Aggregate the time distances between consecutive key presses for each possible pair of keys.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import constants
from statistics import median, mean

def handle(func, arr):
    try:
        return func(arr)
    except:
        return -1

OUTLIER_CUTOFF = 0.40

DIR_IN = "../native_raw_data_time_series"
KEYBOARD = "HP_Spectre"

DIR_OUT = "../features/time_series"

# either 'mean' or 'median'
CENTER = "median"

LIMITED_KEYS = 27

key_nn_features = open(os.path.join(DIR_OUT, "key_pairs.csv"), 'w')
key_nn_features.write("prev_key,curr_key,time_delay\n")

finger_nn_features = open(os.path.join(DIR_OUT, "finger_pairs.csv"), 'w')
finger_nn_features.write("prev_finger,curr_finger,time_delay\n")

time_delay_by_key_pair = [[[] for i in range(len(constants.KEY_MAP))] for j in range(len(constants.KEY_MAP))]
time_delay_by_finger_pair = [[[] for i in range(len(constants.FINGERS))] for j in range(len(constants.FINGERS))]

count = 0
for f in os.listdir(os.path.join(DIR_IN, KEYBOARD)):
    (basename, extension) = f.split(".")
    print("basename ", basename)
    if extension != "csv":
        continue

    with open(os.path.join(DIR_IN, KEYBOARD, f), 'r') as fr:
        prev_key = ""
        prev_time = 0
        i = 0
        for x in fr:
            if i > 0:
                try:
                    [curr_key, curr_time] = x.split(',')
                except:
                    print("i=", i)
                    print("x=", x)

                curr_time = float(curr_time)
                time_delay = curr_time - prev_time

                try:
                    prev_key_index = constants.KEY_MAP.index(prev_key)
                    curr_key_index = constants.KEY_MAP.index(curr_key)

                    prev_finger = constants.FINGER_MAP[prev_key_index]
                    curr_finger = constants.FINGER_MAP[curr_key_index]

                    prev_finger_index = constants.FINGERS.index(prev_finger)
                    curr_finger_index = constants.FINGERS.index(curr_finger)
                except:
                    print("Key {} or {} not in key map".format(prev_key, curr_key))

                if prev_key != "space" and time_delay < OUTLIER_CUTOFF:

                    time_delay_by_key_pair[prev_key_index][curr_key_index].append(time_delay)
                    key_nn_features.write("{},{},{}\n".format(prev_key, curr_key, time_delay))

                    time_delay_by_finger_pair[prev_finger_index][curr_finger_index].append(time_delay)
                    finger_nn_features.write("{},{},{}\n".format(prev_finger, curr_finger, time_delay))

                prev_time = curr_time
                prev_key = curr_key

            i += 1
    count+= 1

key_nn_features.close()
finger_nn_features.close()

# Plot average time delay for all key pairs
fig, ax = plt.subplots()
if CENTER == "mean":
    im = ax.imshow([[handle(mean, time_delay_by_key_pair[i][j]) for j in range(LIMITED_KEYS)] for i in range(LIMITED_KEYS)])
elif CENTER == "median":
    im = ax.imshow([[handle(median, time_delay_by_key_pair[i][j]) for j in range(LIMITED_KEYS)] for i in range(LIMITED_KEYS)])

for i in range(LIMITED_KEYS):
    for j in range(LIMITED_KEYS):
        if len(time_delay_by_key_pair[i][j]) > 0:
            if CENTER == "mean":
                text = ax.text(j, i, round(handle(mean, time_delay_by_key_pair[i][j]), 2), ha="center", va="center", color="w")
            elif CENTER == "median":
                text = ax.text(j, i, round(handle(median, time_delay_by_key_pair[i][j]), 2), ha="center", va="center", color="w")

        text.set_color('black')
        text.set_size(8)

ax.set_title("Average Time Delay (s)")
labels = [(constants.FINGER_MAP[i] + "    " + constants.KEY_MAP[i]) for i in range(LIMITED_KEYS)]
ax.set_xticks(np.arange(LIMITED_KEYS))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_yticks(np.arange(LIMITED_KEYS))
ax.set_yticklabels(labels)

#fig.tight_layout()
plt.show()
plt.close()

# Plot number of data points for all key pairs
fig, ax = plt.subplots()
im = ax.imshow([[len(time_delay_by_key_pair[i][j]) for j in range(LIMITED_KEYS)] for i in range(LIMITED_KEYS)])
for i in range(LIMITED_KEYS):
    for j in range(LIMITED_KEYS):
        text = ax.text(j, i, len(time_delay_by_key_pair[i][j]), ha="center", va="center", color="w")
        text.set_color('black')
        text.set_size(8)

labels = [(constants.FINGER_MAP[i] + "    " + constants.KEY_MAP[i]) for i in range(LIMITED_KEYS)]
ax.set_title("Number of Data Points for Each Key Pair")
ax.set_xticks(np.arange(LIMITED_KEYS))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_yticks(np.arange(LIMITED_KEYS))
ax.set_yticklabels(labels)

fig.tight_layout()
plt.show()

# Plot average time delay for all finger pairs
fig, ax = plt.subplots()
if CENTER == "mean":
    im = ax.imshow([[handle(mean, arr) for arr in arr2] for arr2 in time_delay_by_finger_pair])
elif CENTER == "median":
    im = ax.imshow([[handle(median, arr) for arr in arr2] for arr2 in time_delay_by_finger_pair])

for i in range(len(time_delay_by_finger_pair)):
    for j in range(len(time_delay_by_finger_pair[i])):
        if CENTER == "mean":
            text = ax.text(j, i, round(handle(mean, time_delay_by_finger_pair[i][j]), 2), ha="center", va="center", color="w")
        elif CENTER == "median":
            text = ax.text(j, i, round(handle(median, time_delay_by_finger_pair[i][j]), 2), ha="center", va="center", color="w")

        text.set_color('black')
        text.set_size(8)

ax.set_title("Average Time Delay By Finger Pair (s)")
labels = [constants.FINGERS[i] for i in range(len(constants.FINGERS))]
ax.set_xticks(np.arange(len(constants.FINGERS)))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_yticks(np.arange(len(constants.FINGERS)))
ax.set_yticklabels(labels)

fig.tight_layout()
plt.show()

# Plot number of data points for all finger pairs
fig, ax = plt.subplots()
im = ax.imshow([[len(arr) for arr in arr2] for arr2 in time_delay_by_finger_pair], cmap='hot', interpolation='nearest')
for i in range(len(time_delay_by_finger_pair)):
    for j in range(len(time_delay_by_finger_pair[i])):
        text = ax.text(j, i, len(time_delay_by_finger_pair[i][j]), ha="center", va="center", color="w")
        text.set_color('black')
        text.set_size(8)

labels = [constants.FINGERS[i] for i in range(len(constants.FINGERS))]
ax.set_title("Number of Data Points for All Finger Pairs")
ax.set_xticks(np.arange(len(constants.FINGERS)))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_yticks(np.arange(len(constants.FINGERS)))
ax.set_yticklabels(labels)

fig.tight_layout()
plt.show()
