"""
Plot the time distance between consecutive key presses for various words.
"""

import os
import numpy as np

from json import JSONDecoder
import matplotlib.pyplot as plt

# words = ["gamma", "pedal", "brake", "climb", "walks", "hello", "world", "smile", "zebra", "space", "rover", "gears", "wheel", "stain"]
words = ["the", "why", "ask", "mail", "data", "soon", "wall", "hold", "four", "gold", "thyme", "under",
           "gamma", "pedal", "brake", "climb", "walks", "hello", "world", "smile", "zebra", "space", "rover", "gears", "wheel", "stain",
           "static", "corral", "source", "groggy", "eleven", "quotes", "slight", "reward", "joseph"]
y = {}
for i, w in enumerate(words):
    y[w] = []

DIR_IN = "native_raw_data_time_series_words"
KEYBOARD = "HP_Spectre"
#DIR_OUT = "plots"
#SUBDIR_OUT = "time_series"

for f in os.listdir(os.path.join(DIR_IN, KEYBOARD)):
    (basename, extension) = f.split(".")

    # ignore audio data for now, only look at ground truth
    if extension == "wav":
        continue

    if extension == "json":
        labels_file = open(os.path.join(DIR_IN, KEYBOARD, f))
        labels = JSONDecoder().decode(labels_file.read())

        prev_time = 0
        ts = np.asarray(list(labels.keys()), dtype=np.int)
        keys = np.array(list(labels.values()))
    elif extension == "csv":
        fr = open(os.path.join(DIR_IN, KEYBOARD, f))
        i = 0
        keys = []
        ts = []
        for line in fr:
            if i > 0:
                try:
                    [curr_key, curr_time] = line.split(',')
                    keys.append(curr_key)
                    curr_time = float(curr_time)
                    ts.append(1000 * curr_time)
                except:
                    print("i=", i)
            i += 1

    word = "".join(keys)
    if word in y:
        # Compute the 1st difference of the list of timestamps
        time_delays = np.diff(ts, n=1).tolist()

        # right-pad 'time_delays' with zeroes to length 5
        while len(time_delays) < 5:
            time_delays.append(0)

        y[word].append(time_delays)

plt.title("Time delays by word")
n = 1
x = np.arange(len(ts) - n)
cnt = 0
for w in y:
    cnt += 1

    time_delays = np.array(y[w])
    assert type(time_delays) == np.ndarray
    min_time_delays = np.min(time_delays, axis=0)
    max_time_delays = np.max(time_delays, axis=0)
    mean_time_delays = np.mean(time_delays, axis=0)
    std_dev_time_delays = np.std(time_delays, axis=0)
    #range_time_delays = np.ptp(time_delays, axis=0)
    error_bars = [[mean_time_delays[i] - min_time_delays[i] for i in range(np.shape(time_delays)[1])],
                  [max_time_delays[i] - mean_time_delays[i] for i in range(np.shape(time_delays)[1])]]

    plt.errorbar(x, mean_time_delays, yerr=error_bars, label=w)
    #plt.errorbar(x, mean_time_delays, label=w)
    plt.scatter(x, mean_time_delays)

    # Annotate the plot with which keys were pressed
    ax = plt.gca()
    for i in range(len(w) - n):
        ax.annotate(w[i] + " -> " + w[i+1], (i - 0.25, mean_time_delays[i] - 5))
        #ax.annotate(w, (i - 0.15, mean_time_delays[i] - 10))

    plt.ylim([0, 300])
    plt.xlim([-1, 5])
    plt.ylabel("Time interval (ms)")
    plt.legend()

    if cnt % 3 == 0:
        plt.show()

    # plt.savefig(os.path.join(DIR_OUT, SUBDIR_OUT, word + ".jpg"))
