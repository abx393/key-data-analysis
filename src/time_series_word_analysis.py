import os
import numpy as np

from json import JSONDecoder
import matplotlib.pyplot as plt

DIR_IN = "web_raw_data"
SUBDIR_IN = "time_series"
DIR_OUT = "plots"
SUBDIR_OUT = "time_series"

for f in os.listdir(os.path.join(DIR_IN, SUBDIR_IN)):
    (basename, extension) = f.split(".")

    if extension == "json":
        labels_file = open(os.path.join(DIR_IN, SUBDIR_IN, f))
        labels = JSONDecoder().decode(labels_file.read())

        t_prev = 0
        ts = np.asarray(list(labels.keys()), dtype=np.int)
        keys = np.array(list(labels.values()))

        # Plot the 1st difference of the list of timestamps
        n = 1
        x = np.arange(len(ts) - n)
        y = np.diff(ts, n=n)

        plt.plot(x, y)
        plt.scatter(x, y)

        # Annotate the plot with which keys were pressed
        ax = plt.gca()
        for i in range(len(keys) - n):
            ax.annotate(keys[i] + " -> " + keys[i+1], (i - 0.2, y[i] - 30))

        word = ""
        for key in keys:
            word += key

        plt.title(word)
        plt.ylim([0, 300])
        plt.ylabel("Time interval (ms)")
        plt.savefig(os.path.join(DIR_OUT, SUBDIR_OUT, word + ".jpg"))
        plt.show()
