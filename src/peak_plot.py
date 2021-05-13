"""
Draw a 3D plot of peaks
ie., x-axis = peak1, y-axis = peak2, z-axis = peak3
"""

import os
import numpy as np
import pandas as pd

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

DIR_IN = "features"
KEYBOARD_TYPE = "mechanical"

df = pd.read_csv(os.path.join(DIR_IN, KEYBOARD_TYPE, "peaks.csv"))
keys = set(np.array(df.key))
keys = ["space", "backspace", "a"]

df.set_index("key", inplace=True)

ax = plt.axes(projection='3d')
legend = []
for key in keys:
    peak1 = np.array(df.loc[key, "peak_1"])
    peak2 = np.array(df.loc[key, "peak_2"])
    peak3 = np.array(df.loc[key, "peak_3"])

    ax.scatter3D(peak1, peak2, peak3)
    plt.xlabel("peak1")
    plt.ylabel("peak2")
    #plt.zlabel("peak3")
    legend.append("key={}".format(key))

plt.legend(legend)
plt.show()