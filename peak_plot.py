import numpy as np
import pandas as pd

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

PATH_IN = "features/peaks.csv"

df = pd.read_csv(PATH_IN)
keys = set(np.array(df.key))
keys = ["a", "b", "c"]

df.set_index("key", inplace=True)

ax = plt.axes(projection='3d')
for key in keys:
    peak1 = np.array(df.loc[key, "peak_1"])
    peak2 = np.array(df.loc[key, "peak_2"])
    peak3 = np.array(df.loc[key, "peak_3"])

    ax.scatter3D(peak1, peak2, peak3)
    plt.xlabel("peak1")
    plt.ylabel("peak2")
    #plt.zlabel("peak3")
    plt.legend(["key={}".format(key)])

plt.show()