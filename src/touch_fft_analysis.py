import os
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

dir_in = "../features"
keyboard_type = "mechanical"

df = pd.read_csv(os.path.join(dir_in, keyboard_type, "touch_fft.csv"))

labels = set(df["key"])
labels.remove("f")
labels.remove("a")

df.set_index("key", inplace=True)

legend = []

for label in labels:
    x = np.array(df.loc[label, :])
    x_mean = np.mean(x, axis=0)

    freq_bins = np.load(os.path.join(dir_in, keyboard_type, "freq_bins_metadata.npy"))
    print("FREQ_BINS")
    print(freq_bins)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.plot(freq_bins, x_mean)
    legend.append(label)

plt.legend(legend)
plt.title("Average of the Normalized FFTs for Each Key")
plt.show()


