"""
Uses t-SNE dimensionality reduction to visualize raw, high-dimensional feature vectors.
ie., reduces 128-dimensional VGGish feature embeddings to k=2 or k=3 dimensions.
"""

import os
import numpy as np
import pandas as pd

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Dimension of the embedded space
dim = 2

# Dimensionality reduction algorithm used
algorithm = "PCA"

dir_in = "../features/"
keyboard_type = "HP_Spectre"

input = "push_fft"
#index_col = "word"
index_col = "key"

#df = pd.read_csv(os.path.join(dir_in, keyboard_type, input + ".csv"))
df = pd.read_csv(os.path.join(dir_in, keyboard_type, input + ".csv"))
print(df.head())

labels = set(df.iloc[:, 0])
print("labels ", labels)
# labels = ["mouse_click", "mouse_scroll", "a"]
df.set_index(index_col, inplace=True)
legend = []

if dim == 3:
    ax = plt.axes(projection="3d")

for label in labels:

    if label == "a":
        legend.append("key")
    else:
        legend.append(label)

    if input == "push_fft":
        cols = ["freq_bin_" + str(i) for i in range(1, df.shape[1], 10)]
        x = np.array(df.loc[label, cols])
    elif input == "vggish_embeddings":
        x = np.array(df.loc[label])
    elif input == "time_delays":
        x = np.array(df.loc[label])

    print(label)
    print(x.shape)
    print()

    # First reduce high-dimensional data with PCA before applying TSNE
    x = np.array(df.loc[label, :])
    #x = PCA(n_components=50).fit_transform(x)

    if algorithm ==  "PCA":
        x_embedded = PCA(n_components=dim).fit_transform(x)
    elif algorithm == "t-SNE":
        x_embedded = TSNE(n_components=dim).fit_transform(x)

    if dim == 1:
        plt.scatter(x_embedded, np.zeros(x_embedded.shape))
    elif dim == 2:
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
    elif dim == 3:
        ax.scatter3D(x_embedded[:, 0], x_embedded[:, 1], x_embedded[:, 2])

plt.legend(legend)
plt.title(algorithm + ": Dimensionality Reduction of Feature Embeddings")
plt.show()
print(x_embedded.shape)
