"""
Uses t-SNE dimensionality reduction to visualize VGGish feature embeddings.
Reduces 128-dimensional VGGish feature embeddings to k=2 or k=3 dimensions.
"""

import os
import numpy as np
import pandas as pd

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

dir_in = "../features"
keyboard_type = "mechanical"

df = pd.read_csv(os.path.join(dir_in, keyboard_type, "vggish_embeddings.csv"))
print(df.head())

labels = set(df.iloc[:, 0])
df.set_index("key", inplace=True)
legend = []

ax = plt.axes(projection="3d")
for label in labels:
    legend.append(label)
    x = np.array(df.loc[label, :])
    print(label)
    print(x.shape)
    print()
    x_embedded = TSNE(n_components=3).fit_transform(x)
    ax.scatter3D(x_embedded[:, 0], x_embedded[:, 1], x_embedded[:, 2])

plt.legend(legend)
plt.show()
print(x_embedded.shape)
